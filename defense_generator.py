#!/usr/bin/python3
# defense_generator.py
# CPU-safe PyTorch implementation:
#  - train_surrogate(...) => trains a differentiable attacker proxy on top-k features
#  - train_generator(...) => trains generator G(p) to produce small perturbations that fool attacker
#  - defended_predict_with_generator(...) => runtime wrapper to use generator at inference

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Optional
import utils


# ---- Hyperparameters (defaults) ----
DEFAULT_TOPK = 10
SURROGATE_HIDDEN = 128
GEN_HIDDEN = 128
SURROGATE_LR = 1e-3
GEN_LR = 1e-3
SURROGATE_EPOCHS = 20
GEN_EPOCHS = 30
BATCH_SIZE = 256
MIN_PROB = 1e-8


# --------------------------
# Model definitions
# --------------------------
"""A small MLP that predicts membership probability from top-k features."""
class SurrogateMLP(nn.Module):
    def __init__(self, inp_dim: int, hidden: int = SURROGATE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max(32, hidden // 2)),
            nn.ReLU(),
            nn.Linear(max(32, hidden // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns probability (0..1) shape (B,)
        x = self.net(x)
        return torch.sigmoid(x).squeeze(-1)


"""Generator G: input full prob vector (C), outputs raw delta (C). We will bound delta via tanh * eps."""
class Generator(nn.Module):
    def __init__(self, inp_dim: int, hidden: int = GEN_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, inp_dim),
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        # returns raw delta (unbounded). We'll scale with tanh outside.
        return self.net(p)


# --------------------------
# Helpers
# --------------------------
"""
probs_np: (N, C) numpy array
returns: (N, topk) sorted descending probabilities
"""
def topk_sorted_probs_from_full(probs_np: np.ndarray, topk: int = DEFAULT_TOPK) -> np.ndarray:
    idx = np.argsort(-probs_np, axis=1)[:, :topk]
    topk_vals = np.take_along_axis(probs_np, idx, axis=1)
    return topk_vals

"""
    Compute full C-dim probability vectors for the given dataset tensor_all using predict_fn.
    predict_fn(batch, device) returns logits (torch.Tensor). We'll softmax to get probs.
    Returns numpy array (N, C).
"""
def compute_probs_for_dataset(predict_fn, tensor_all: torch.Tensor, device: str, batch_size: int = 256) -> np.ndarray:
    probs_list = []
    device_local = device
    n = tensor_all.shape[0]
    for i in range(0, n, batch_size):
        batch = tensor_all[i: i + batch_size]
        with torch.no_grad():
            logits = predict_fn(batch.to(device_local), device_local)  # logits (B, C)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
    return np.concatenate(probs_list, axis=0)


# --------------------------
# Surrogate training
# --------------------------
"""
Train surrogate attacker on features (N, topk) and binary labels (N,) {0,1}.
Returns trained SurrogateMLP on CPU.
"""
def train_surrogate(features_np: np.ndarray, labels_np: np.ndarray,
                    hidden: int = SURROGATE_HIDDEN,
                    epochs: int = SURROGATE_EPOCHS,
                    bs: int = BATCH_SIZE,
                    lr: float = SURROGATE_LR,
                    device: str = "cpu") -> SurrogateMLP:

    device = torch.device(device)
    model = SurrogateMLP(features_np.shape[1], hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    ds = torch.utils.data.TensorDataset(torch.from_numpy(features_np).float(), torch.from_numpy(labels_np).float())
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        total_n = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)
        avg = total_loss / (total_n + 1e-12)
        print(f"[Surrogate] Epoch {ep+1}/{epochs} - loss {avg:.6f}")
    model.eval()
    return model


# --------------------------
# Generator training
# --------------------------
"""
    Train generator G to produce delta bounded by eps (via tanh) that causes surrogate to predict non-member.
    shadow_probs_full: np.ndarray (N, C) full probability vectors from shadow queries
    shadow_labels: np.ndarray (N,) 0/1 membership labels (used only for logging, training forces surrogate -> 0)
    surrogate_uses_topk: if True, surrogate was trained on topk sorted features; we must feed it same.
"""
def train_generator(generator: Generator,
                    surrogate: SurrogateMLP,
                    shadow_probs_full: np.ndarray,
                    shadow_labels: np.ndarray,
                    lambda_preserve: float = 1.0,
                    eps: float = 0.08,
                    lambda_attack: float = 1.0,
                    lambda_util: float = 1.0,
                    epochs: int = GEN_EPOCHS,
                    bs: int = BATCH_SIZE,
                    lr: float = GEN_LR,
                    device: str = "cpu",
                    surrogate_uses_topk: bool = True,
                    topk: int = DEFAULT_TOPK) -> Generator:

    device = torch.device(device)
    generator = generator.to(device)
    surrogate = surrogate.to(device)
    opt = optim.Adam(generator.parameters(), lr=lr)
    bce = nn.BCELoss(reduction='mean')

    N = shadow_probs_full.shape[0]
    indices = np.arange(N)

    # Precompute surrogate input representation if that helps in speed? We will compute on-the-fly.
    for ep in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0.0
        total_n = 0
        for i in range(0, N, bs):
            idx = indices[i:i+bs]
            p_batch_np = shadow_probs_full[idx]  # (B, C)
            p_batch = torch.from_numpy(p_batch_np).float().to(device)

            # forward through generator
            raw = generator(p_batch)                       # (B, C)
            delta = eps * torch.tanh(raw)                  # bound to [-eps, eps]
            p_new = p_batch + delta
            p_new = torch.clamp(p_new, MIN_PROB, 1.0)
            p_new = p_new / p_new.sum(dim=1, keepdim=True)

            # ----- Prepare surrogate input -----
            if surrogate_uses_topk:
                p_new_np = p_new.detach().cpu().numpy()
                feats = topk_sorted_probs_from_full(p_new_np, topk=topk)
                feats_t = torch.from_numpy(feats).float().to(device)
                pred_member_prob = surrogate(feats_t)     # (B,)
            else:
                pred_member_prob = surrogate(p_new)

            # ----- Compute losses -----
            target_zero = torch.zeros_like(pred_member_prob).to(device)
            attack_loss = bce(pred_member_prob, target_zero)

            # KL divergence between original and defended probs
            kl = F.kl_div(torch.log(p_new + 1e-12), p_batch, reduction='batchmean')

            # Argmax-preserve term
            orig_labels = torch.argmax(p_batch, dim=1)
            log_p_new = torch.log(p_new + 1e-12)
            argmax_preserve_loss = -log_p_new[torch.arange(p_new.size(0)), orig_labels].mean()

            # ----- Combine all losses -----
            loss = (
                lambda_attack * attack_loss
                + lambda_util * kl
                + lambda_preserve * argmax_preserve_loss
            )

            # ----- Optimize -----
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * p_batch.size(0)
            total_n += p_batch.size(0)

        avg_loss = total_loss / (total_n + 1e-12)
        print(f"[Generator] Epoch {ep+1}/{epochs} - loss {avg_loss:.6f} (eps={eps} lambda_a={lambda_attack} lambda_u={lambda_util})")

    generator.eval()
    return generator


# --------------------------
# Save / load helpers
# --------------------------
def save_model_torch(model: nn.Module, fp: str):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    torch.save(model.state_dict(), fp)
    print(f"Saved model to {fp}")


def load_surrogate(fp: str, inp_dim: int, hidden: int = SURROGATE_HIDDEN, device: str = "cpu") -> SurrogateMLP:
    model = SurrogateMLP(inp_dim, hidden)
    model.load_state_dict(torch.load(fp, map_location=device))
    model.eval()
    return model


def load_generator(fp: str, inp_dim: int, hidden: int = GEN_HIDDEN, device: str = "cpu") -> Generator:
    model = Generator(inp_dim, hidden)
    model.load_state_dict(torch.load(fp, map_location=device))
    model.eval()
    return model


# --------------------------
# Deployment wrapper (runtime)
# --------------------------
import diffpure_utils
from diffpure_utils import SDE_Adv_Model, parse_args_and_config, Logger

@torch.no_grad()
def diffpure_predict(model, x, device="cuda"):
    
    args, config = parse_args_and_config()
    
    args.classifier = model
    
    log_dir = os.path.join(args.image_folder, args.classifier_name,
                           'seed' + str(args.seed), 'data' + str(args.data_seed))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    logger = Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)

    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size * ngpus
    # print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')

    # load model
    # print('starting the model and loader...')
    new_model = SDE_Adv_Model(args, config, classifier=model)
    if ngpus > 1:
        new_model = torch.nn.DataParallel(new_model)
    new_model = new_model.eval().to(config.device)
    
    
    # --- SDE-based prediction (DiffPure path) ---
    # print('starting the prediction with SDE...')
    x = x.to(config.device if hasattr(config, "device") else device)
    out = new_model(x)  # SDE_Adv_Model does purification + classification
    logits = out[0] if isinstance(out, (tuple, list)) else out
    
    logger.close()
    return logits

"""
Given target model and input batch x (torch.Tensor), returns defended logits.
    - model: returns logits
    - generator: maps probs -> raw delta; we apply tanh*eps to bound it
    - returns logits (torch.Tensor) corresponding to log(p')
"""
@torch.no_grad()
def defended_predict_with_generator(model: torch.nn.Module,
                                    x: torch.Tensor,
                                    generator: Generator, p_rand=0.3,
                                    eps: float = 0.08,
                                    device: str = "cpu") -> torch.Tensor:

    device = torch.device(device)
    model = model.to(device)
    x = x.to(device)
    logits = model(x)                      # (B, C)
    probs = F.softmax(logits, dim=1)       # torch (B, C)
    # generator expects probs in same device/dtype
    raw = generator(probs)
    if p_rand > 0:
        mask = (torch.rand(raw.shape[0]) < p_rand).to(raw.device)
        noise = 0.02 * torch.randn_like(raw)
        raw[mask] += noise[mask]
    delta = eps * torch.tanh(raw)
    p_new = probs + delta
    p_new = torch.clamp(p_new, MIN_PROB, 1.0)
    p_new = p_new / p_new.sum(dim=1, keepdim=True)
    defended_logits = torch.log(p_new)
    return defended_logits


@torch.no_grad()
def memgard_diffpure(model: torch.nn.Module,
                                    x: torch.Tensor,
                                    generator: Generator, p_rand=0.3,
                                    eps: float = 0.08,
                                    device: str = "cpu") -> torch.Tensor:
    
    args, config = parse_args_and_config()
    device = torch.device(device)
    
    args.classifier = model
    
    log_dir = os.path.join(args.image_folder, args.classifier_name,
                           'seed' + str(args.seed), 'data' + str(args.data_seed))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    logger = Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)

    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size * ngpus

    # load model
    new_model = SDE_Adv_Model(args, config, classifier=model)
    if ngpus > 1:
        new_model = torch.nn.DataParallel(new_model)
    new_model = new_model.eval().to(config.device)
    
    
    # --- SDE-based prediction (DiffPure path) ---
    x = x.to(config.device if hasattr(config, "device") else device)
    # out = new_model(x)  # SDE_Adv_Model does purification + classification
    # logits = out[0] if isinstance(out, (tuple, list)) else out
    
    logits = new_model(x)                      # (B, C)
    probs = F.softmax(logits, dim=1)       # torch (B, C)
    
    # generator expects probs in same device/dtype
    raw = generator(probs)
    if p_rand > 0:
        mask = (torch.rand(raw.shape[0]) < p_rand).to(raw.device)
        noise = 0.02 * torch.randn_like(raw)
        raw[mask] += noise[mask]
    delta = eps * torch.tanh(raw)
    p_new = probs + delta
    p_new = torch.clamp(p_new, MIN_PROB, 1.0)
    p_new = p_new / p_new.sum(dim=1, keepdim=True)
    defended_logits = torch.log(p_new)

    logger.close()

    return defended_logits


# --------------------------
# Utilities to prepare training data from shadow loaders or npz
# --------------------------
"""
    Use predict_fn to compute full prob vectors for tensor_all (torch.Tensor).
    This returns (full_probs_np, topk_feats_np).
    Note: membership labels must be provided separately by the caller.
"""
def prepare_shadow_dataset_from_loader(predict_fn, tensor_all: torch.Tensor,
                                       topk: int = DEFAULT_TOPK,
                                       device: str = "cpu",
                                       batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:

    full_probs = compute_probs_for_dataset(predict_fn, tensor_all, device=device, batch_size=batch_size)
    topk_feats = topk_sorted_probs_from_full(full_probs, topk=topk)
    return full_probs, topk_feats


# --------------------------
# Quick evaluators
# --------------------------
"""
Run a trained (non-differentiable) attack_model on the given top-k features.
Returns membership preds (N, 1).
"""
def evaluate_attack_on_probs(attack_model, probs_np: np.ndarray) -> np.ndarray:
    feats = topk_sorted_probs_from_full(probs_np, topk=DEFAULT_TOPK)
    preds = utils.predict_membership_with_trained_attack(attack_model, feats)
    return np.array(preds).reshape(-1, 1)


# --------------------------
# CLI / Example flows
# --------------------------
"""
Example pipeline that reads shadow data from an npz with keys:
    - 'probs' : (N, C) full probability vectors produced by target-like models on shadow inputs
    - 'labels': (N,) membership labels 0/1 for those probs (1=member)
    If you have only shadow input images, use prepare_shadow_dataset_from_loader() instead.
"""
def example_train_from_shadow_npz(shadow_npz_path: str,
                                  attack_model_path: Optional[str],
                                  surrogate_out_fp: str,
                                  generator_out_fp: str,
                                  topk: int = DEFAULT_TOPK,
                                  device: str = "cpu"):

    data = np.load(shadow_npz_path)
    assert 'probs' in data and 'labels' in data, "shadow npz must contain 'probs' and 'labels'"
    shadow_probs = data['probs']  # (N, C)
    shadow_labels = data['labels'].astype(np.int64).reshape(-1)

    # prepare surrogate features (topk)
    topk_feats = topk_sorted_probs_from_full(shadow_probs, topk=topk)

    print("Training surrogate on shadow top-k features...")
    surrogate = train_surrogate(topk_feats, shadow_labels, hidden=SURROGATE_HIDDEN,
                                epochs=SURROGATE_EPOCHS, bs=BATCH_SIZE, lr=SURROGATE_LR, device=device)
    save_model_torch(surrogate, surrogate_out_fp)

    print("Training generator (using surrogate) ...")
    generator = Generator(inp_dim=shadow_probs.shape[1], hidden=GEN_HIDDEN)
    # load surrogate in eval mode (already in memory)
    generator = train_generator(generator, surrogate, shadow_probs, shadow_labels,
                            eps=0.08, lambda_attack=5.0, lambda_util=0.5, lambda_preserve=1.0,
                            epochs=60, bs=256, lr=1e-3, device='cpu', surrogate_uses_topk=True, topk=10)
    save_model_torch(generator, generator_out_fp)
    print("Training complete. Surrogate and generator saved.")


# --------------------------
# If run as script (simple CLI)
# --------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["example_npz", "train_surrogate_only"], default="example_npz",
                   help="Which flow to run.")
    p.add_argument("--shadow_npz", type=str, default="shadow_data.npz",
                   help="Path to shadow npz containing 'probs' and 'labels' if using example_npz mode.")
    p.add_argument("--attack_model", type=str, default=None,
                   help="Path to existing attack model (optional - used only for evaluation/debug).")
    p.add_argument("--surrogate_out", type=str, default="./out/surrogate.pth")
    p.add_argument("--generator_out", type=str, default="./out/generator.pth")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    args = p.parse_args()

    if args.mode == "example_npz":
        if not os.path.exists(args.shadow_npz):
            print("Example shadow npz not found. Create an npz with keys 'probs' and 'labels' and rerun.")
            print("You can generate this with your shadow models using compute_probs_for_dataset(...)")
            exit(0)
        example_train_from_shadow_npz(args.shadow_npz, args.attack_model, args.surrogate_out, args.generator_out,
                                      topk=args.topk, device=args.device)
    elif args.mode == "train_surrogate_only":
        print("train_surrogate_only not implemented in simple CLI. Use script functions/programmatic API from your main.py.")

