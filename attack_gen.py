# attack_gen.py
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils
from tqdm import trange

def fgsm_attack(model, x, y, eps_norm, device="cuda"):
    x_adv = x.clone().detach().to(device).requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y.to(device))
    loss.backward()
    grad = x_adv.grad.detach()
    x_adv = x_adv + (eps_norm * grad.sign())
    return x_adv.detach()

def pgd_attack(model, x, y, eps_norm, alpha_norm, iters, device="cuda", rand_start=True):
    x_nat = x.clone().detach().to(device)
    if rand_start:
        x_adv = x_nat + torch.empty_like(x_nat).uniform_(-1,1) * eps_norm
        x_adv = x_adv.detach()
    else:
        x_adv = x_nat.clone().detach()

    for _ in range(iters):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y.to(device))
        loss.backward()
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha_norm * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_nat + eps_norm), x_nat - eps_norm).detach()
    return x_adv.detach()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model_fp = './target_model.pt'
    model, fg = utils.load_model(model_fp, device=device)
    model.eval()

    # grab 2 batches from validation/test (batch_size 256 -> ~512 examples)
    val_loader = utils.make_loader('./data/valtest.npz', 'val_x', 'val_y', batch_size=256, shuffle=False)
    benign_x, benign_y = utils.grab_from_loader(val_loader, num_batches=2)
    print("Benign batch shape:", benign_x.shape, benign_y.shape)

    # PGD / FGSM params (pixel space)
    eps_pixel = 8.0/255.0
    alpha_pixel = 2.0/255.0
    iters = 10

    std = utils.CIFAR10_STD.to(device)
    eps_norm = (eps_pixel / std).view(1,3,1,1).to(device)
    alpha_norm = (alpha_pixel / std).view(1,3,1,1).to(device)

    x_nat = benign_x.to(device)
    y_nat = benign_y.to(device)

    print("Running FGSM...")
    adv_fgsm = fgsm_attack(model, x_nat, y_nat, eps_norm, device=device)

    print("Running PGD (it=%d)..." % iters)
    adv_pgd = pgd_attack(model, x_nat, y_nat, eps_norm, alpha_norm, iters, device=device, rand_start=True)

    with torch.no_grad():
        pred_nat = model(x_nat).argmax(1).cpu().numpy()
        pred_fgsm = model(adv_fgsm).argmax(1).cpu().numpy()
        pred_pgd  = model(adv_pgd).argmax(1).cpu().numpy()
    y_np = y_nat.cpu().numpy()
    acc_nat = (pred_nat == y_np).mean()
    acc_fgsm = (pred_fgsm == y_np).mean()
    acc_pgd  = (pred_pgd == y_np).mean()
    print(f"Benign acc (this subset): {acc_nat:.4f}, FGSM acc: {acc_fgsm:.4f}, PGD acc: {acc_pgd:.4f}")

    # Save stronger PGD adversarial set
    adv_x_np = adv_pgd.cpu().numpy()
    benign_x_np = x_nat.cpu().numpy()
    benign_y_np = y_nat.cpu().numpy()

    out_fp = 'advexp1.npz'
    np.savez_compressed(out_fp, adv_x=adv_x_np, benign_x=benign_x_np, benign_y=benign_y_np)
    print("Saved adversarial file to", out_fp)
    print("Done.")
