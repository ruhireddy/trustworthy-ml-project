# mia_strong.py
"""
Stronger Membership Inference Attack (feature-rich + MLP attacker).
Produces balanced train/test for attacker and prints metrics + advantage.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import utils

# ---------------- config ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
INCLUDE_LABEL_ONEHOT = True   # set False if you want label-agnostic attack
RANDOM_SEED = 42
TEST_SIZE = 0.2               # fraction for attacker test set
RANDOM_STATE = 42
MLP_HIDDEN = (128, 64)
MLP_MAX_ITERS = 500
# ----------------------------------------

print("Device:", DEVICE)

# load model
model_fp = "./target_model.pt"
model, fg = utils.load_model(model_fp, device=DEVICE)
print("Loaded model from", model_fp)

# make loaders
train_loader = utils.make_loader('./data/train.npz', 'train_x', 'train_y', batch_size=BATCH_SIZE, shuffle=False)
val_loader   = utils.make_loader('./data/valtest.npz', 'val_x', 'val_y', batch_size=BATCH_SIZE, shuffle=False)

def extract_features_and_labels(loader, tag):
    """
    For each sample produce:
      - softmax probabilities (num_classes)
      - entropy
      - top1 confidence
      - top2 margin
      - per-sample loss (cross-entropy with true label)
      - optionally one-hot label vector
    Returns: features (N x D), labels (actual class), membership_label (1 for train-members, 0 for non-members)
    """
    model.eval()
    feats_list = []
    true_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)                  # (N, C)
            probs = F.softmax(logits, dim=1)    # (N, C)
            probs_np = probs.cpu().numpy()
            logits_np = logits.cpu().numpy()
            # entropy
            entropy = -np.sum(probs_np * np.log(probs_np + 1e-12), axis=1)  # (N,)
            max_conf = np.max(probs_np, axis=1)
            sorted_probs = np.sort(probs_np, axis=1)
            top2_margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            # per-sample loss (using true label)
            # compute negative log prob of true label
            true_label_indices = yb.cpu().numpy().astype(int)
            neglogprob = -np.log(probs_np[np.arange(len(probs_np)), true_label_indices] + 1e-12)
            # construct feature vector: [probs..., entropy, max_conf, top2_margin, neglogprob, (optionally one-hot label)]
            if INCLUDE_LABEL_ONEHOT:
                # one-hot label vector of length C
                num_classes = probs_np.shape[1]
                labels_onehot = np.zeros((probs_np.shape[0], num_classes), dtype=float)
                labels_onehot[np.arange(probs_np.shape[0]), true_label_indices] = 1.0
                feat = np.concatenate([probs_np, entropy.reshape(-1,1), max_conf.reshape(-1,1),
                                       top2_margin.reshape(-1,1), neglogprob.reshape(-1,1), labels_onehot], axis=1)
            else:
                feat = np.concatenate([probs_np, entropy.reshape(-1,1), max_conf.reshape(-1,1),
                                       top2_margin.reshape(-1,1), neglogprob.reshape(-1,1)], axis=1)

            feats_list.append(feat)
            true_labels.append(true_label_indices)
    if len(feats_list) == 0:
        return np.zeros((0, feat.shape[1])), np.zeros((0,), dtype=int)
    X = np.vstack(feats_list)
    y = np.concatenate(true_labels)
    return X, y

print("Extracting features for TRAIN (members)...")
X_train_feats, y_train_labels = extract_features_and_labels(train_loader, tag="train")
print("Extracting features for VAL (non-members)...")
X_val_feats, y_val_labels = extract_features_and_labels(val_loader, tag="val")

print("Shapes:", X_train_feats.shape, X_val_feats.shape)

# membership labels
m_train = np.ones(X_train_feats.shape[0], dtype=int)
m_val   = np.zeros(X_val_feats.shape[0], dtype=int)

# Balance dataset by downsampling larger class to size of smaller
n_train = X_train_feats.shape[0]
n_val = X_val_feats.shape[0]
n_target = min(n_train, n_val)

# sample indices
rng = np.random.RandomState(RANDOM_SEED)
train_idx = rng.choice(n_train, n_target, replace=False) if n_train > n_target else rng.choice(n_train, n_target, replace=True)
val_idx   = rng.choice(n_val, n_target, replace=False)   if n_val > n_target   else rng.choice(n_val, n_target, replace=True)

X_bal = np.vstack([X_train_feats[train_idx], X_val_feats[val_idx]])
y_bal = np.concatenate([m_train[train_idx], m_val[val_idx]])

print("Balanced attacker dataset shape:", X_bal.shape)

# train/test split for attacker
X_tr, X_te, y_tr, y_te = train_test_split(X_bal, y_bal, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_bal)

# standardize features
scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)

# MLP attacker
mlp = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITERS, random_state=RANDOM_STATE, verbose=False)
print("Training MLP attacker...")
mlp.fit(X_tr_s, y_tr)

# evaluate
y_pred = mlp.predict(X_te_s)
acc = accuracy_score(y_te, y_pred)
prec = precision_score(y_te, y_pred)
rec = recall_score(y_te, y_pred)
cm = confusion_matrix(y_te, y_pred)
tn, fp, fn, tp = cm.ravel()
tpr = tp / (tp + fn + 1e-12)
fpr = fp / (fp + tn + 1e-12)
adv = tpr - fpr

print("\n----- Strong MIA Results -----")
print(f"Attacker accuracy: {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall (TPR): {rec*100:.2f}%")
print(f"Advantage (TPR-FPR): {adv:.3f}")
print("Confusion matrix (tn, fp, fn, tp):", (tn, fp, fn, tp))
print("------------------------------")

# save model + scaler
joblib.dump({"mlp": mlp, "scaler": scaler}, "mia_strong_attack.joblib")
print("Saved mia_strong_attack.joblib")
