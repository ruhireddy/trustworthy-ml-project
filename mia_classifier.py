# mia_classifier.py
"""
Improved Membership Inference Attack (MIA) using balanced data and engineered features.
Uses utils.make_loader(...) to build train/val loaders.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import resample
import joblib
import utils

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load pretrained target model
model_fp = "./target_model.pt"
model, fg = utils.load_model(model_fp, device=device)
print(f"Loaded model from {model_fp}")

# Create data loaders using utils.make_loader
# Adjust batch_size if you want fewer/more examples per batch
batch_size = 512
print("Creating train and val loaders...")
train_loader = utils.make_loader('./data/train.npz', 'train_x', 'train_y', batch_size=batch_size, shuffle=False)
val_loader   = utils.make_loader('./data/valtest.npz', 'val_x', 'val_y', batch_size=batch_size, shuffle=False)

# Optional: sanity-check the loaders
# utils.check_loader(train_loader, verb=0)
# utils.check_loader(val_loader, verb=0)

# Function to extract logits and labels from a loader
def extract_logits_and_labels(loader):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)                   # (N, num_classes)
            all_logits.append(logits.cpu())
            all_labels.append(yb.cpu())
    if len(all_logits) == 0:
        return np.zeros((0,10)), np.zeros((0,), dtype=int)
    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return logits, labels

print("Extracting logits for TRAIN (members)...")
train_logits, train_labels = extract_logits_and_labels(train_loader)
print("Extracting logits for VAL (non-members)...")
val_logits, val_labels = extract_logits_and_labels(val_loader)

# Feature engineering: max confidence, entropy, top2 margin
def compute_features(logits_np):
    logits_t = torch.tensor(logits_np)
    probs = F.softmax(logits_t, dim=1).numpy()
    max_conf = np.max(probs, axis=1)
    # entropy per sample
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    # top2 margin: top1 - top2
    sorted_probs = np.sort(probs, axis=1)
    top2_margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    features = np.vstack([max_conf, entropy, top2_margin]).T
    return features

train_feats = compute_features(train_logits)
val_feats   = compute_features(val_logits)

# Labels for MIA: 1 for member (train), 0 for non-member (val)
y_train_mia = np.ones(len(train_feats), dtype=int)
y_val_mia   = np.zeros(len(val_feats), dtype=int)

# Combine and downsample/upsample to balance classes
X = np.vstack([train_feats, val_feats])
y = np.concatenate([y_train_mia, y_val_mia])

# If classes are imbalanced, resample to balance
n_train = len(train_feats)
n_val = len(val_feats)
n_target = min(n_train, n_val)  # choose balanced size
# sample n_target examples from each class
train_sample_idx = np.random.choice(n_train, n_target, replace=False) if n_train > n_target else np.random.choice(n_train, n_target, replace=True)
val_sample_idx   = np.random.choice(n_val, n_target, replace=False)   if n_val > n_target   else np.random.choice(n_val, n_target, replace=True)

X_bal = np.vstack([train_feats[train_sample_idx], val_feats[val_sample_idx]])
y_bal = np.concatenate([np.ones(n_target, dtype=int), np.zeros(n_target, dtype=int)])

print(f"Balanced dataset size: {X_bal.shape[0]} (members: {n_target}, non-members: {n_target})")

# Shuffle balanced dataset
perm = np.random.RandomState(42).permutation(X_bal.shape[0])
X_bal = X_bal[perm]
y_bal = y_bal[perm]

# Train logistic regression MIA
clf = LogisticRegression(max_iter=2000, solver="liblinear")
clf.fit(X_bal, y_bal)

# Evaluate on the balanced training set (quick check)
y_pred = clf.predict(X_bal)
acc = accuracy_score(y_bal, y_pred)
prec = precision_score(y_bal, y_pred)
rec = recall_score(y_bal, y_pred)
cm = confusion_matrix(y_bal, y_pred)
# compute advantage TPR - FPR
tn, fp, fn, tp = cm.ravel()
tpr = tp / (tp + fn + 1e-12)
fpr = fp / (fp + tn + 1e-12)
adv = tpr - fpr

print("\n------------ Improved MIA Results ------------")
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall:    {rec * 100:.2f}%")
print(f"Advantage: {adv:.3f}")
print("Confusion Matrix:")
print(cm)
print("----------------------------------------------")

# Save attack model
joblib.dump(clf, "mia_logreg_improved.joblib")
print("Saved improved MIA model as mia_logreg_improved.joblib")
