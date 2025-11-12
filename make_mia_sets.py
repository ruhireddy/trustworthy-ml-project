# make_mia_sets.py
import numpy as np, os

os.makedirs("./data", exist_ok=True)

dtr = np.load("./data/train.npz")      # expects keys: train_x, train_y
dva = np.load("./data/valtest.npz")    # expects keys: val_x, val_y

train_x, train_y = dtr["train_x"], dtr["train_y"]
val_x, val_y     = dva["val_x"], dva["val_y"]

# Ensure labels are 1-D class indices
if train_y.ndim > 1: train_y = train_y.argmax(1)
if val_y.ndim   > 1: val_y   = val_y.argmax(1)

# Sample sizes (tune smaller if RAM/slow)
rng   = np.random.default_rng(6261)
n_in  = min(10000, len(train_x))   # "members" from TRAIN
n_out = min(10000, len(val_x))     # "nonmembers" from VAL

in_idx  = rng.choice(len(train_x), size=n_in,  replace=False)
out_idx = rng.choice(len(val_x),   size=n_out, replace=False)

members_x    = train_x[in_idx]
members_y    = train_y[in_idx]
nonmembers_x = val_x[out_idx]
nonmembers_y = val_y[out_idx]

# Save with the exact keys part1.py expects
np.savez("./data/members.npz",    members_x=members_x,    members_y=members_y)
np.savez("./data/nonmembers.npz", nonmembers_x=nonmembers_x, nonmembers_y=nonmembers_y)

print("Wrote members.npz:", members_x.shape, "nonmembers.npz:", nonmembers_x.shape)
