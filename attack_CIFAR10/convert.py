import numpy as np
import pandas as pd

df = pd.read_csv("ResNet_pretrained_False_num_shadow_128_CIFAR10.csv")

# Extract columns
prob_cols = [c for c in df.columns if c.startswith("top_")]
shadow_probs = df[prob_cols].to_numpy(dtype=np.float32)
shadow_labels = df["is_member"].astype(int).to_numpy()

# Save in .npz format (used by defense_generator.py)
np.savez("shadow_data.npz", probs=shadow_probs, labels=shadow_labels)
print("Saved shadow_data.npz with shape:", shadow_probs.shape)
