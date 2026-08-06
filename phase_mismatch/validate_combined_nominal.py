import numpy as np
base = np.load("datasets/spiral_dataset.npz")
X_true = base["X"][40:50]
nom = np.load("datasets/mismatch/combined_worst_0.npz")
diff = np.abs(X_true - nom["X"])
print("max abs difference: ", diff.max())
print("mean abs difference:", diff.mean())