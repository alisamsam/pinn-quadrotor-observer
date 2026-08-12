import numpy as np
data = np.load("datasets/spiral_dataset.npz")
print("Keys:", list(data.keys()))
for k in data.keys():
    print(f"  {k}: shape {data[k].shape}")