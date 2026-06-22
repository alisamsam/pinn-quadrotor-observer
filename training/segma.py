import numpy as np
data = np.load("data/singha_dataset.npz")
X = data["X"]                          # (50, 1800, 12)
X_all = X.reshape(-1, 12)              # (90000, 12)
x_column = X_all[:, 0]                 # just the x-position (column 0)

print("mean of x:", x_column.mean())
print("sigma of x (by hand recipe):", np.sqrt(((x_column - x_column.mean())**2).mean()))
print("sigma of x (numpy .std):    ", x_column.std())