import numpy as np

data = np.load("datasets/hover_dataset.npz")   # open the bundle
T = data["T"]                                # pull out the time vector
X = data["X"]                                # pull out the trajectories

print("Loaded T shape:", T.shape)            # (500,)
print("Loaded X shape:", X.shape)            # (50, 500, 12)

# Look at ONE clip (the first one), its FIRST frame and LAST frame:
print("\nClip 0, frame 0 (start):", np.round(X[0, 0, :], 3))
print("Clip 0, frame -1 (end): ", np.round(X[0, -1, :], 3))