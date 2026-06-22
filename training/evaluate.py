"""Step 7 eval: plot true vs noisy vs estimated for one trajectory."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.pinn_observer import PINNObserver

NOISE_STD = 0.02
STATE_IDX = 2          # which state to plot: 2 = z (altitude)
STATE_NAME = "z (altitude)"
TRAJ_IDX = 0           # which trajectory to plot

# ----- load dataset -----
data = np.load("datasets/hover_dataset.npz")
T = data["T"]                      # (500,)
X = data["X"]                      # (50, 500, 12)

# pick one trajectory
x_true = X[TRAJ_IDX]               # (500, 12)

# rebuild the SAME noisy measurement (same seed as training)
rng = np.random.default_rng(0)
# note: we must regenerate noise the same way; simplest is fresh noise here
y_noisy = x_true + rng.normal(0, NOISE_STD, x_true.shape)   # (500, 12)

# ----- load trained model -----
model = PINNObserver()
model.load_state_dict(torch.load("models/pinn_trained.pth"))
model.eval()                       # put network in "evaluation mode"

# ----- run the network on this trajectory -----
t_t = torch.tensor(T.reshape(-1, 1), dtype=torch.float32)   # (500, 1)
y_t = torch.tensor(y_noisy, dtype=torch.float32)            # (500, 12)
with torch.no_grad():              # no gradients needed for plotting
    x_est = model(t_t, y_t).numpy()                         # (500, 12)

# ----- plot one state -----
plt.figure(figsize=(9, 5))
plt.plot(T, x_true[:, STATE_IDX], 'g-',  label="True",      linewidth=2)
plt.plot(T, y_noisy[:, STATE_IDX], 'k.', label="Noisy meas", markersize=2, alpha=0.4)
plt.plot(T, x_est[:, STATE_IDX],  'r--', label="Estimated",  linewidth=1.5)
plt.xlabel("Time (s)")
plt.ylabel(STATE_NAME)
plt.title(f"PINN observer: {STATE_NAME}  (trajectory {TRAJ_IDX})")
plt.legend()
plt.grid(True, alpha=0.3)

os.makedirs("docs", exist_ok=True)
plt.savefig("docs/eval_z.png", dpi=120, bbox_inches="tight")
print("Saved plot -> docs/eval_z.png")