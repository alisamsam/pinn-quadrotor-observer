"""Money figure: true vs noisy vs 3 lambda estimates, one state."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from models.pinn_observer import PINNObserver

NOISE_STD = 0.02
STATE_IDX = 2
STATE_NAME = "z (altitude)"
TRAJ_IDX = 0

MODELS = [
    ("models/pinn_lam0.1.pth", "tab:blue",   "lambda=0.1 (mimic)"),
    ("models/pinn_lam1.0.pth", "tab:orange", "lambda=1.0 (balanced)"),
    ("models/pinn_lam10.0.pth","tab:red",    "lambda=10 (purist)"),
]

data = np.load("datasets/hover_dataset.npz")
T = data["T"]
X = data["X"]
x_true = X[TRAJ_IDX]
rng = np.random.default_rng(0)
y_noisy = x_true + rng.normal(0, NOISE_STD, x_true.shape)

t_t = torch.tensor(T.reshape(-1, 1), dtype=torch.float32)
y_t = torch.tensor(y_noisy, dtype=torch.float32)

plt.figure(figsize=(10, 6))
plt.plot(T, x_true[:, STATE_IDX], 'g-', label="True", linewidth=2.5)
plt.plot(T, y_noisy[:, STATE_IDX], 'k.', label="Noisy meas", markersize=2, alpha=0.3)

for fname, color, label in MODELS:
    model = PINNObserver()
    model.load_state_dict(torch.load(fname))
    model.eval()
    with torch.no_grad():
        x_est = model(t_t, y_t).numpy()
    plt.plot(T, x_est[:, STATE_IDX], '--', color=color, label=label, linewidth=1.8)

plt.xlabel("Time (s)")
plt.ylabel(STATE_NAME)
plt.title(f"PINN observer lambda-sweep: {STATE_NAME}")
plt.legend()
plt.grid(True, alpha=0.3)

os.makedirs("docs", exist_ok=True)
plt.savefig("docs/eval_overlay_z.png", dpi=120, bbox_inches="tight")
print("Saved -> docs/eval_overlay_z.png")