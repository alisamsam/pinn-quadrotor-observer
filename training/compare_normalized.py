"""Side-by-side: old lambda=1 vs normalized model, all 12 states."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from models.pinn_observer import PINNObserver

NOISE_STD = 0.02
TRAJ_IDX = 0

STATE_INFO = [
    ("x","m"),("y","m"),("z","m"),
    ("x_dot","m/s"),("y_dot","m/s"),("z_dot","m/s"),
    ("phi (roll)","rad"),("theta (pitch)","rad"),("psi (yaw)","rad"),
    ("phi_dot","rad/s"),("theta_dot","rad/s"),("psi_dot","rad/s"),
]

data = np.load("datasets/hover_dataset.npz")
T = data["T"]; X = data["X"]
x_true = X[TRAJ_IDX]
rng = np.random.default_rng(0)
y_noisy = x_true + rng.normal(0, NOISE_STD, x_true.shape)
t_t = torch.tensor(T.reshape(-1,1), dtype=torch.float32)
y_t = torch.tensor(y_noisy, dtype=torch.float32)

def run(path):
    m = PINNObserver(); m.load_state_dict(torch.load(path)); m.eval()
    with torch.no_grad():
        return m(t_t, y_t).numpy()

est_old  = run("models/pinn_lam1.0.pth")
est_norm = run("models/pinn_normalized.pth")

fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.flatten()
for i,(name,unit) in enumerate(STATE_INFO):
    ax = axes[i]
    ax.plot(T, x_true[:,i], 'g-', linewidth=2, label="True")
    ax.plot(T, est_old[:,i], 'r--', linewidth=1.1, alpha=0.7, label="Old (unnorm)")
    ax.plot(T, est_norm[:,i], 'b-', linewidth=1.3, label="Normalized")
    ax.set_title(f"{name}  [{unit}]", fontsize=11)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

fig.suptitle("Old vs Normalized observer: all 12 states", fontsize=15)
fig.tight_layout(rect=[0,0,1,0.98])
os.makedirs("docs", exist_ok=True)
plt.savefig("docs/compare_normalized.png", dpi=110, bbox_inches="tight")
print("Saved -> docs/compare_normalized.png")