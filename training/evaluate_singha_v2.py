"""Compare v1 (2x64) vs v2 (3x128) observer on Singha scenario."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch
import matplotlib.pyplot as plt
from models.pinn_observer import PINNObserver

NOISE_STD = 0.02; TRAJ_IDX = 0
STATE_INFO = [("x","m"),("y","m"),("z","m"),
    ("x_dot","m/s"),("y_dot","m/s"),("z_dot","m/s"),
    ("phi","rad"),("theta","rad"),("psi (yaw)","rad"),
    ("phi_dot","rad/s"),("theta_dot","rad/s"),("psi_dot","rad/s")]

data = np.load("data/singha_dataset.npz")
T = data["T"]; X = data["X"]
x_true = X[TRAJ_IDX]
rng = np.random.default_rng(0)
y_noisy = x_true + rng.normal(0, NOISE_STD, x_true.shape)
t_t = torch.tensor(T.reshape(-1,1), dtype=torch.float32)
y_t = torch.tensor(y_noisy, dtype=torch.float32)

# v1: small network
m1 = PINNObserver(hidden=64, n_hidden_layers=2)
m1.load_state_dict(torch.load("models/pinn_singha.pth")); m1.eval()
# v2: big network
m2 = PINNObserver(hidden=128, n_hidden_layers=3)
m2.load_state_dict(torch.load("models/pinn_singha_v2.pth")); m2.eval()
with torch.no_grad():
    est1 = m1(t_t, y_t).numpy()
    est2 = m2(t_t, y_t).numpy()

fig, axes = plt.subplots(4, 3, figsize=(15, 12)); axes = axes.flatten()
for i,(name,unit) in enumerate(STATE_INFO):
    ax = axes[i]
    ax.plot(T, x_true[:,i], 'g-', lw=2.2, label="True")
    ax.plot(T, est1[:,i], 'r--', lw=1, alpha=0.6, label="v1 (2x64)")
    ax.plot(T, est2[:,i], 'b-', lw=1.2, label="v2 (3x128)")
    ax.set_title(f"{name} [{unit}]", fontsize=11); ax.grid(alpha=0.3)
    if i==0: ax.legend(fontsize=8)
fig.suptitle("Singha: v1 (small) vs v2 (big network) vs truth", fontsize=15)
fig.tight_layout(rect=[0,0,1,0.98])
os.makedirs("docs", exist_ok=True)
plt.savefig("docs/eval_singha_v2.png", dpi=110, bbox_inches="tight")
print("Saved -> docs/eval_singha_v2.png")