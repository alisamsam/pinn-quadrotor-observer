"""Capacity study figure: v1 (2x64) vs v2 (3x128) vs v3 (3x256)."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch
import matplotlib.pyplot as plt
from models.pinn_observer import PINNObserver

NOISE_STD = 0.02; TRAJ_IDX = 0
data = np.load("data/singha_dataset.npz")
T = data["T"]; X = data["X"]
x_true = X[TRAJ_IDX]
rng = np.random.default_rng(0)
y_noisy = x_true + rng.normal(0, NOISE_STD, x_true.shape)
t_t = torch.tensor(T.reshape(-1,1), dtype=torch.float32)
y_t = torch.tensor(y_noisy, dtype=torch.float32)

def run(path, h, nl):
    m = PINNObserver(hidden=h, n_hidden_layers=nl)
    m.load_state_dict(torch.load(path)); m.eval()
    with torch.no_grad():
        return m(t_t, y_t).numpy()

e1 = run("models/pinn_singha.pth", 64, 2)
e2 = run("models/pinn_singha_v2.pth", 128, 3)
e3 = run("models/pinn_singha_v3.pth", 256, 3)

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))

# LEFT: loss comparison (final-epoch values you recorded)
models = ["v1\n2x64", "v2\n3x128", "v3\n3x256"]
total = [1.52, 0.84, 0.64]
phys  = [0.64, 0.54, 0.39]
xpos = np.arange(3); w = 0.35
ax[0].bar(xpos-w/2, total, w, label="Total loss", color="#1C7293")
ax[0].bar(xpos+w/2, phys, w, label="Physics loss", color="#E8915B")
ax[0].set_xticks(xpos); ax[0].set_xticklabels(models)
ax[0].set_ylabel("Final loss"); ax[0].set_title("Loss vs network capacity")
ax[0].legend(); ax[0].grid(axis='y', alpha=0.3)

# RIGHT: z-altitude tracking (the clearest win)
ax[1].plot(T, x_true[:,2], 'g-', lw=3, label="True")
ax[1].plot(T, e1[:,2], 'r--', lw=1.2, alpha=0.7, label="v1 (2x64)")
ax[1].plot(T, e3[:,2], 'b-', lw=1.5, label="v3 (3x256)")
ax[1].axhline(10, color='k', ls=':', alpha=0.4, label="target 10m")
ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("z (m)")
ax[1].set_title("Altitude tracking: small vs large network")
ax[1].legend(); ax[1].grid(alpha=0.3)

fig.suptitle("Network capacity study (Singha scenario)", fontsize=15)
fig.tight_layout()
os.makedirs("docs", exist_ok=True)
plt.savefig("docs/capacity_study.png", dpi=120, bbox_inches="tight")
print("Saved -> docs/capacity_study.png")