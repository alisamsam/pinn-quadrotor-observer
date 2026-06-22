"""Clean result shown properly: z vs time (the axis that matters)."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib.pyplot as plt
from models.pinn_observer import PINNObserver

NOISE_STD = 0.02; TRAJ_IDX = 0
data = np.load("datasets/setpoint_clean.npz")
T = data["T"]; X = data["X"]
x_true = X[TRAJ_IDX]
rng = np.random.default_rng(0)
y_noisy = x_true + rng.normal(0, NOISE_STD, x_true.shape)
t_t = torch.tensor(T.reshape(-1,1), dtype=torch.float32)
y_t = torch.tensor(y_noisy, dtype=torch.float32)
m = PINNObserver(); m.load_state_dict(torch.load("models/pinn_clean.pth")); m.eval()
with torch.no_grad():
    x_est = m(t_t, y_t).numpy()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, name in zip([0,1,2], ["x (m)","y (m)","z (m)"]):
    ax[i].plot(T, x_true[:,i], 'g-', lw=2, label="True")
    ax[i].plot(T, y_noisy[:,i], 'k.', ms=1.5, alpha=0.2, label="Noisy")
    ax[i].plot(T, x_est[:,i], 'b-', lw=1.2, label="Est")
    ax[i].set_title(name); ax[i].grid(alpha=0.3); ax[i].set_xlabel("t (s)")
    if i==0: ax[i].legend(fontsize=8)
ax[2].axhline(2.0, color='r', ls=':', label="target")
fig.suptitle("Phase 2 (clean): position vs time")
fig.tight_layout()
plt.savefig("docs/clean_position_vs_time.png", dpi=120, bbox_inches="tight")
print("Saved -> docs/clean_position_vs_time.png")