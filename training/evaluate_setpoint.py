"""Phase 2 Step 2.7: evaluate observer on controlled setpoint flight."""
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

data = np.load("data/setpoint_dataset.npz")
T = data["T"]; X = data["X"]
x_true = X[TRAJ_IDX]
rng = np.random.default_rng(0)
y_noisy = x_true + rng.normal(0, NOISE_STD, x_true.shape)
t_t = torch.tensor(T.reshape(-1,1), dtype=torch.float32)
y_t = torch.tensor(y_noisy, dtype=torch.float32)

model = PINNObserver()
model.load_state_dict(torch.load("models/pinn_setpoint.pth"))
model.eval()
with torch.no_grad():
    x_est = model(t_t, y_t).numpy()

fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.flatten()
for i,(name,unit) in enumerate(STATE_INFO):
    ax = axes[i]
    ax.plot(T, x_true[:,i], 'g-', linewidth=2, label="True")
    ax.plot(T, y_noisy[:,i], 'k.', markersize=1.5, alpha=0.2, label="Noisy")
    ax.plot(T, x_est[:,i], 'b-', linewidth=1.3, label="Estimated")
    ax.set_title(f"{name}  [{unit}]", fontsize=11)
    ax.grid(True, alpha=0.3)
    if i == 0: ax.legend(fontsize=8)

fig.suptitle("Phase 2: observer on controlled setpoint flight, all 12 states", fontsize=15)
fig.tight_layout(rect=[0,0,1,0.98])
os.makedirs("docs", exist_ok=True)
plt.savefig("docs/eval_setpoint.png", dpi=110, bbox_inches="tight")
print("Saved -> docs/eval_setpoint.png")