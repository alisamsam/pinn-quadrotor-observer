import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from pinn_observer_v4 import PINNObserverV4
from data_phase4 import MEAS_IDX

MODEL_PATH = "phase1a/pinn_4x100.pth"
FLIGHT = 45   # same unseen flight as states_12_flight45.png

# --- load the validated model ---
model = PINNObserverV4(hidden=100, n_hidden_layers=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# --- load the TRUE trajectory straight from the dataset (guarantees same parameters) ---
data = np.load("datasets/spiral_dataset.npz")
X = data["X"]           # shape (n_flights, n_steps, 12)
T_all = data["T"] if "T" in data else np.arange(X.shape[1]) * 0.01

x_true = X[FLIGHT]                      # (n_steps, 12), the true state
x0 = x_true[0]                          # initial condition of this flight
T = T_all if T_all.ndim == 1 else T_all[FLIGHT]

# --- observer estimate for this flight ---
Tt  = torch.tensor(T, dtype=torch.float32).unsqueeze(1)
X0t = torch.tensor(np.tile(x0, (len(T), 1)), dtype=torch.float32)
with torch.no_grad():
    x_hat = model(Tt, X0t).numpy()      # (n_steps, 12), the estimate

# --- position RMSE for the caption (x, y, z are indices 0,1,2) ---
pos_rmse = np.sqrt(((x_hat[:, :3] - x_true[:, :3])**2).mean())

# --- 3D figure: true vs estimated path ---
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2],
        color="blue", linewidth=2.5, label="True")
ax.plot(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2],
        color="red", linestyle="--", linewidth=2.0, label="Estimated")
ax.scatter(x_true[0, 0], x_true[0, 1], x_true[0, 2],
           color="green", s=60, marker="*", label="start")

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
ax.set_title(f"Spiral — unseen flight {FLIGHT}: true vs estimated (4×100)")
ax.legend(loc="upper right")

fig.tight_layout()
os.makedirs("docs", exist_ok=True)
fig.savefig("docs/spiral_3d_flight45_4x100.png", dpi=140, bbox_inches="tight")
plt.close(fig)

print(f"Saved docs/spiral_3d_flight45_4x100.png")
print(f"Flight {FLIGHT} position RMSE (x,y,z): {pos_rmse:.4f} m")