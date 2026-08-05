import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import MEAS_IDX

MODEL_PATH = "phase1a/pinn_4x100.pth"
LAYERS, HIDDEN = 4, 100
FLIGHT = 45                      # an unseen test flight (40-49)

STATE_NAMES = ["x","y","z","vx","vy","vz","phi","theta","psi","p","q","r"]
UNITS = ["m","m","m","m/s","m/s","m/s","rad","rad","rad","rad/s","rad/s","rad/s"]
HID = [i for i in range(12) if i not in MEAS_IDX]

# --- load the flight ---
d = np.load("datasets/spiral_dataset.npz")
T, X = d["T"], d["X"]
x_true = X[FLIGHT]                       # (steps, 12) ground truth
x0 = x_true[0]                           # this flight's initial condition

# --- load model and estimate ---
model = PINNObserverV4(hidden=HIDDEN, n_hidden_layers=LAYERS)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

Tt  = torch.tensor(T, dtype=torch.float32).unsqueeze(1)
X0t = torch.tensor(np.tile(x0, (len(T), 1)), dtype=torch.float32)
with torch.no_grad():
    x_hat = model(Tt, X0t).numpy()       # (steps, 12) estimates

# --- per-state RMSE for the panel titles ---
rmse = np.sqrt(((x_hat - x_true)**2).mean(axis=0))

os.makedirs("docs", exist_ok=True)
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
for i, ax in enumerate(axes.flat):
    ax.plot(T, x_true[:, i], color="#0B3C5D", lw=1.8, label="true")
    ax.plot(T, x_hat[:, i], color="#E08E0B", lw=1.4, ls="--", label="estimate")
    kind = "measured" if i in MEAS_IDX else "hidden"
    tcol = "#1C7293" if i in MEAS_IDX else "#C0392B"
    ax.set_title(f"{STATE_NAMES[i]} ({kind})   RMSE={rmse[i]:.3f}", color=tcol, fontsize=11)
    ax.set_xlabel("t (s)"); ax.set_ylabel(UNITS[i]); ax.grid(alpha=0.3)
    if i == 0: ax.legend(loc="upper right", fontsize=9)

fig.suptitle(f"Observer estimate vs true state — 4x100 model, unseen flight {FLIGHT}",
             fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig("docs/states_12_flight45.png", dpi=140, bbox_inches="tight")
plt.close(fig)

print(f"Saved docs/states_12_flight45.png")
print(f"Mean measured RMSE: {rmse[MEAS_IDX].mean():.4f} | mean hidden RMSE: {rmse[HID].mean():.4f}")