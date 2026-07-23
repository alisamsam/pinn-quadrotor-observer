import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pinn_observer_v5 import PINNObserverV5

STATE_NAMES = ["x","y","z","vx","vy","vz","phi","theta","psi","p","q","r"]
MEAS_NAMES  = ["err_x","err_y","err_z","err_phi","err_theta","err_psi"]
FLIGHT = 45

d = np.load("datasets/spiral_dataset.npz")
T, X = d["T"], d["X"]
x0 = X[FLIGHT, 0, :]
Tt  = torch.tensor(T, dtype=torch.float32).unsqueeze(1)
X0t = torch.tensor(np.tile(x0, (len(T), 1)), dtype=torch.float32)

model = PINNObserverV5()
model.load_state_dict(torch.load("phase1a/pinn_phase5.pth", map_location="cpu"))
model.eval()
with torch.no_grad():
    _, L = model.get_state_and_gain(Tt, X0t)
L = L.numpy()

os.makedirs("docs", exist_ok=True)
fig, axes = plt.subplots(4, 3, figsize=(16, 13))
for i, ax in enumerate(axes.flat):
    for j in range(6):
        ax.plot(T, L[:, i, j], linewidth=1.2, label=MEAS_NAMES[j])
    ax.set_title(f"L row for state {STATE_NAMES[i]}")
    ax.grid(alpha=0.3)
    if i == 0: ax.legend(fontsize=8, ncol=2)
fig.suptitle(f"Learned gain L(t) per state - Phase 5 model, unseen spiral flight {FLIGHT}", fontsize=15)
fig.tight_layout()
fig.savefig("docs/L_behavior_phase5.png", dpi=140, bbox_inches="tight")
plt.close(fig)

print("Mean |L| per STATE (row):")
for i, nm in enumerate(STATE_NAMES):
    print(f"  {nm:>6}: {np.abs(L[:, i, :]).mean():.3f}")
print()
print("Mean |L| per MEASURED ERROR (column):")
for j, nm in enumerate(MEAS_NAMES):
    print(f"  {nm:>10}: {np.abs(L[:, :, j]).mean():.3f}")
print()
print("Saved docs/L_behavior_phase5.png")