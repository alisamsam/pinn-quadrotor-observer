import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pinn_observer_v4 import PINNObserverV4   # no L
from pinn_observer_v5 import PINNObserverV5   # with L

STATE_NAMES = ["x","y","z","vx","vy","vz","phi","theta","psi","p","q","r"]
HIDDEN = [3,4,5,9,10,11]
TEST_FLIGHT = 45

# --- load unseen spiral flight ---
d = np.load("datasets/spiral_dataset.npz")
T, X = d["T"], d["X"]
x_true = X[TEST_FLIGHT]
x0 = x_true[0]
Tt = torch.tensor(T, dtype=torch.float32).unsqueeze(1)
X0t = torch.tensor(np.tile(x0, (len(T), 1)), dtype=torch.float32)

# --- run both models ---
m4 = PINNObserverV4()
m4.load_state_dict(torch.load("phase1a/pinn_phase4.pth", map_location="cpu"))
m4.eval()
with torch.no_grad():
    est_noL = m4(Tt, X0t).numpy()

m5 = PINNObserverV5()
m5.load_state_dict(torch.load("phase1a/pinn_phase5.pth", map_location="cpu"))
m5.eval()
with torch.no_grad():
    xh, _ = m5.get_state_and_gain(Tt, X0t)
    est_L = xh.numpy()

err_noL = np.abs(est_noL - x_true)
err_L   = np.abs(est_L   - x_true)

os.makedirs("docs", exist_ok=True)

# --- Plot 1: error curves, both models, 12 panels ---
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
for i, ax in enumerate(axes.flat):
    ax.plot(T, err_noL[:, i], "b-", label="without L", linewidth=1.5)
    ax.plot(T, err_L[:, i],  "r--", label="with L", linewidth=1.5)
    tag = " (HIDDEN)" if i in HIDDEN else ""
    ax.set_title(f"|error| {STATE_NAMES[i]}{tag}")
    ax.grid(alpha=0.3)
    if i == 0: ax.legend()
fig.suptitle(f"Estimation error: with vs without gain L — unseen spiral flight {TEST_FLIGHT}", fontsize=14)
fig.tight_layout()
fig.savefig("docs/compare_L_errors.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# --- Plot 2: RMSE bar chart ---
rmse_noL = np.sqrt((err_noL**2).mean(axis=0))
rmse_L   = np.sqrt((err_L**2).mean(axis=0))
xpos = np.arange(12)
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(xpos - 0.2, rmse_noL, width=0.4, label="without L", color="tab:blue")
ax.bar(xpos + 0.2, rmse_L,   width=0.4, label="with L",   color="tab:red")
ax.set_xticks(xpos)
ax.set_xticklabels([f"{n}\n(H)" if i in HIDDEN else n for i, n in enumerate(STATE_NAMES)])
ax.set_ylabel("RMSE")
ax.set_title(f"RMSE per state: with vs without L — unseen spiral flight {TEST_FLIGHT}  (H = hidden)")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.savefig("docs/compare_L_rmse.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("Saved docs/compare_L_errors.png and docs/compare_L_rmse.png")

# --- Comparison table ---
print(f"\n{'state':>6} | {'no L':>8} | {'with L':>8} | {'ratio':>6} | {'type'}")
print("-" * 48)
for i, nm in enumerate(STATE_NAMES):
    ratio = rmse_L[i] / rmse_noL[i]
    tag = "HIDDEN" if i in HIDDEN else "measured"
    print(f"{nm:>6} | {rmse_noL[i]:>8.4f} | {rmse_L[i]:>8.4f} | {ratio:>6.2f} | {tag}")

print("-" * 48)
hid, meas = HIDDEN, [i for i in range(12) if i not in HIDDEN]
print(f"avg measured | {rmse_noL[meas].mean():.4f} | {rmse_L[meas].mean():.4f}")
print(f"avg hidden   | {rmse_noL[hid].mean():.4f} | {rmse_L[hid].mean():.4f}")