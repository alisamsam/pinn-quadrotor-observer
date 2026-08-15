"""Per-state observer RMSE, without vs with the learned gain L, on the unseen test flights.
without L = PINNObserverV4 (pinn_4x100_v2.pth); with L = PINNObserverV5 (pinn_4x100_v2_L.pth).
Both 4x100. Averaged over the 10 test flights (flattened). House style; outputs into this folder.
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pinn_observer_v4 import PINNObserverV4
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
NAMES = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
UNITS = ['m','m','m','m/s','m/s','m/s','rad','rad','rad','rad/s','rad/s','rad/s']
HID = [i for i in range(12) if i not in MEAS_IDX]
OFF, ON = "#1C6FB0", "#C0392B"

_, test = load_phase4_data(os.path.join(PROJ, "datasets/spiral_v2_dataset.npz"))
Tt, X0t, Xt = test["T"], test["X0"], test["X"]

m4 = PINNObserverV4(hidden=100, n_hidden_layers=4)
m4.load_state_dict(torch.load(os.path.join(PROJ, "phase1a/pinn_4x100_v2.pth"), map_location="cpu")); m4.eval()
m5 = PINNObserverV5(hidden=100, n_hidden_layers=4)
m5.load_state_dict(torch.load(os.path.join(PROJ, "phase1a/pinn_4x100_v2_L.pth"), map_location="cpu")); m5.eval()
with torch.no_grad():
    est_off = m4(Tt, X0t).numpy()
    est_on, _ = m5.get_state_and_gain(Tt, X0t); est_on = est_on.numpy()
Xt = Xt.numpy()
rmse_off = np.sqrt(((est_off - Xt)**2).mean(axis=0))
rmse_on  = np.sqrt(((est_on  - Xt)**2).mean(axis=0))
print("avg measured  off/on:", rmse_off[MEAS_IDX].mean(), rmse_on[MEAS_IDX].mean())
print("avg hidden    off/on:", rmse_off[HID].mean(),      rmse_on[HID].mean())

# ---- grouped bar (per state) ----
xt = [f"{NAMES[i]}\n[{UNITS[i]}]" + ("\n(H)" if i in HID else "") for i in range(12)]
x = np.arange(12); wbar = 0.4
fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x-wbar/2, rmse_off, wbar, label="without L", color=OFF, edgecolor="white", linewidth=0.5)
b2 = ax.bar(x+wbar/2, rmse_on,  wbar, label="with L",    color=ON,  edgecolor="white", linewidth=0.5)
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.004, f"{b.get_height():.3f}",
            ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=8)
ax.set_xlabel("State  (unit in brackets; H = hidden)", fontsize=11)
ax.set_ylabel("RMSE  (in each state's own unit)", fontsize=11)
ax.set_title("Per-state observer RMSE: without vs with the learned gain L", fontsize=13)
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
ax.legend(loc="upper left"); ax.margins(y=0.15)
plt.tight_layout()
PNG = os.path.join(HERE, "lgain_perstate.png")
plt.savefig(PNG, dpi=140, bbox_inches="tight"); plt.close(fig)
print("saved", PNG)

# ---- CSV ----
with open(os.path.join(HERE, "lgain_perstate_v2.csv"), "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["state","unit","type","rmse_without_L","rmse_with_L"])
    for i in range(12):
        w.writerow([NAMES[i], UNITS[i], "hidden" if i in HID else "measured",
                    f"{rmse_off[i]:.4f}", f"{rmse_on[i]:.4f}"])
print("saved lgain_perstate_v2.csv")
