"""Evaluate the retrained observer (pinn_4x100_v2) on the spiral_v2 test flights.
Prints per-state RMSE (measured/hidden) and plots the 12 states (true vs estimated)
for one unseen test flight. Run AFTER training pinn_4x100_v2.pth on the cluster.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX

DATASET = "datasets/spiral_v2_dataset.npz"
MODEL   = "phase1a/pinn_4x100_v2.pth"
NAMES = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
HID = [i for i in range(12) if i not in MEAS_IDX]

if not os.path.exists(MODEL):
    print(f"[!] {MODEL} not found — train it on the cluster first (train_4x100_v2.py).")
    sys.exit(0)

model = PINNObserverV4(hidden=100, n_hidden_layers=4)
model.load_state_dict(torch.load(MODEL, map_location="cpu")); model.eval()

# ---- overall RMSE on the 10 unseen test flights (flattened) ----
_, test = load_phase4_data(DATASET)
with torch.no_grad():
    Xp = model(test["T"], test["X0"]); Xt = test["X"]
    rmse = torch.sqrt(((Xp - Xt)**2).mean(dim=0)).numpy()

print("\n=== TEST (10 unseen flights) per-state RMSE ===")
for i,n in enumerate(NAMES):
    tag = "measured" if i in MEAS_IDX else "HIDDEN"
    print(f"  {n:>6} | {rmse[i]:.4f} | {tag}")
print(f"  avg measured: {rmse[MEAS_IDX].mean():.4f}")
print(f"  avg hidden:   {rmse[HID].mean():.4f}")

# ---- per-state RMSE bar chart (measured vs hidden), units in x-tick labels ----
import matplotlib.patches as mpatches
UNITS = ['m','m','m','m/s','m/s','m/s','rad','rad','rad','rad/s','rad/s','rad/s']
xticklabels = [f"{NAMES[i]}\n[{UNITS[i]}]" for i in range(12)]
colors = ["#E08E0B" if i in MEAS_IDX else "#0B3C5D" for i in range(12)]
figb, axb = plt.subplots(figsize=(12, 6))
bars = axb.bar(range(12), rmse, color=colors, edgecolor="white", linewidth=0.6)
for b, v in zip(bars, rmse):
    axb.annotate(f"{v:.3f}", (b.get_x()+b.get_width()/2, v), xytext=(0,3),
                 textcoords="offset points", ha="center", va="bottom", fontsize=8)
axb.set_xticks(range(12)); axb.set_xticklabels(xticklabels, fontsize=9)
axb.set_xlabel("State  (unit in brackets)", fontsize=11)
axb.set_ylabel("RMSE  (in each state's own unit)", fontsize=11)
axb.set_title("PINN observer per-state RMSE on unseen test flights", fontsize=13)
axb.grid(axis="y", alpha=0.3); axb.set_axisbelow(True)
axb.legend(handles=[mpatches.Patch(color="#E08E0B", label="measured states"),
                    mpatches.Patch(color="#0B3C5D", label="hidden states")],
           loc="upper right")
axb.margins(y=0.15)
outb = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "observer_rmse_bar.png")
figb.tight_layout(); figb.savefig(outb, dpi=140, bbox_inches="tight"); plt.close(figb)
print("saved ->", outb)

# ---- 12-state plot for one unseen flight (index 40) ----
d = np.load(DATASET); T = d["T"]; X = d["X"]; flight = 40
t_col = torch.tensor(T, dtype=torch.float32).unsqueeze(1)
x0_rep = torch.tensor(np.repeat(X[flight,0:1,:], len(T), axis=0), dtype=torch.float32)
with torch.no_grad():
    Xp_f = model(t_col, x0_rep).numpy()
Xt_f = X[flight]

UNITS = ['m','m','m','m/s','m/s','m/s','rad','rad','rad','rad/s','rad/s','rad/s']
fig, axes = plt.subplots(4,3, figsize=(15,11))
for i,ax in enumerate(axes.flat):
    r = float(np.sqrt(((Xp_f[:,i]-Xt_f[:,i])**2).mean()))
    ax.plot(T, Xt_f[:,i], 'b-', lw=1.5, label='true')
    ax.plot(T, Xp_f[:,i], 'r--', lw=1.3, label='estimated')
    tag = "meas" if i in MEAS_IDX else "HID"
    ax.set_title(f"{NAMES[i]} ({tag})  RMSE={r:.3f} {UNITS[i]}", fontsize=10)
    ax.set_xlabel("time  t  (s)", fontsize=9)
    ax.set_ylabel(f"{NAMES[i]}  ({UNITS[i]})", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.suptitle("PINN observer — 12 states on an unseen test flight: true vs estimated", fontsize=13)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "eval_12states_v2.png")
plt.tight_layout(rect=[0,0,1,0.98]); plt.savefig(out, dpi=120, bbox_inches="tight")
print("saved ->", out)
