# L_behavior_figs.py -- L-gain behaviour figures from the saved 4x100 checkpoint
# (clear axis labels, plain-English titles). Run FROM REPO ROOT: python L_behavior_figs.py

import sys, os
ROOT = os.getcwd()
for p in [ROOT, os.path.join(ROOT, "phase1a")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

CKPT = "phase1a/pinn_4x100_v2_L.pth"
DATASET = "datasets/spiral_v2_dataset.npz"
FLIGHT, NSTEPS, STEP, OUT = 0, 3000, 25, "paper_figures"
os.makedirs(OUT, exist_ok=True)
NAMES = ["x","y","z","vx","vy","vz","phi","theta","psi","p","q","r"]
MEAS_NAMES = ["e_x","e_y","e_z","e_phi","e_theta","e_psi"]

train, test = load_phase4_data(DATASET)
a = FLIGHT*NSTEPS; b = a+NSTEPS
T = test["T"][a:b]; X = test["X"][a:b]; U = test["U"][a:b]; x0 = test["X0"][a]
model = PINNObserverV5(hidden=100, n_hidden_layers=4)
model.load_state_dict(torch.load(CKPT, map_location="cpu")); model.eval()
with torch.no_grad():
    _, L_all = model.get_state_and_gain(T, x0.view(1,12).repeat(NSTEPS,1))
t = T.view(-1).numpy(); L = L_all.numpy()

# ---- Fig 7: overall size of the gain ----
Lf = np.sqrt((L**2).reshape(NSTEPS,-1).sum(axis=1))
np.savetxt(f"{OUT}/fig_L_norm_timeseries.csv", np.column_stack([t,Lf]),
           delimiter=",", header="t,L_fro", comments="", fmt="%.6g")
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(t, Lf, color="crimson", lw=1.5, label="gain size at each instant")
mval = float(Lf.mean())
ax.axhline(mval, ls="--", color="gray", lw=1.2, label=f"average over flight = {mval:.1f}")
ax.set_xlabel("time  t  (seconds)")
ax.set_ylabel(r"overall size of the gain,  $\|L(t)\|$")
ax.set_title("How big the learned correction gain is over the flight")
ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/spiral_L_norm.pdf"); fig.savefig(f"{OUT}/spiral_L_norm.png", dpi=300); plt.close(fig)

# ---- Fig 8: does the error grow or shrink ----
C = np.zeros((6,12))
for r,c in enumerate(MEAS_IDX): C[r,c] = 1.0
times, maxre = [], []
for k in range(0, NSTEPS, STEP):
    uk = U[k]
    A = torch.autograd.functional.jacobian(
        lambda xx: quadrotor_dynamics_torch(xx.unsqueeze(0), uk.unsqueeze(0)).squeeze(0), X[k]).numpy()
    ev = np.linalg.eigvals(A - L[k] @ C)
    times.append(t[k]); maxre.append(ev.real.max())
times = np.array(times); maxre = np.array(maxre); frac = float((maxre>0).mean())
np.savetxt(f"{OUT}/fig_L_eig_timeseries.csv", np.column_stack([times,maxre]),
           delimiter=",", header="t,max_re_eig", comments="", fmt="%.6g")
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(times, maxre, "r-", lw=1.6, label="error growth rate")
ax.axhline(0, color="k", lw=1)
ax.fill_between(times, 0, maxre, where=(maxre>0), color="red", alpha=0.15,
                label="unstable (rate > 0)")
ax.set_xlabel("time  t  (seconds)")
ax.set_ylabel("error growth rate\n(largest real part of eig(A-LC))")
ax.set_title(f"Does the estimation error grow or shrink?  ({frac*100:.0f}% of flight: growing)")
ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/spiral_L_eig.pdf"); fig.savefig(f"{OUT}/spiral_L_eig.png", dpi=300); plt.close(fig)

# ---- Fig 9: every gain entry over time ----
fig, axes = plt.subplots(4, 3, figsize=(15,10), sharex=True)
fig.suptitle("Every entry of the learned gain L(t) over the flight  "
             "(one panel per corrected state; one line per sensor-error channel)", fontsize=14)
for i in range(12):
    axc = axes.flat[i]
    for j in range(6):
        axc.plot(t, L[:, i, j], lw=1.0, label=MEAS_NAMES[j])
    axc.set_title(f"correcting state: {NAMES[i]}")
    axc.grid(alpha=0.3)
    if i >= 9: axc.set_xlabel("time  t  (s)")
    if i == 0: axc.legend(title="uses sensor error", fontsize=7, ncol=2)
try:
    fig.supylabel(r"gain value  $L_{ij}(t)$  (how strongly each error is used)", fontsize=13)
except Exception:
    for i in (0,3,6,9): axes.flat[i].set_ylabel(r"gain $L_{ij}(t)$")
fig.tight_layout()
fig.savefig(f"{OUT}/spiral_L_perstate.pdf"); fig.savefig(f"{OUT}/spiral_L_perstate.png", dpi=200); plt.close(fig)

print("mean ||L|| = %.3f | max eig %.3f | %% unstable %.1f" % (mval, maxre.max(), frac*100))
print("regenerated with clear labels")
