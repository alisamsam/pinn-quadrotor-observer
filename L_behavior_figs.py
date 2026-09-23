# L_behavior_figs.py
# Produce the L-gain behavior figures from the SAVED 4x100 spiral checkpoint
# (no retraining). Run from the repo ROOT:
#   python L_behavior_figs.py
# Outputs (into paper_figures/): ||L(t)|| curve, per-state L(t) panels,
# and max Re eig(A - LC) over time (the stability view).

import sys, os
ROOT = os.getcwd()
for p in [ROOT, os.path.join(ROOT, "phase1a")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

CKPT    = "phase1a/pinn_4x100_v2_L.pth"       # 4x100 spiral learned-gain model
DATASET = "datasets/spiral_v2_dataset.npz"    # controller_v2 spiral
FLIGHT  = 0                                    # one unseen test flight
NSTEPS  = 3000
STEP    = 25                                   # eig eval every 25th step (120 points)
OUT     = "paper_figures"
os.makedirs(OUT, exist_ok=True)

NAMES = ["x","y","z","vx","vy","vz","phi","theta","psi","p","q","r"]
MEAS_NAMES = ["e_x","e_y","e_z","e_phi","e_theta","e_psi"]

train, test = load_phase4_data(DATASET)
a = FLIGHT * NSTEPS; b = a + NSTEPS
T = test["T"][a:b]; X = test["X"][a:b]; U = test["U"][a:b]; x0 = test["X0"][a]

model = PINNObserverV5(hidden=100, n_hidden_layers=4)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

with torch.no_grad():
    _, L_all = model.get_state_and_gain(T, x0.view(1, 12).repeat(NSTEPS, 1))  # (N,12,6)
t = T.view(-1).numpy()
L = L_all.numpy()

# ---- Figure 1: Frobenius norm ||L(t)|| over time ----
Lf = np.sqrt((L ** 2).reshape(NSTEPS, -1).sum(axis=1))
np.savetxt(f"{OUT}/fig_L_norm_timeseries.csv", np.column_stack([t, Lf]),
           delimiter=",", header="t,L_fro", comments="", fmt="%.6g")
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, Lf, color="crimson", lw=1.5)
m = float(Lf.mean())
ax.axhline(m, ls="--", color="gray", lw=1, label=f"time mean = {m:.1f}")
ax.set_xlabel("time t  (s)"); ax.set_ylabel(r"$\|L(t)\|_F$")
ax.set_title("Learned gain magnitude over time (spiral, 4x100)")
ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/spiral_L_norm.pdf"); fig.savefig(f"{OUT}/spiral_L_norm.png", dpi=300)
plt.close(fig)

# ---- Figure 2: per-state L(t) (12 panels, 6 measured-error columns each) ----
fig, axes = plt.subplots(4, 3, figsize=(15, 10))
fig.suptitle("Learned gain L(t): each panel is one state's row, six measured-error columns",
             fontsize=14)
for i in range(12):
    axc = axes.flat[i]
    for j in range(6):
        axc.plot(t, L[:, i, j], lw=1.0, label=MEAS_NAMES[j])
    axc.set_title(NAMES[i]); axc.grid(alpha=0.3); axc.set_xlabel("t (s)")
    if i == 0:
        axc.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}/spiral_L_perstate.pdf"); fig.savefig(f"{OUT}/spiral_L_perstate.png", dpi=200)
plt.close(fig)

# ---- Figure 3: max Re eig(A(t) - L(t)C) over time (stability) ----
C = np.zeros((6, 12))
for r, c in enumerate(MEAS_IDX):
    C[r, c] = 1.0
times, maxre = [], []
for k in range(0, NSTEPS, STEP):
    uk = U[k]
    f_single = lambda xx: quadrotor_dynamics_torch(xx.unsqueeze(0), uk.unsqueeze(0)).squeeze(0)
    A = torch.autograd.functional.jacobian(f_single, X[k]).numpy()   # (12,12)
    Lk = L[k]                                                        # (12,6)
    ev = np.linalg.eigvals(A - Lk @ C)
    times.append(t[k]); maxre.append(ev.real.max())
times = np.array(times); maxre = np.array(maxre)
frac = float((maxre > 0).mean())
np.savetxt(f"{OUT}/fig_L_eig_timeseries.csv", np.column_stack([times, maxre]),
           delimiter=",", header="t,max_re_eig", comments="", fmt="%.6g")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(times, maxre, "r-", lw=1.6, label="max Re eig(A - LC)")
ax.axhline(0, color="k", lw=1)
ax.fill_between(times, 0, maxre, where=(maxre > 0), color="red", alpha=0.15)
ax.set_xlabel("time t  (s)"); ax.set_ylabel("largest real part of eig(A - LC)")
ax.set_title(f"Stability of the learned gain over time  ({frac*100:.0f}% of flight unstable)")
ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/spiral_L_eig.pdf"); fig.savefig(f"{OUT}/spiral_L_eig.png", dpi=300)
plt.close(fig)

print("SAVED figures to paper_figures/")
print("mean ||L|| = %.3f" % m)
print("max Re eig over flight = %.3f | %% of flight unstable = %.1f" % (maxre.max(), frac * 100))
