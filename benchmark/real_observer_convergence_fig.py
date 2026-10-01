# -*- coding: utf-8 -*-
"""
Convergence-from-unknown-x0 figure for reviewer [15].

Runs the Luenberger feedback observer on one unseen flight from two starts:
  (a) true hidden velocities  (leakage baseline)
  (b) zero / wrong hidden velocities (unknown)
and plots the hidden-state error norm ||x_hidden - x_hat_hidden|| vs time.
Both collapse within ~1 s and are then indistinguishable: the estimate after
the transient is set by the measurements, not by the initial guess.
Also prints hidden RMSE with the transient INCLUDED vs EXCLUDED, so the table
in the paper can state both.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from core.generate_quadrotor_data import quadrotor_dynamics

MEAS = [0, 1, 2, 6, 7, 8]
HID  = [3, 4, 5, 9, 10, 11]
FLIGHT = 40                      # first unseen flight
DEV = "cpu"

d = np.load(os.path.join(PROJ, "datasets", "spiral_v2_dataset.npz"))
T, X, U = d["T"], d["X"], d["U"]
dt = float(T[1] - T[0]); N = X.shape[1]

def build_L_luen(w):
    L = np.zeros((12, 6))
    for j in range(6):
        L[MEAS[j], j] = 2.0 * w
        L[HID[j],  j] = w * w
    return L

def run(Xtrue, Uf, L, xh0):
    xh = xh0.copy(); est = np.zeros_like(Xtrue); est[0] = xh
    rhs = lambda xh, u, y: quadrotor_dynamics(0.0, xh, u) + L @ (y - xh[MEAS])
    for k in range(N - 1):
        u, y = Uf[k], Xtrue[k, MEAS]
        k1 = rhs(xh, u, y); k2 = rhs(xh+0.5*dt*k1, u, y)
        k3 = rhs(xh+0.5*dt*k2, u, y); k4 = rhs(xh+dt*k3, u, y)
        xh = xh + dt/6.0*(k1+2*k2+2*k3+k4); est[k+1] = xh
    return est

L = build_L_luen(10.0)
Xt, Uf = X[FLIGHT], U[FLIGHT]

xh_true = np.zeros(12); xh_true[MEAS] = Xt[0, MEAS]; xh_true[HID] = Xt[0, HID]
xh_zero = np.zeros(12); xh_zero[MEAS] = Xt[0, MEAS]; xh_zero[HID] = 0.0

est_true = run(Xt, Uf, L, xh_true)
est_zero = run(Xt, Uf, L, xh_zero)

err_true = np.linalg.norm(est_true[:, HID] - Xt[:, HID], axis=1)
err_zero = np.linalg.norm(est_zero[:, HID] - Xt[:, HID], axis=1)

# RMSE with / without transient
def hid_rmse(est, skip):
    e = est[skip:, HID] - Xt[skip:, HID]
    return np.sqrt((e**2).mean())
print(f"flight {FLIGHT}")
print(f"  true-start  hidden RMSE: full {hid_rmse(est_true,0):.4f} | skip1s {hid_rmse(est_true,100):.4f}")
print(f"  zero-start  hidden RMSE: full {hid_rmse(est_zero,0):.4f} | skip1s {hid_rmse(est_zero,100):.4f}")
print(f"  max |err_true - err_zero| after 1s: {np.abs(err_true-err_zero)[100:].max():.2e}")

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(T, err_zero, color="tab:red", lw=1.8, label="unknown start (velocities = 0)")
ax.plot(T, err_true, color="tab:green", lw=1.4, ls="--", label="true-velocity start (leakage)")
ax.axvspan(0, 1.0, color="0.85", alpha=0.6, zorder=0, label="transient (excluded from RMSE)")
ax.set_xlabel("time  t  [s]")
ax.set_ylabel(r"hidden-state error  $\|x_{\rm hid}-\hat{x}_{\rm hid}\|$")
ax.set_title("Observer convergence from an unknown initial velocity")
ax.set_xlim(0, T[-1]); ax.set_ylim(bottom=0)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(PROJ, "benchmark", f"real_observer_convergence.{ext}"), dpi=200)
print("saved -> benchmark/real_observer_convergence.pdf / .png")
