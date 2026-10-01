# -*- coding: utf-8 -*-
"""
REAL feedback observer test (answers reviewer comment [15]).

The estimate is produced by integrating the observer ODE forward with the test
flight's ACTUAL control u(t) and measurement y(t):

    x_hat_dot = f(x_hat, u) + L . ( y - C x_hat )

and the observer is started from an UNKNOWN hidden state (the 6 velocities /
rates are NOT given the true value; the 6 measured states are initialised from
the measurement y(0), which is known). This removes both problems flagged in
[15]: the estimate uses y(t) and u(t) online, and it does not rely on the true
initial velocities contained in x0.

Three gains are compared, all under the same unknown-x0 start:
  (1) Structured Luenberger (bandwidth w)      - constant, hand-designed
  (2) Designed LQE / steady-state Kalman gain  - constant, model-based
  (3) PINN learned gain L(t) trained through the integration (pinn_4x100_v2_L_ode)

For reference we also report the LEAKAGE baseline (true hidden x0) so the cost
of not knowing the velocities is visible.
"""
import sys, os, csv
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "phase1a"))
sys.path.insert(0, os.path.join(PROJ, "L_gain", "L_gain_TrainODE"))

import torch
from scipy.linalg import solve_continuous_are
from core.generate_quadrotor_data import quadrotor_dynamics          # numpy f(t,x,u)
from dynamics_torch import quadrotor_dynamics_torch                  # torch f(x,u)
from pinn_observer_v5 import PINNObserverV5
from observer_v2_L_gain import integrate_observer, build_C

MEAS = [0, 1, 2, 6, 7, 8]        # x, y, z, phi, theta, psi
HID  = [3, 4, 5, 9, 10, 11]      # vx, vy, vz, p, q, r
NAMES = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
TEST = range(40, 50)
SKIP = 100                       # drop first 1.0 s (transient) from RMSE
DEV = "cpu"

# ----------------------------------------------------------------------
# data
d = np.load(os.path.join(PROJ, "datasets", "spiral_v2_dataset.npz"))
T, X, U = d["T"], d["X"], d["U"]
dt = float(T[1] - T[0])
N = X.shape[1]

# ----------------------------------------------------------------------
# gain (1): structured Luenberger
def build_L_luen(w):
    L = np.zeros((12, 6))
    for j in range(6):
        L[MEAS[j], j] = 2.0 * w
        L[HID[j],  j] = w * w
    return L

# ----------------------------------------------------------------------
# gain (2): designed steady-state LQE (continuous ARE) at hover linearisation
def design_lqe(q=100.0, r=1.0):
    C = np.zeros((6, 12))
    for r_, c_ in enumerate(MEAS):
        C[r_, c_] = 1.0
    x_h = torch.zeros(12)
    u_h = torch.tensor([1.80 * 9.81, 0., 0., 0.])
    A0 = torch.autograd.functional.jacobian(
        lambda x: quadrotor_dynamics_torch(x.unsqueeze(0), u_h.unsqueeze(0)).squeeze(0), x_h
    ).numpy()
    P = solve_continuous_are(A0.T, C.T, q * np.eye(12), r * np.eye(6))
    L = P @ C.T / r
    return L

# ----------------------------------------------------------------------
# numpy RK4 integration of the observer with a CONSTANT gain L, unknown x0
def run_const_gain(Xtrue, Uf, L, x_hat0):
    xh = x_hat0.copy()
    est = np.zeros_like(Xtrue); est[0] = xh
    def rhs(xh, u, y):
        return quadrotor_dynamics(0.0, xh, u) + L @ (y - xh[MEAS])
    for k in range(N - 1):
        u, y = Uf[k], Xtrue[k, MEAS]
        k1 = rhs(xh, u, y)
        k2 = rhs(xh + 0.5*dt*k1, u, y)
        k3 = rhs(xh + 0.5*dt*k2, u, y)
        k4 = rhs(xh + dt*k3, u, y)
        xh = xh + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)
        if not np.all(np.isfinite(xh)):
            est[k+1:] = np.nan; return est
        est[k+1] = xh
    return est

# ----------------------------------------------------------------------
# PINN learned-gain: query L(t) ONCE (vectorised), then integrate in numpy
def load_pinn():
    m = PINNObserverV5(hidden=100, n_hidden_layers=4).to(DEV)
    m.load_state_dict(torch.load(os.path.join(PROJ, "phase1a", "pinn_4x100_v2_L_ode.pth"),
                                 map_location=DEV))
    m.eval()
    return m

_L_CACHE = {}
@torch.no_grad()
def pinn_L_series(model, Xtrue, f_idx):
    """L(t) along the flight as numpy (N,12,6); cached per flight."""
    if f_idx in _L_CACHE:
        return _L_CACHE[f_idx]
    tt = torch.tensor(T.reshape(-1, 1), dtype=torch.float32)
    x0rep = torch.tensor(Xtrue[0], dtype=torch.float32).view(1, 12).repeat(len(T), 1)
    _, L_all = model.get_state_and_gain(tt, x0rep)       # (N,12,6)
    L_np = L_all.cpu().numpy()
    _L_CACHE[f_idx] = L_np
    return L_np

def run_pinn_gain(model, Xtrue, Uf, x_hat0, f_idx):
    L_series = pinn_L_series(model, Xtrue, f_idx)        # (N,12,6)
    xh = x_hat0.copy()
    est = np.zeros_like(Xtrue); est[0] = xh
    def rhs(xh, u, y, Lk):
        return quadrotor_dynamics(0.0, xh, u) + Lk @ (y - xh[MEAS])
    for k in range(N - 1):
        u, y, Lk = Uf[k], Xtrue[k, MEAS], L_series[k]
        k1 = rhs(xh, u, y, Lk)
        k2 = rhs(xh + 0.5*dt*k1, u, y, Lk)
        k3 = rhs(xh + 0.5*dt*k2, u, y, Lk)
        k4 = rhs(xh + dt*k3, u, y, Lk)
        xh = xh + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)
        if not np.all(np.isfinite(xh)):
            est[k+1:] = np.nan; return est
        est[k+1] = xh
    return est

# ----------------------------------------------------------------------
# build an initial estimate with a chosen hidden-state guess
def make_x_hat0(Xtrue, mode, rng):
    xh0 = np.zeros(12)
    xh0[MEAS] = Xtrue[0, MEAS]          # measured states known from y(0)
    if mode == "true":                  # leakage baseline
        xh0[HID] = Xtrue[0, HID]
    elif mode == "zero":                # no knowledge of velocities
        xh0[HID] = 0.0
    elif mode == "noisy":               # true + large noise
        xh0[HID] = Xtrue[0, HID] + rng.normal(0, 0.5, size=6)
    return xh0

# ----------------------------------------------------------------------
def rmse_per_state(est, Xtrue):
    e = est[SKIP:] - Xtrue[SKIP:]
    return np.sqrt(np.nanmean(e**2, axis=0))

def evaluate(run_fn, mode, seed=0):
    rng = np.random.default_rng(seed)
    per = []
    for f in TEST:
        xh0 = make_x_hat0(X[f], mode, rng)
        est = run_fn(X[f], U[f], xh0, f)
        per.append(rmse_per_state(est, X[f]))
    per = np.nanmean(np.array(per), axis=0)
    return per

# ----------------------------------------------------------------------
if __name__ == "__main__":
    L_luen = build_L_luen(10.0)
    L_lqe  = design_lqe(q=100.0, r=1.0)
    model  = load_pinn()

    runners = {
        "Luenberger (w=10)": lambda Xt, Uf, xh0, fi: run_const_gain(Xt, Uf, L_luen, xh0),
        "Designed LQE (q=100)": lambda Xt, Uf, xh0, fi: run_const_gain(Xt, Uf, L_lqe, xh0),
        "PINN learned L(t)": lambda Xt, Uf, xh0, fi: run_pinn_gain(model, Xt, Uf, xh0, fi),
    }
    modes = ["true", "zero", "noisy"]

    rows = []
    print("="*68)
    print("REAL feedback observer - hidden-state RMSE (transient 1s excluded)")
    print("start 'true' = leakage baseline; 'zero'/'noisy' = unknown velocities")
    print("="*68)
    header = f"{'gain':<22} | " + " | ".join(f"{m:>8}" for m in modes)
    print(header); print("-"*len(header))
    for name, fn in runners.items():
        cells = []
        for mode in modes:
            per = evaluate(fn, mode)
            hid = per[HID].mean()
            cells.append(hid)
            rows.append([name, mode, f"{per[MEAS].mean():.4f}", f"{hid:.4f}"])
        print(f"{name:<22} | " + " | ".join(f"{c:8.4f}" for c in cells))

    # detailed per-state for the headline case: unknown (zero) start
    print("\n" + "="*68)
    print("Per-state RMSE, UNKNOWN (zero-velocity) start, 10 unseen flights")
    print("="*68)
    detail_rows = []
    for name, fn in runners.items():
        per = evaluate(fn, "zero")
        print(f"\n-- {name} --")
        for i, n in enumerate(NAMES):
            tag = "measured" if i in MEAS else "HIDDEN"
            print(f"  {n:>6} | {per[i]:.4f} | {tag}")
            detail_rows.append([name, n, tag, f"{per[i]:.4f}"])
        print(f"  avg measured {per[MEAS].mean():.4f} | avg hidden {per[HID].mean():.4f}")

    out = os.path.join(PROJ, "benchmark", "real_observer_unknownx0_summary.csv")
    with open(out, "w", newline="") as fp:
        wr = csv.writer(fp)
        wr.writerow(["gain", "start", "avg_measured_RMSE", "avg_hidden_RMSE"])
        wr.writerows(rows)
        wr.writerow([])
        wr.writerow(["gain", "state", "group", "RMSE_zero_start"])
        wr.writerows(detail_rows)
    print(f"\nsaved -> {out}")
