"""Luenberger observer baseline on the spiral_v2 unseen test flights.
x_hat_dot = f(x_hat, u) + L (y - C x_hat), integrated forward (RK4) with the recorded
controls u(t) and clean measurements y(t). Nominal Singha model. Start = true x0.
Structured gain: each measured state paired with its unmeasured derivative
(x<->vx, y<->vy, z<->vz, phi<->phidot, theta<->thetadot, psi<->psidot), poles at -w
=> l1 = 2w on the measured row, l2 = w^2 on its derivative row.
"""
import sys, os, csv
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import numpy as np
from core.generate_quadrotor_data import quadrotor_dynamics

MEAS = [0, 1, 2, 6, 7, 8]        # x, y, z, phi, theta, psi
PART = [3, 4, 5, 9, 10, 11]      # vx, vy, vz, phidot, thetadot, psidot (their derivatives)
HID  = [3, 4, 5, 9, 10, 11]
NAMES = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']

def build_L(w):
    L = np.zeros((12, 6))
    for j in range(6):
        L[MEAS[j], j] = 2.0 * w      # l1
        L[PART[j], j] = w * w        # l2
    return L

def obs_rhs(xh, u, y, L):
    return quadrotor_dynamics(0.0, xh, u) + L @ (y - xh[MEAS])

def run_flight(Xtrue, U, dt, L):
    N = Xtrue.shape[0]
    xh = Xtrue[0].copy()
    est = np.zeros_like(Xtrue); est[0] = xh
    for k in range(N - 1):
        u, y = U[k], Xtrue[k, MEAS]
        k1 = obs_rhs(xh, u, y, L)
        k2 = obs_rhs(xh + 0.5*dt*k1, u, y, L)
        k3 = obs_rhs(xh + 0.5*dt*k2, u, y, L)
        k4 = obs_rhs(xh + dt*k3, u, y, L)
        xh = xh + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)
        est[k+1] = xh
    return est

d = np.load("datasets/spiral_v2_dataset.npz")
T, X, U = d["T"], d["X"], d["U"]
dt = float(T[1] - T[0])
test = range(40, 50)

def evaluate(w):
    per = []
    for f in test:
        est = run_flight(X[f], U[f], dt, build_L(w))
        per.append(np.sqrt(((est - X[f])**2).mean(axis=0)))
    per = np.array(per).mean(axis=0)   # per-state RMSE averaged over 10 flights
    return per

print("=== Luenberger sensitivity (bandwidth w) ===", flush=True)
for w in [5.0, 10.0, 20.0]:
    per = evaluate(w)
    print(f"  w={w:>4}: meas {per[MEAS].mean():.4f} | hidden {per[HID].mean():.4f}", flush=True)

W = 10.0
per = evaluate(W)
print(f"\n=== Luenberger per-state RMSE (w={W}) — 10 unseen flights ===", flush=True)
for i, n in enumerate(NAMES):
    tag = "measured" if i in MEAS else "HIDDEN"
    print(f"  {n:>6} | {per[i]:.4f} | {tag}", flush=True)
print(f"  avg measured: {per[MEAS].mean():.4f}", flush=True)
print(f"  avg hidden:   {per[HID].mean():.4f}", flush=True)
print(f"  (PINN v2: meas 0.0442 | hidden 0.0559)", flush=True)

os.makedirs("docs", exist_ok=True)
with open("docs/luenberger_v2_summary.csv", "w", newline="") as fp:
    wr = csv.writer(fp); wr.writerow(["state","group","luenberger_RMSE"])
    for i, n in enumerate(NAMES):
        wr.writerow([n, "measured" if i in MEAS else "hidden", f"{per[i]:.4f}"])
    wr.writerow(["avg_measured","", f"{per[MEAS].mean():.4f}"])
    wr.writerow(["avg_hidden","", f"{per[HID].mean():.4f}"])
print("saved -> docs/luenberger_v2_summary.csv", flush=True)
