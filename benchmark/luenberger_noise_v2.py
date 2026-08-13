"""Luenberger under measurement noise (fair comparison vs PINN noise study).
Gaussian noise is added to the 6 measured signals fed to the observer at every step
(positions sigma_pos, angles sigma_att); evaluated vs the TRUE states. Same 3 levels as
the PINN study. Two gains shown (w=10 as on clean, w=5 as a noise-robust choice).
"""
import sys, os, csv
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import numpy as np
from core.generate_quadrotor_data import quadrotor_dynamics

MEAS = [0, 1, 2, 6, 7, 8]; PART = [3, 4, 5, 9, 10, 11]; HID = [3, 4, 5, 9, 10, 11]
CASES = [("clean", 0.00, 0.00), ("moderate", 0.05, 0.01), ("strong", 0.10, 0.02)]
PINN = {"clean": (0.0442, 0.0559), "moderate": (0.0775, 0.0699), "strong": (0.0850, 0.0720)}

def build_L(w):
    L = np.zeros((12, 6))
    for j in range(6):
        L[MEAS[j], j] = 2.0*w; L[PART[j], j] = w*w
    return L

def rhs(xh, u, y, L):
    return quadrotor_dynamics(0.0, xh, u) + L @ (y - xh[MEAS])

def run(Xtrue, U, Ymeas, dt, L):
    N = Xtrue.shape[0]; xh = Xtrue[0].copy(); est = np.zeros_like(Xtrue); est[0] = xh
    for k in range(N-1):
        u, y = U[k], Ymeas[k]
        k1 = rhs(xh,u,y,L); k2 = rhs(xh+0.5*dt*k1,u,y,L)
        k3 = rhs(xh+0.5*dt*k2,u,y,L); k4 = rhs(xh+dt*k3,u,y,L)
        xh = xh + dt/6.0*(k1+2*k2+2*k3+k4); est[k+1] = xh
    return est

d = np.load("datasets/spiral_v2_dataset.npz"); T, X, U = d["T"], d["X"], d["U"]
dt = float(T[1]-T[0]); test = list(range(40, 50))

def evaluate(sp, sa, w, seed=0):
    rng = np.random.default_rng(seed); L = build_L(w); per = []
    for f in test:
        Ym = X[f][:, MEAS].copy()
        if sp > 0: Ym[:, 0:3] += sp*rng.standard_normal(Ym[:, 0:3].shape)
        if sa > 0: Ym[:, 3:6] += sa*rng.standard_normal(Ym[:, 3:6].shape)
        est = run(X[f], U[f], Ym, dt, L)
        per.append(np.sqrt(((est - X[f])**2).mean(axis=0)))
    per = np.array(per).mean(axis=0)
    return per[MEAS].mean(), per[HID].mean()

rows = []
print("=== Luenberger under noise vs PINN ===", flush=True)
print(f"  {'case':<9}{'w':>4}{'L meas':>9}{'L hid':>8}{'PINN meas':>11}{'PINN hid':>10}", flush=True)
for name, sp, sa in CASES:
    for w in [10.0, 5.0]:
        mr, hr = evaluate(sp, sa, w)
        pm, ph = PINN[name]
        print(f"  {name:<9}{w:>4.0f}{mr:>9.4f}{hr:>8.4f}{pm:>11.4f}{ph:>10.4f}", flush=True)
        rows.append([name, sp, sa, w, f"{mr:.4f}", f"{hr:.4f}", f"{pm:.4f}", f"{ph:.4f}"])

os.makedirs("docs", exist_ok=True)
with open("docs/luenberger_noise_v2.csv", "w", newline="") as fp:
    wr = csv.writer(fp)
    wr.writerow(["case","sigma_pos","sigma_att","w","L_meas_RMSE","L_hidden_RMSE","PINN_meas_RMSE","PINN_hidden_RMSE"])
    wr.writerows(rows)
print("\nsaved -> docs/luenberger_noise_v2.csv", flush=True)
