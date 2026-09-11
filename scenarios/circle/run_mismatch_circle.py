# -*- coding: utf-8 -*-
"""Model-mismatch study for controller_v3 + frozen CIRCLE observer (pinn_circle_4x100).
Same 'unaware' protocol as the spiral study (run_mismatch_v2.py): the true dynamics
use perturbed parameters; the controller stays nominal; the frozen observer
(nominal-trained) is evaluated vs the true states. Writes docs/circle_mismatch_summary.csv.

FULLY SELF-CONTAINED: the controller (controller_v3), the circle reference, the
perturbed Singha dynamics and a fixed-step RK4 (dt = 0.01 s) are all inlined, so the
ONLY repo imports are PINNObserverV4 / quadrotor_dynamics_torch / data_phase4 (the same
modules the spiral study and the cluster training already use). No scipy, no matplotlib,
no core.controller_v3 / scenarios.* needed on the cluster.

Circle observer config: 4x100, loss weights (0.5, 0.5, 1.0), circle_v3 dataset.
If phase1a/pinn_circle_4x100.pth is absent it is trained once and cached, then frozen.

Run from repo root:
    module load pytorch/2.0.0/gpu
    nohup python3 scenarios/circle/run_mismatch_circle.py >> mismatch_circle_log.txt 2>&1 &
"""
import sys, os, csv
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "phase1a"))
import numpy as np
import torch
import torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from dynamics_torch import quadrotor_dynamics_torch
from data_phase4 import load_phase4_data, MEAS_IDX

# ===== physical constants (core/generate_quadrotor_data) =====
M, G, L_ARM, IX, IY, IZ = 1.80, 9.81, 0.20, 0.03, 0.03, 0.04
NOMINAL = dict(m=1.80, Ix=0.03, Iy=0.03, Iz=0.04, l=0.20)

# ===== circle reference (scenarios/circle/circle_reference.py) =====
R_CIRC, OMEGA, Z_LEVEL = 10.0, 0.3, 2.0
def circular_reference(t):
    w = OMEGA
    return (np.array([R_CIRC*np.sin(w*t), R_CIRC*np.cos(w*t), Z_LEVEL]),
            np.array([R_CIRC*w*np.cos(w*t), -R_CIRC*w*np.sin(w*t), 0.0]),
            np.array([-R_CIRC*w**2*np.sin(w*t), -R_CIRC*w**2*np.cos(w*t), 0.0]))
def yaw_reference(t):
    w = OMEGA
    return 0.2*np.sin(w*t), 0.2*w*np.cos(w*t)

# ===== controller_v3 (core/controller_v3.py) =====
GAINS = dict(kp_xy=4.0, kd_xy=4.0, kp_z=9.0, kd_z=6.0,
             kp_att=49.0, kd_att=14.0, kp_yaw=9.0, kd_yaw=6.0)
LIM = dict(u1_min=1.0, u1_max=35.0, tilt_max=np.radians(30.0), tau_max=5.0)
def tilt_from_accel(Uex, Uey, psi):
    phi_d = np.arcsin(np.clip(Uex*np.sin(psi) - Uey*np.cos(psi), -1.0, 1.0))
    theta_d = np.arcsin(np.clip((Uex*np.cos(psi) + Uey*np.sin(psi))/np.cos(phi_d), -1.0, 1.0))
    return phi_d, theta_d
def full_control_v3(t, x, ref_fn, yaw_fn):
    g, lim = GAINS, LIM
    pos_d, vel_d, acc_d = ref_fn(t); psi_d, psi_d_dot = yaw_fn(t)
    phi, theta, psi = x[6], x[7], x[8]; p, q, r = x[9], x[10], x[11]
    az = acc_d[2] + g['kp_z']*(pos_d[2]-x[2]) + g['kd_z']*(vel_d[2]-x[5])
    den = np.cos(phi)*np.cos(theta); den = np.sign(den)*max(abs(den), 0.25)
    U1 = float(np.clip(M*(G+az)/den, lim['u1_min'], lim['u1_max']))
    ax = acc_d[0] + g['kp_xy']*(pos_d[0]-x[0]) + g['kd_xy']*(vel_d[0]-x[3])
    ay = acc_d[1] + g['kp_xy']*(pos_d[1]-x[1]) + g['kd_xy']*(vel_d[1]-x[4])
    phi_d, theta_d = tilt_from_accel((M/U1)*ax, (M/U1)*ay, psi)
    phi_d = float(np.clip(phi_d, -lim['tilt_max'], lim['tilt_max']))
    theta_d = float(np.clip(theta_d, -lim['tilt_max'], lim['tilt_max']))
    U2 = (IX/L_ARM)*(g['kp_att']*(phi_d-phi) - g['kd_att']*p - q*r*(IY-IZ)/IX)
    U3 = (IY/L_ARM)*(g['kp_att']*(theta_d-theta) - g['kd_att']*q - p*r*(IZ-IX)/IY)
    U4 = IZ*(g['kp_yaw']*(psi_d-psi) + g['kd_yaw']*(psi_d_dot-r) - p*q*(IX-IY)/IZ)
    U2 = float(np.clip(U2, -lim['tau_max'], lim['tau_max']))
    U3 = float(np.clip(U3, -lim['tau_max'], lim['tau_max']))
    U4 = float(np.clip(U4, -lim['tau_max'], lim['tau_max']))
    return np.array([U1, U2, U3, U4])

# ===== perturbed Singha dynamics + fixed-step RK4 =====
def quadrotor_dynamics(x, u, P):
    m, Ix, Iy, Iz, l = P["m"], P["Ix"], P["Iy"], P["Iz"], P["l"]
    _x, _y, _z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot = x
    U1, U2, U3, U4 = u
    Ux = np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi)
    Uy = np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi)
    return np.array([xdot, ydot, zdot,
                     (Ux/m)*U1, (Uy/m)*U1, (np.cos(phi)*np.cos(theta)/m)*U1 - G,
                     phidot, thetadot, psidot,
                     thetadot*psidot*(Iy-Iz)/Ix + (l/Ix)*U2,
                     psidot*phidot*(Iz-Ix)/Iy + (l/Iy)*U3,
                     phidot*thetadot*(Ix-Iy)/Iz + (l/Iz)*U4])
def rk4_rollout(P, x0, t_eval):
    def deriv(tt, xx):
        return quadrotor_dynamics(xx, full_control_v3(tt, xx, circular_reference, yaw_reference), P)
    N = len(t_eval); X = np.empty((N, 12)); x = np.asarray(x0, dtype=float); X[0] = x
    for k in range(N - 1):
        tk = t_eval[k]; h = t_eval[k + 1] - tk
        k1 = deriv(tk, x); k2 = deriv(tk + h/2, x + (h/2)*k1)
        k3 = deriv(tk + h/2, x + (h/2)*k2); k4 = deriv(tk + h, x + h*k3)
        x = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4); X[k + 1] = x
    return X

HID = [i for i in range(12) if i not in MEAS_IDX]
DATASET = "datasets/circle_v3_dataset.npz"
CKPT    = "phase1a/pinn_circle_4x100.pth"
LAYERS, HIDDEN = 4, 100
W0, WODE, WY = 0.5, 0.5, 1.0            # circle best weights (Case 3)

# ---- frozen circle observer (train once if the checkpoint is missing) ----
model = PINNObserverV4(hidden=HIDDEN, n_hidden_layers=LAYERS)
if os.path.exists(CKPT):
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    print("loaded", CKPT, flush=True)
else:
    print("training circle observer (4x100, w=0.5/0.5/1.0) ...", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    train, _ = load_phase4_data(DATASET)
    T = train["T"].to(dev); X0 = train["X0"].to(dev); Y = train["Y"].to(dev)
    U = train["U"].to(dev); X = train["X"].to(dev)
    N = T.shape[0]; t0m = (T[:, 0] == T[:, 0].min()); T0, X0_0, Xtrue0 = T[t0m], X0[t0m], X[t0m]
    def residual(m, t, x0, u):
        t = t.clone().requires_grad_(True); xh = m(t, x0); dx = torch.zeros_like(xh)
        for i in range(12):
            dx[:, i] = torch.autograd.grad(xh[:, i].sum(), t, create_graph=True)[0][:, 0]
        return dx - quadrotor_dynamics_torch(xh, u)
    torch.manual_seed(0); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, 2001):
        perm = torch.randperm(N, device=dev)
        for s in range(0, N, 4096):
            idx = perm[s:s+4096]; tb, x0b, yb, ub = T[idx], X0[idx], Y[idx], U[idx]
            opt.zero_grad(); xh = model(tb, x0b)
            ly = nn.functional.mse_loss(xh[:, MEAS_IDX], yb)
            res = residual(model, tb, x0b, ub)
            lg = nn.functional.mse_loss(res, torch.zeros_like(res))
            l0 = nn.functional.mse_loss(model(T0, X0_0), Xtrue0)
            (WY*ly + WODE*lg + W0*l0).backward(); opt.step()
        if ep == 1 or ep % 500 == 0:
            print(f"  ep {ep:4d} | MSE_0 {l0.item():.3e} | MSE_g {lg.item():.3e} | MSE_y {ly.item():.3e}", flush=True)
    torch.save(model.state_dict(), CKPT); print("saved", CKPT, flush=True)
    model = model.cpu()
model.eval()

base = np.load(DATASET); Xb = base["X"]
x0s = [Xb[i, 0, :] for i in range(40, 50)]   # 10 unseen test starts
t_eval = np.arange(0.0, 30.0, 0.01)

def eval_set(P_true):
    me, he = [], []
    for x0 in x0s:
        xt = rk4_rollout(P_true, x0, t_eval)
        Tt = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
        X0t = torch.tensor(np.tile(x0, (len(t_eval), 1)), dtype=torch.float32)
        with torch.no_grad():
            xh = model(Tt, X0t).numpy()
        rmse = np.sqrt(((xh - xt)**2).mean(axis=0))
        me.append(rmse[MEAS_IDX].mean()); he.append(rmse[HID].mean())
    return float(np.mean(me)), float(np.mean(he))

devs = [-0.20, -0.10, 0.0, 0.10, 0.20]
params = {"mass": ["m"], "inertia": ["Ix", "Iy", "Iz"], "arm": ["l"]}
res = {}
for pname, keys in params.items():
    for dev in devs:
        P = dict(NOMINAL)
        for k in keys: P[k] = NOMINAL[k]*(1+dev)
        mr, hr = eval_set(P)
        res[(pname, dev)] = (mr, hr)
        print(f"{pname:>8} {dev:+.0%}: meas {mr:.4f} | hidden {hr:.4f}", flush=True)

# combined worst: all params +20%
Pw = dict(NOMINAL)
for k in ["m", "Ix", "Iy", "Iz", "l"]: Pw[k] = NOMINAL[k]*1.20
cw = eval_set(Pw); print(f"combined +20%: meas {cw[0]:.4f} | hidden {cw[1]:.4f}", flush=True)

cols = ["-20%", "-10%", "nominal", "+10%", "+20%"]
os.makedirs("docs", exist_ok=True)
with open("docs/circle_mismatch_summary.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["parameter", "metric"] + cols)
    for pname in params:
        w.writerow([pname, "measured RMSE"] + [f"{res[(pname, d)][0]:.4f}" for d in devs])
        w.writerow([pname, "hidden RMSE"]   + [f"{res[(pname, d)][1]:.4f}" for d in devs])
    w.writerow(["combined worst", "measured RMSE", "", "", f"{res[('mass', 0.0)][0]:.4f}", "", f"{cw[0]:.4f}"])
    w.writerow(["combined worst", "hidden RMSE",   "", "", f"{res[('mass', 0.0)][1]:.4f}", "", f"{cw[1]:.4f}"])
print("saved -> docs/circle_mismatch_summary.csv", flush=True)
