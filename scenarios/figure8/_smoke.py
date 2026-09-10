# -*- coding: utf-8 -*-
"""Model-mismatch study for controller_v3 + frozen FIGURE-8 observer (pinn_figure8_9x100).
Same 'unaware' protocol as the spiral study (run_mismatch_v2.py): the true dynamics
use perturbed parameters; the controller stays nominal; the frozen observer
(nominal-trained) is evaluated vs the true states. Runs in memory (generation + eval).
Writes /tmp/smoke_f8.csv.

Figure-8 observer config: 9x100, loss weights (1.0, 1.0, 1.0), figure8_v3 dataset.
If /tmp/smoke_f8.pth is absent it is trained once and cached, then frozen.

Run from repo root:
    module load pytorch/2.0.0/gpu
    nohup python3 scenarios/figure8/run_mismatch_figure8.py >> mismatch_figure8_log.txt 2>&1 &
"""
import sys, os, csv
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "phase1a"))
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from phase_mismatch.generate_perturbed import quadrotor_dynamics, NOMINAL
from core.controller_v3 import full_control_v3
from scenarios.figure8.figure8_reference import figure8_reference, yaw_reference
from pinn_observer_v4 import PINNObserverV4
from dynamics_torch import quadrotor_dynamics_torch
from data_phase4 import load_phase4_data, MEAS_IDX

HID = [i for i in range(12) if i not in MEAS_IDX]
DATASET = "datasets/figure8_v3_dataset.npz"
CKPT    = "/tmp/smoke_f8.pth"
LAYERS, HIDDEN = 9, 100
W0, WODE, WY = 1.0, 1.0, 1.0            # figure-8 best weights (Case 1)

# ---- frozen figure-8 observer (train once if the checkpoint is missing) ----
model = PINNObserverV4(hidden=HIDDEN, n_hidden_layers=LAYERS)
if os.path.exists(CKPT):
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    print("loaded", CKPT, flush=True)
else:
    print("training figure-8 observer (9x100, w=1/1/1) ...", flush=True)
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
    for ep in range(1, 2):
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

def ctrl(t, x):
    return full_control_v3(t, x, figure8_reference, yaw_reference)   # nominal (unaware)

base = np.load(DATASET); Xb = base["X"]
x0s = [Xb[i, 0, :] for i in range(40, 41)]   # 10 unseen test starts
t_eval = np.arange(0.0, 3.0, 0.01)

def eval_set(P_true):
    me, he = [], []
    for x0 in x0s:
        sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, ctrl(t, x), P_true),
                        (0.0, 3.0), x0, t_eval=t_eval, max_step=0.02)
        xt = sol.y.T
        Tt = torch.tensor(sol.t, dtype=torch.float32).unsqueeze(1)
        X0t = torch.tensor(np.tile(x0, (len(sol.t), 1)), dtype=torch.float32)
        with torch.no_grad():
            xh = model(Tt, X0t).numpy()
        rmse = np.sqrt(((xh - xt)**2).mean(axis=0))
        me.append(rmse[MEAS_IDX].mean()); he.append(rmse[HID].mean())
    return float(np.mean(me)), float(np.mean(he))

devs = [0.0]
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
with open("/tmp/smoke_f8.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["parameter", "metric"] + cols)
    for pname in params:
        w.writerow([pname, "measured RMSE"] + [f"{res[(pname, d)][0]:.4f}" for d in devs])
        w.writerow([pname, "hidden RMSE"]   + [f"{res[(pname, d)][1]:.4f}" for d in devs])
    w.writerow(["combined worst", "measured RMSE", "", "", f"{res[('mass', 0.0)][0]:.4f}", "", f"{cw[0]:.4f}"])
    w.writerow(["combined worst", "hidden RMSE",   "", "", f"{res[('mass', 0.0)][1]:.4f}", "", f"{cw[1]:.4f}"])
print("saved -> /tmp/smoke_f8.csv", flush=True)
