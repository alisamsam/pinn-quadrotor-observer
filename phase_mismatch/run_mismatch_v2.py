"""Model-mismatch study for controller_v2 + frozen observer pinn_4x100_v2.
Same 'unaware' protocol: true dynamics use perturbed parameters; controller_v2
stays nominal; the frozen observer (nominal-trained) is evaluated vs the true
states. Runs locally (generation + eval in memory). Writes docs/mismatch_v2_summary.csv.
"""
import sys, os, csv
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "phase1a"))
import numpy as np
import torch
from scipy.integrate import solve_ivp
from phase_mismatch.generate_perturbed import quadrotor_dynamics, NOMINAL
from core.controller_v2 import full_control_v2
from scenarios.spiral.spiral_reference import spiral_reference, yaw_reference
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import MEAS_IDX

HID = [i for i in range(12) if i not in MEAS_IDX]
model = PINNObserverV4(hidden=100, n_hidden_layers=4)
model.load_state_dict(torch.load("phase1a/pinn_4x100_v2.pth", map_location="cpu")); model.eval()

def ctrl(t, x):
    return full_control_v2(t, x, spiral_reference, yaw_reference)   # nominal (unaware)

base = np.load("datasets/spiral_v2_dataset.npz"); Xb = base["X"]
x0s = [Xb[i, 0, :] for i in range(40, 50)]   # 10 unseen test starts
t_eval = np.arange(0.0, 30.0, 0.01)

def eval_set(P_true):
    me, he = [], []
    for x0 in x0s:
        sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, ctrl(t, x), P_true),
                        (0.0, 30.0), x0, t_eval=t_eval, max_step=0.02)
        xt = sol.y.T
        Tt = torch.tensor(sol.t, dtype=torch.float32).unsqueeze(1)
        X0t = torch.tensor(np.tile(x0, (len(sol.t), 1)), dtype=torch.float32)
        with torch.no_grad():
            xh = model(Tt, X0t).numpy()
        rmse = np.sqrt(((xh - xt)**2).mean(axis=0))
        me.append(rmse[MEAS_IDX].mean()); he.append(rmse[HID].mean())
    return float(np.mean(me)), float(np.mean(he))

devs = [-0.20, -0.10, 0.0, 0.10, 0.20]
params = {"mass": ["m"], "inertia": ["Ix","Iy","Iz"], "arm": ["l"]}
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
for k in ["m","Ix","Iy","Iz","l"]: Pw[k] = NOMINAL[k]*1.20
cw = eval_set(Pw); print(f"combined +20%: meas {cw[0]:.4f} | hidden {cw[1]:.4f}", flush=True)

cols = ["-20%","-10%","nominal","+10%","+20%"]
os.makedirs("docs", exist_ok=True)
with open("docs/mismatch_v2_summary.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["parameter","metric"]+cols)
    for pname in params:
        w.writerow([pname,"measured RMSE"]+[f"{res[(pname,d)][0]:.4f}" for d in devs])
        w.writerow([pname,"hidden RMSE"]+[f"{res[(pname,d)][1]:.4f}" for d in devs])
    w.writerow(["combined worst","measured RMSE","","",f"{res[('mass',0.0)][0]:.4f}","",f"{cw[0]:.4f}"])
    w.writerow(["combined worst","hidden RMSE","","",f"{res[('mass',0.0)][1]:.4f}","",f"{cw[1]:.4f}"])
print("saved -> docs/mismatch_v2_summary.csv", flush=True)
