"""Combined-worst mismatch (v2): each parameter at its individually-worst direction
(mass -20%, inertia +20%, arm -20%). Prints the RMSE; used to fix the summary CSV."""
import sys, os
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, "phase1a"))
import numpy as np, torch
from scipy.integrate import solve_ivp
from phase_mismatch.generate_perturbed import quadrotor_dynamics, NOMINAL
from core.controller_v2 import full_control_v2
from scenarios.spiral.spiral_reference import spiral_reference, yaw_reference
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import MEAS_IDX

HID = [i for i in range(12) if i not in MEAS_IDX]
model = PINNObserverV4(hidden=100, n_hidden_layers=4)
model.load_state_dict(torch.load("phase1a/pinn_4x100_v2.pth", map_location="cpu")); model.eval()
ctrl = lambda t, x: full_control_v2(t, x, spiral_reference, yaw_reference)
Xb = np.load("datasets/spiral_v2_dataset.npz")["X"]; x0s = [Xb[i,0,:] for i in range(40,50)]
t_eval = np.arange(0.0, 30.0, 0.01)

P = dict(NOMINAL)
P["m"] *= 0.80                                   # mass worst = -20%
P["Ix"] *= 1.20; P["Iy"] *= 1.20; P["Iz"] *= 1.20  # inertia worst = +20%
P["l"] *= 0.80                                   # arm worst = -20%

me, he = [], []
for x0 in x0s:
    sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, ctrl(t, x), P),
                    (0.0, 30.0), x0, t_eval=t_eval, max_step=0.02)
    xt = sol.y.T
    Tt = torch.tensor(sol.t, dtype=torch.float32).unsqueeze(1)
    X0t = torch.tensor(np.tile(x0, (len(sol.t),1)), dtype=torch.float32)
    with torch.no_grad(): xh = model(Tt, X0t).numpy()
    rmse = np.sqrt(((xh - xt)**2).mean(axis=0))
    me.append(rmse[MEAS_IDX].mean()); he.append(rmse[HID].mean())
print(f"COMBINED_WORST meas {np.mean(me):.4f} hidden {np.mean(he):.4f}", flush=True)
