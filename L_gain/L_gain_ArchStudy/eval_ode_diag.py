# -*- coding: utf-8 -*-
"""Diagnose the observer-ODE integration (Farkane inference) on ONE unseen test flight:
   (1) magnitude of the learned gain L(t), (2) does the integration diverge, and does a
   finer integration step fix it (step-size / stiffness check)."""
import sys, os, torch
ROOT = os.getcwd()
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "phase1a"))
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch
from observer_v2_L_gain import build_C, integrate_observer

DATASET = "datasets/spiral_v2_dataset.npz"
CKPT = "phase1a/pinn_4x100_v2_L.pth"
device = "cpu"                                   # sequential RK loop; CPU is fine
HID = [i for i in range(12) if i not in MEAS_IDX]
N = 3000                                         # full flight

model = PINNObserverV5(hidden=100, n_hidden_layers=4).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device)); model.eval()
_, test = load_phase4_data(DATASET); C = build_C(MEAS_IDX, device)

sl = slice(0, N)
T = test["T"][sl]; U = test["U"][sl]; Y = test["Y"][sl]
Xtrue = test["X"][sl]; x0 = test["X0"][sl][0]

with torch.no_grad():
    _, L_all = model.get_state_and_gain(T, x0.view(1, 12).repeat(N, 1))
Lnorm = L_all.flatten(1).norm(dim=1)
print(f"||L(t)|| Frobenius: mean {Lnorm.mean():.3f} | max {Lnorm.max():.3f} | min {Lnorm.min():.3f}", flush=True)

def rmse(a, b):
    return torch.sqrt(((a - b) ** 2).mean(dim=0))

for ss in [1, 5, 20]:
    traj, x_hat_direct, div = integrate_observer(
        model, quadrotor_dynamics_torch, T, U, Y, x0, C, device, substeps=ss)
    if torch.isfinite(traj).all().item():
        r = rmse(traj, Xtrue)
        print(f"substeps {ss:3d} (dt={0.01/ss:.4f}s): STABLE  | meas {r[MEAS_IDX].mean():.4f} | hidden {r[HID].mean():.4f}", flush=True)
    else:
        print(f"substeps {ss:3d} (dt={0.01/ss:.4f}s): DIVERGED at t={div:.2f}s", flush=True)

rd = rmse(model.get_state_and_gain(T, x0.view(1,12).repeat(N,1))[0].detach(), Xtrue)
print(f"(reference) DIRECT network output: meas {rd[MEAS_IDX].mean():.4f} | hidden {rd[HID].mean():.4f}", flush=True)
