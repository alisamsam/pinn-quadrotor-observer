# -*- coding: utf-8 -*-
"""Step 3 validation: integrate the observer ODE on ONE unseen test flight with the
already-trained L-model, and compare against the direct-output estimate."""
import sys, os, time, torch
ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "phase1a"))
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch
from observer_v2_L_gain import build_C, integrate_observer

DATASET = "datasets/spiral_v2_dataset.npz"
CKPT = "phase1a/pinn_4x100_v2_L.pth"
device = "cpu"
HID = [i for i in range(12) if i not in MEAS_IDX]
NSTEPS = 3000                      # one flight

model = PINNObserverV5(hidden=100, n_hidden_layers=4).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

_, test = load_phase4_data(DATASET)
C = build_C(MEAS_IDX, device)

sl = slice(0, NSTEPS)              # first test flight
T = test["T"][sl]; U = test["U"][sl]; Y = test["Y"][sl]
Xtrue = test["X"][sl]; x0 = test["X0"][sl][0]

t0 = time.time()
traj, x_hat_direct = integrate_observer(model, quadrotor_dynamics_torch, T, U, Y, x0, C, device)
dt = time.time() - t0

def rmse(a, b):
    return torch.sqrt(((a - b) ** 2).mean(dim=0))

r_int = rmse(traj, Xtrue)
r_dir = rmse(x_hat_direct, Xtrue)
finite = torch.isfinite(traj).all().item()

print(f"integration finite: {finite} | {dt:.1f}s")
print("Test flight #0 (unseen):")
print(f"  DIRECT output  : meas RMSE {r_dir[MEAS_IDX].mean():.4f} | hidden RMSE {r_dir[HID].mean():.4f}")
print(f"  ODE-integrated : meas RMSE {r_int[MEAS_IDX].mean():.4f} | hidden RMSE {r_int[HID].mean():.4f}")
