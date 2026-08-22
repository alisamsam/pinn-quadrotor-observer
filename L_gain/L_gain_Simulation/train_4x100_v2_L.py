"""Gain-L test in the [t, x0] architecture, on the controller_v2 dataset.
Uses PINNObserverV5 (outputs state x_hat AND adaptive gain L, 12x6). The physics
residual gets the Farkane correction  L * (y - y_hat).  Everything else matches the
v2 run (4x100, weights 0.5,1.5,1.0) so this is a clean L-ON vs L-OFF comparison
against pinn_4x100_v2.pth (L-OFF: meas 0.0442 / hidden 0.0559).
"""
import sys, os, time
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "phase1a"))

import torch
import torch.nn as nn
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 100
LAYERS, HIDDEN = 4, 100
W0, WODE, WY = 0.5, 1.5, 1.0
SEED = 0
DATASET = "datasets/spiral_v2_dataset.npz"
SAVE_PATH = "phase1a/pinn_4x100_v2_L.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)
print(f"Gain-L test: {LAYERS}x{HIDDEN} on {DATASET}, weights ({W0},{WODE},{WY})", flush=True)

train, test = load_phase4_data(DATASET)
T  = train["T"].to(device);  X0 = train["X0"].to(device)
Y  = train["Y"].to(device);  U  = train["U"].to(device); X = train["X"].to(device)
N  = T.shape[0]
t0_mask = (T[:, 0] == T[:, 0].min())
T0, X0_0, Xtrue_0 = T[t0_mask], X0[t0_mask], X[t0_mask]
Tt, X0t, Xt = test["T"].to(device), test["X0"].to(device), test["X"].to(device)
HID = [i for i in range(12) if i not in MEAS_IDX]

def residual_with_L(model, t, x0, u, y_meas):
    t = t.clone().requires_grad_(True)
    x_hat, L = model.get_state_and_gain(t, x0)       # x_hat (B,12), L (B,12,6)
    dxdt = torch.zeros_like(x_hat)
    for i in range(12):
        g = torch.autograd.grad(x_hat[:, i].sum(), t, create_graph=True)[0]
        dxdt[:, i] = g[:, 0]
    f = quadrotor_dynamics_torch(x_hat, u)
    innov = (y_meas - x_hat[:, MEAS_IDX]).unsqueeze(-1)   # (B,6,1)
    corr = torch.bmm(L, innov).squeeze(-1)               # (B,12)
    return x_hat, dxdt - f - corr

torch.manual_seed(SEED)
model = PINNObserverV5(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

t_start = time.time()
for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(N, device=device)
    for s in range(0, N, BATCH):
        idx = perm[s:s+BATCH]
        tb, x0b, yb, ub = T[idx], X0[idx], Y[idx], U[idx]
        opt.zero_grad()
        x_hat, res = residual_with_L(model, tb, x0b, ub, yb)
        l_y = nn.functional.mse_loss(x_hat[:, MEAS_IDX], yb)
        l_g = nn.functional.mse_loss(res, torch.zeros_like(res))
        x0p, _ = model.get_state_and_gain(T0, X0_0)
        l_0 = nn.functional.mse_loss(x0p, Xtrue_0)
        (WY*l_y + WODE*l_g + W0*l_0).backward()
        opt.step()
    if epoch == 1 or epoch % LOG_EVERY == 0:
        print(f"    epoch {epoch:4d} | MSE_0 {l_0.item():.3e} | MSE_g {l_g.item():.3e} | MSE_y {l_y.item():.3e}", flush=True)

torch.save(model.state_dict(), SAVE_PATH)
train_time = time.time() - t_start
model.eval()
with torch.no_grad():
    Xp, _ = model.get_state_and_gain(Tt, X0t)
    rmse = torch.sqrt(((Xp - Xt)**2).mean(dim=0))
print(f"\nSaved model to {SAVE_PATH}", flush=True)
print(f"L-ON  TEST meas RMSE {rmse[MEAS_IDX].mean().item():.4f} | hidden RMSE {rmse[HID].mean().item():.4f} | {train_time:.0f}s", flush=True)
print(f"compare L-OFF (pinn_4x100_v2): meas 0.0442 | hidden 0.0559", flush=True)
