import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))

import torch
import torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

# Retrain the validated 4x100 observer on the IMPROVED controller_v2 dataset.
# Same architecture and best weights (Case 2). New dataset + new model file.
EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 100
LAYERS, HIDDEN = 4, 100
W0, WODE, WY = 0.5, 1.5, 1.0
SEED = 0
DATASET = "datasets/spiral_v2_dataset.npz"
SAVE_PATH = "phase1a/pinn_4x100_v2.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)
print(f"Training {LAYERS}x{HIDDEN} on {DATASET}, weights ({W0},{WODE},{WY})", flush=True)

train, test = load_phase4_data(DATASET)
T  = train["T"].to(device);  X0 = train["X0"].to(device)
Y  = train["Y"].to(device);  U  = train["U"].to(device); X = train["X"].to(device)
N  = T.shape[0]
t0_mask = (T[:, 0] == T[:, 0].min())
T0, X0_0, Xtrue_0 = T[t0_mask], X0[t0_mask], X[t0_mask]
Tt, X0t, Xt = test["T"].to(device), test["X0"].to(device), test["X"].to(device)
HID = [i for i in range(12) if i not in MEAS_IDX]

def physics_residual(model, t, x0, u):
    t = t.clone().requires_grad_(True)
    x_hat = model(t, x0)
    dxdt = torch.zeros_like(x_hat)
    for i in range(12):
        g = torch.autograd.grad(x_hat[:, i].sum(), t, create_graph=True)[0]
        dxdt[:, i] = g[:, 0]
    f = quadrotor_dynamics_torch(x_hat, u)
    return dxdt - f

torch.manual_seed(SEED)
model = PINNObserverV4(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

t_start = time.time()
for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(N, device=device)
    for s in range(0, N, BATCH):
        idx = perm[s:s+BATCH]
        tb, x0b, yb, ub = T[idx], X0[idx], Y[idx], U[idx]
        opt.zero_grad()
        x_hat = model(tb, x0b)
        l_y = nn.functional.mse_loss(x_hat[:, MEAS_IDX], yb)
        res = physics_residual(model, tb, x0b, ub)
        l_g = nn.functional.mse_loss(res, torch.zeros_like(res))
        x0p = model(T0, X0_0)
        l_0 = nn.functional.mse_loss(x0p, Xtrue_0)
        (WY*l_y + WODE*l_g + W0*l_0).backward()
        opt.step()
    if epoch == 1 or epoch % LOG_EVERY == 0:
        print(f"    epoch {epoch:4d} | MSE_0 {l_0.item():.3e} | MSE_g {l_g.item():.3e} | MSE_y {l_y.item():.3e}", flush=True)

torch.save(model.state_dict(), SAVE_PATH)
train_time = time.time() - t_start
model.eval()
with torch.no_grad():
    Xp = model(Tt, X0t)
    rmse = torch.sqrt(((Xp - Xt)**2).mean(dim=0))
print(f"\nSaved model to {SAVE_PATH}", flush=True)
print(f"TEST meas RMSE {rmse[MEAS_IDX].mean().item():.4f} | hidden RMSE {rmse[HID].mean().item():.4f} | {train_time:.0f}s", flush=True)
