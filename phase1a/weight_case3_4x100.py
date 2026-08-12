import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))

import torch
import torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 100
LAYERS, HIDDEN = 4, 100                 # YOUR architecture
W0, WODE, WY = 1.5, 0.5, 1.0            # Farkane Case 3 weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)
print(f"Config: {LAYERS}x{HIDDEN}, w0={W0}, w_ode={WODE}, wy={WY}", flush=True)

train, test = load_phase4_data()
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

torch.manual_seed(0)
model = PINNObserverV4(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
t0 = time.time()
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

model.eval()
with torch.no_grad():
    Xp = model(Tt, X0t)
    rmse = torch.sqrt(((Xp - Xt)**2).mean(dim=0))
mr, hr = rmse[MEAS_IDX].mean().item(), rmse[HID].mean().item()

print("\n================ RESULT ================", flush=True)
print(f"  Case 3 weights at 4x100: meas RMSE {mr:.4f} | hidden RMSE {hr:.4f} | {time.time()-t0:.0f}s", flush=True)
print(f"  Compare: your Case 2 (0.5,1.5,1.0) hidden = 0.0330; noise band +/-0.009", flush=True)