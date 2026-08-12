import sys, os, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))

import torch
import torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

# ---------------------------------------------------------------
# Farkane-recipe test on our spiral system (single configuration).
# Architecture: 9 hidden layers x 20 neurons (their Table A.3 best)
# Weights:      w0=1.5, w_ode=0.5, wy=1.0
# Metrics: best loss, convergence iteration (and epoch), RMSE and
# MAE (measured/hidden) on unseen flights, inference time, training time.
# ---------------------------------------------------------------

EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 100
W0, WODE, WY = 1.5, 0.5, 1.0
LAYERS, HIDDEN = 9, 20
SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)
print(f"Config: {LAYERS}x{HIDDEN}, w0={W0}, w_ode={WODE}, wy={WY}, seed={SEED}", flush=True)

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

torch.manual_seed(SEED)
model = PINNObserverV4(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params}", flush=True)
opt = torch.optim.Adam(model.parameters(), lr=LR)

best_loss = float("inf")
best_iteration = 0
best_epoch = 0
iteration = 0

t_train0 = time.time()
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
        loss = WY*l_y + WODE*l_g + W0*l_0
        loss.backward()
        opt.step()
        iteration += 1
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_iteration = iteration
            best_epoch = epoch
    if epoch == 1 or epoch % LOG_EVERY == 0:
        print(f"    epoch {epoch:4d} | MSE_0 {l_0.item():.3e} | MSE_g {l_g.item():.3e} | MSE_y {l_y.item():.3e}", flush=True)
train_time = time.time() - t_train0

# ---- evaluation on the 10 unseen flights ----
model.eval()
with torch.no_grad():
    Xp = model(Tt, X0t)
    err = Xp - Xt
    rmse = torch.sqrt((err**2).mean(dim=0))
    mae  = err.abs().mean(dim=0)
rmse_meas, rmse_hid = rmse[MEAS_IDX].mean().item(), rmse[HID].mean().item()
mae_meas,  mae_hid  = mae[MEAS_IDX].mean().item(),  mae[HID].mean().item()

# ---- inference time: average over 10 timed passes on the test set ----
with torch.no_grad():
    for _ in range(3):                       # warm-up passes
        _ = model(Tt, X0t)
    if device.type == "cuda": torch.cuda.synchronize()
    t_inf0 = time.perf_counter()
    REPS = 10
    for _ in range(REPS):
        _ = model(Tt, X0t)
    if device.type == "cuda": torch.cuda.synchronize()
    t_inf = (time.perf_counter() - t_inf0) / REPS
n_test = Tt.shape[0]
inf_per_sample_us = t_inf / n_test * 1e6     # microseconds per sample

os.makedirs("docs", exist_ok=True)
with open("docs/farkane_recipe.csv", "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["layers","neurons","w0","w_ode","wy","params",
                "best_loss","conv_iteration","conv_epoch",
                "rmse_meas","rmse_hidden","mae_meas","mae_hidden",
                "inference_us_per_sample","train_time_s"])
    w.writerow([LAYERS, HIDDEN, W0, WODE, WY, n_params,
                f"{best_loss:.4e}", best_iteration, best_epoch,
                f"{rmse_meas:.4f}", f"{rmse_hid:.4f}",
                f"{mae_meas:.4f}", f"{mae_hid:.4f}",
                f"{inf_per_sample_us:.2f}", f"{train_time:.0f}"])

print("\n================ RESULT ================", flush=True)
print(f"  best loss:            {best_loss:.4e}", flush=True)
print(f"  convergence iteration:{best_iteration}  (epoch {best_epoch} of {EPOCHS})", flush=True)
print(f"  RMSE  measured/hidden: {rmse_meas:.4f} / {rmse_hid:.4f}", flush=True)
print(f"  MAE   measured/hidden: {mae_meas:.4f} / {mae_hid:.4f}", flush=True)
print(f"  inference: {inf_per_sample_us:.2f} microseconds per sample "
      f"({t_inf*1000:.1f} ms for {n_test} samples)", flush=True)
print(f"  training time: {train_time:.0f} s", flush=True)
print("Saved docs/farkane_recipe.csv", flush=True)