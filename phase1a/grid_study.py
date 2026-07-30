import sys, os, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))

import torch
import torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

# ---------------------------------------------------------------
# Full grid study on NEUTRAL weights (w0 = w_ode = wy = 1.0):
# layers in {4, 9, 12} x neurons in {20, 60, 100, 128} = 12 runs.
# Metrics per run: params, best loss, convergence iteration/epoch,
# RMSE and MAE (measured/hidden) on unseen flights, training time.
# ---------------------------------------------------------------

EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 500
W0, WODE, WY = 1.0, 1.0, 1.0
SEED = 0
LAYERS_LIST  = [4, 9, 12]
NEURONS_LIST = [20, 60, 100, 128]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)
print(f"Neutral weights: w0={W0}, w_ode={WODE}, wy={WY}, seed={SEED}", flush=True)

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

def train_one(layers, hidden):
    torch.manual_seed(SEED)
    model = PINNObserverV4(hidden=hidden, n_hidden_layers=layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_loss, best_it, best_ep, it = float("inf"), 0, 0, 0
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
            it += 1
            if loss.item() < best_loss:
                best_loss, best_it, best_ep = loss.item(), it, epoch
        if epoch == 1 or epoch % LOG_EVERY == 0:
            print(f"    epoch {epoch:4d} | MSE_0 {l_0.item():.3e} | MSE_g {l_g.item():.3e} | MSE_y {l_y.item():.3e}", flush=True)
    model.eval()
    with torch.no_grad():
        Xp = model(Tt, X0t)
        err = Xp - Xt
        rmse = torch.sqrt((err**2).mean(dim=0))
        mae  = err.abs().mean(dim=0)
    return (n_params, best_loss, best_it, best_ep,
            rmse[MEAS_IDX].mean().item(), rmse[HID].mean().item(),
            mae[MEAS_IDX].mean().item(),  mae[HID].mean().item())

os.makedirs("docs", exist_ok=True)
rows = []
for L in LAYERS_LIST:
    for H in NEURONS_LIST:
        print(f"\n=== {L} layers x {H} neurons ===", flush=True)
        t0 = time.time()
        p, bl, bi, be, mr, hr, mm, mh = train_one(L, H)
        dt = time.time() - t0
        print(f"  -> params {p} | best loss {bl:.3e} at it {bi} (ep {be}) | "
              f"RMSE {mr:.4f}/{hr:.4f} | MAE {mm:.4f}/{mh:.4f} | {dt:.0f}s", flush=True)
        rows.append([L, H, p, f"{bl:.4e}", bi, be,
                     f"{mr:.4f}", f"{hr:.4f}", f"{mm:.4f}", f"{mh:.4f}", f"{dt:.0f}"])

with open("docs/grid_study.csv", "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["layers","neurons","params","best_loss","conv_iteration","conv_epoch",
                "rmse_meas","rmse_hidden","mae_meas","mae_hidden","train_time_s"])
    w.writerows(rows)

best = min(rows, key=lambda r: float(r[7]))
print("\n================ SUMMARY ================", flush=True)
print(f"{'L':>3} {'H':>4} {'params':>8} {'best_loss':>11} {'conv_it':>8} {'RMSEm':>7} {'RMSEh':>7} {'MAEm':>7} {'MAEh':>7} {'time':>6}", flush=True)
for r in rows:
    print(f"{r[0]:>3} {r[1]:>4} {r[2]:>8} {r[3]:>11} {r[4]:>8} {r[6]:>7} {r[7]:>7} {r[8]:>7} {r[9]:>7} {r[10]:>6}", flush=True)
print(f"\nBEST by hidden RMSE: {best[0]} layers x {best[1]} neurons -> {best[7]}", flush=True)
print("Saved docs/grid_study.csv", flush=True)