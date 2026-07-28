import sys, os, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))

import torch
import torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

# ---------------------------------------------------------------
# Architecture study (Farkane Table A.3 style), no-gain model.
# Loss weights fixed at the weight-study winner (Case 4):
#   w0 = 1.0, w_ode = 2.0, wy = 1.0
# Pass 1: width 128 fixed, depth in {2, 4, 6, 9}
# Pass 2: best depth fixed, width in {20, 64, 256}  (128 done in pass 1)
# For each config: final MSE_0 / MSE_g / MSE_y, TEST RMSE, params, time.
# ---------------------------------------------------------------

EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 500
W0, WODE, WY = 1.0, 2.0, 1.0

PASS1 = [(2, 128), (4, 128), (6, 128), (9, 128)]
PASS2_WIDTHS = [20, 64, 256]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print(f"Weights fixed at Case 4: w0={W0}, w_ode={WODE}, wy={WY}")

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
    torch.manual_seed(0)   # same init logic every config -> fair comparison
    model = PINNObserverV4(hidden=hidden, n_hidden_layers=layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
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
    return n_params, l_0.item(), l_g.item(), l_y.item(), rmse[MEAS_IDX].mean().item(), rmse[HID].mean().item()

os.makedirs("docs", exist_ok=True)
rows = []

def run_config(layers, hidden, tag):
    print(f"\n=== {tag}: {layers} layers x {hidden} neurons ===", flush=True)
    t0 = time.time()
    n_params, l0, lg, ly, mr, hr = train_one(layers, hidden)
    dt = time.time() - t0
    print(f"  -> params {n_params} | TEST meas RMSE {mr:.4f} | hidden RMSE {hr:.4f} | {dt:.0f}s", flush=True)
    rows.append([layers, hidden, n_params, f"{l0:.3e}", f"{lg:.3e}", f"{ly:.3e}",
                 f"{mr:.4f}", f"{hr:.4f}", f"{dt:.0f}"])
    return hr

# Pass 1: vary depth at width 128
pass1_results = {}
for (L, H) in PASS1:
    hr = run_config(L, H, "Pass 1")
    pass1_results[L] = hr

best_depth = min(pass1_results, key=pass1_results.get)
print(f"\nBest depth from pass 1: {best_depth} layers", flush=True)

# Pass 2: vary width at best depth
for H in PASS2_WIDTHS:
    run_config(best_depth, H, "Pass 2")

with open("docs/arch_study.csv", "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["layers","neurons","params","MSE_0","MSE_g","MSE_y",
                "test_meas_RMSE","test_hidden_RMSE","train_time_s"])
    w.writerows(rows)

best = min(rows, key=lambda r: float(r[7]))
print("\n================ SUMMARY ================", flush=True)
print(f"{'layers':>6} {'neur':>5} {'params':>8} {'MSE_0':>10} {'MSE_g':>10} {'MSE_y':>10} {'measRMSE':>9} {'hidRMSE':>9} {'time_s':>7}")
for r in rows:
    print(f"{r[0]:>6} {r[1]:>5} {r[2]:>8} {r[3]:>10} {r[4]:>10} {r[5]:>10} {r[6]:>9} {r[7]:>9} {r[8]:>7}")
print(f"\nBEST by hidden RMSE: {best[0]} layers x {best[1]} neurons -> hidden RMSE {best[7]}")
print("Saved docs/arch_study.csv")
