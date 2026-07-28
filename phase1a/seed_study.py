import sys, os, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))

import torch
import torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

# ---------------------------------------------------------------
# Seed-repetition study: the winning configuration retrained with
# 5 different random initializations (seeds), to report mean +/- std.
# Config fixed at the Phase-2 winner: 4 layers x 128 neurons,
# weights w0=1.0, w_ode=2.0, wy=1.0. Only the seed changes.
# ---------------------------------------------------------------

EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 500
W0, WODE, WY = 1.0, 2.0, 1.0
SEEDS = [0, 1, 2, 3, 4]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)
print(f"Config: 4x128, w0={W0}, w_ode={WODE}, wy={WY}, seeds={SEEDS}", flush=True)

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

def train_one(seed):
    torch.manual_seed(seed)
    model = PINNObserverV4().to(device)          # 4 x 128 default
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
    return rmse[MEAS_IDX].mean().item(), rmse[HID].mean().item()

os.makedirs("docs", exist_ok=True)
rows = []
for seed in SEEDS:
    print(f"\n=== Seed {seed} ===", flush=True)
    t0 = time.time()
    mr, hr = train_one(seed)
    dt = time.time() - t0
    print(f"  -> TEST meas RMSE {mr:.4f} | hidden RMSE {hr:.4f} | {dt:.0f}s", flush=True)
    rows.append([seed, f"{mr:.4f}", f"{hr:.4f}", f"{dt:.0f}"])

meas = [float(r[1]) for r in rows]; hid = [float(r[2]) for r in rows]
def mean(v): return sum(v)/len(v)
def std(v):
    m = mean(v); return (sum((x-m)**2 for x in v)/(len(v)-1))**0.5

with open("docs/seed_study.csv", "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["seed","test_meas_RMSE","test_hidden_RMSE","train_time_s"])
    w.writerows(rows)
    w.writerow([]); w.writerow(["mean", f"{mean(meas):.4f}", f"{mean(hid):.4f}", ""])
    w.writerow(["std",  f"{std(meas):.4f}",  f"{std(hid):.4f}",  ""])

print("\n================ SUMMARY ================", flush=True)
for r in rows: print(f"  seed {r[0]}: meas {r[1]} | hidden {r[2]}", flush=True)
print(f"\n  meas RMSE:   {mean(meas):.4f} +/- {std(meas):.4f}", flush=True)
print(f"  hidden RMSE: {mean(hid):.4f} +/- {std(hid):.4f}", flush=True)
print("Saved docs/seed_study.csv", flush=True)