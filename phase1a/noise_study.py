import sys, os, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))

import torch
import torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

# ---------------------------------------------------------------
# Noise-robustness study, retained configuration (4x128, w0=1,
# w_ode=2, wy=1). Gaussian sensor noise is added to the MEASURED
# outputs y used in training (positions in metres, angles in rad).
# Evaluation is always against the TRUE states of unseen flights.
# Case A: clean (reference, sigma = 0)
# Case B: moderate noise  (pos 0.05 m, att 0.01 rad)
# Case C: strong noise    (pos 0.10 m, att 0.02 rad)
# ---------------------------------------------------------------

EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 500
W0, WODE, WY = 1.0, 2.0, 1.0
SEED = 0
CASES = [("clean", 0.00, 0.00), ("moderate", 0.05, 0.01), ("strong", 0.10, 0.02)]
POS_IDX_IN_Y = [0, 1, 2]   # first three measured outputs are x, y, z
ATT_IDX_IN_Y = [3, 4, 5]   # last three are phi, theta, psi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

train, test = load_phase4_data()
T  = train["T"].to(device);  X0 = train["X0"].to(device)
Yc = train["Y"].to(device);  U  = train["U"].to(device); X = train["X"].to(device)
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

def train_one(sig_pos, sig_att):
    torch.manual_seed(SEED)
    # noisy training measurements (fresh noise, fixed by the seed)
    Y = Yc.clone()
    if sig_pos > 0:
        Y[:, POS_IDX_IN_Y] += sig_pos * torch.randn_like(Y[:, POS_IDX_IN_Y])
    if sig_att > 0:
        Y[:, ATT_IDX_IN_Y] += sig_att * torch.randn_like(Y[:, ATT_IDX_IN_Y])

    model = PINNObserverV4().to(device)
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
        Xp = model(Tt, X0t)                    # evaluated against TRUE states
        rmse = torch.sqrt(((Xp - Xt)**2).mean(dim=0))
    return rmse[MEAS_IDX].mean().item(), rmse[HID].mean().item()

os.makedirs("docs", exist_ok=True)
rows = []
for name, sp, sa in CASES:
    print(f"\n=== {name}: sigma_pos={sp} m, sigma_att={sa} rad ===", flush=True)
    t0 = time.time()
    mr, hr = train_one(sp, sa)
    dt = time.time() - t0
    print(f"  -> TEST meas RMSE {mr:.4f} | hidden RMSE {hr:.4f} | {dt:.0f}s", flush=True)
    rows.append([name, sp, sa, f"{mr:.4f}", f"{hr:.4f}", f"{dt:.0f}"])

with open("docs/noise_study.csv", "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["case","sigma_pos_m","sigma_att_rad","test_meas_RMSE","test_hidden_RMSE","train_time_s"])
    w.writerows(rows)

print("\n================ SUMMARY ================", flush=True)
for r in rows:
    print(f"  {r[0]:>9}: meas {r[3]} | hidden {r[4]}", flush=True)
print("Saved docs/noise_study.csv", flush=True)