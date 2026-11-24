import sys, os, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))

import numpy as np
import torch
import torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

# Noise-robustness study, PER-STATE + NRMSE version (reviewer comment [2]).
# Gaussian sensor noise is added to the MEASURED outputs y used in TRAINING;
# evaluation is always vs the TRUE states of the unseen flights. 4x100, (0.5,1.5,1.0).
EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 500
LAYERS, HIDDEN = 4, 100
W0, WODE, WY = 0.5, 1.5, 1.0
SEED = 0
DATASET = "datasets/spiral_v2_dataset.npz"
CASES = [("clean", 0.00, 0.00), ("moderate", 0.05, 0.01), ("strong", 0.10, 0.02)]
POS_IDX_IN_Y = [0, 1, 2]
ATT_IDX_IN_Y = [3, 4, 5]
NAMES = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

train, test = load_phase4_data(DATASET)
T  = train["T"].to(device);  X0 = train["X0"].to(device)
Yc = train["Y"].to(device);  U  = train["U"].to(device); X = train["X"].to(device)
N  = T.shape[0]
t0_mask = (T[:, 0] == T[:, 0].min())
T0, X0_0, Xtrue_0 = T[t0_mask], X0[t0_mask], X[t0_mask]
Tt, X0t, Xt = test["T"].to(device), test["X0"].to(device), test["X"].to(device)
HID = [i for i in range(12) if i not in MEAS_IDX]

# per-state ranges from the TRUE test states (for NRMSE)
Xt_np = Xt.cpu().numpy()
RANGE = Xt_np.max(axis=0) - Xt_np.min(axis=0)          # (12,)

def physics_residual(model, t, x0, u):
    t = t.clone().requires_grad_(True)
    x_hat = model(t, x0)
    dxdt = torch.zeros_like(x_hat)
    for i in range(12):
        g = torch.autograd.grad(x_hat[:, i].sum(), t, create_graph=True)[0]
        dxdt[:, i] = g[:, 0]
    return dxdt - quadrotor_dynamics_torch(x_hat, u)

def train_one(sig_pos, sig_att):
    torch.manual_seed(SEED)
    Y = Yc.clone()
    if sig_pos > 0:
        Y[:, POS_IDX_IN_Y] += sig_pos * torch.randn_like(Y[:, POS_IDX_IN_Y])
    if sig_att > 0:
        Y[:, ATT_IDX_IN_Y] += sig_att * torch.randn_like(Y[:, ATT_IDX_IN_Y])
    model = PINNObserverV4(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
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
            l_0 = nn.functional.mse_loss(model(T0, X0_0), Xtrue_0)
            (WY*l_y + WODE*l_g + W0*l_0).backward()
            opt.step()
        if epoch == 1 or epoch % LOG_EVERY == 0:
            print(f"    epoch {epoch:4d} | MSE_y {l_y.item():.3e}", flush=True)
    model.eval()
    with torch.no_grad():
        rmse = torch.sqrt(((model(Tt, X0t) - Xt)**2).mean(dim=0)).cpu().numpy()  # (12,)
    return rmse

os.makedirs("docs", exist_ok=True)
out = "docs/noise_study_v2_perstate.csv"
with open(out, "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["case","sigma_pos_m","sigma_att_rad"] +
               [f"RMSE_{n}" for n in NAMES] +
               ["meas_NRMSE_pct","hidden_NRMSE_pct","overall_NRMSE_pct"])
    for name, sp, sa in CASES:
        print(f"\n=== {name}: sigma_pos={sp} m, sigma_att={sa} rad ===", flush=True)
        t0 = time.time()
        rmse = train_one(sp, sa)
        nr = rmse / RANGE
        meas_nr = 100*nr[MEAS_IDX].mean(); hid_nr = 100*nr[HID].mean(); all_nr = 100*nr.mean()
        print(f"  -> meas NRMSE {meas_nr:.2f}% | hidden NRMSE {hid_nr:.2f}% | overall {all_nr:.2f}% | {time.time()-t0:.0f}s", flush=True)
        w.writerow([name, sp, sa] + [f"{v:.4f}" for v in rmse] +
                   [f"{meas_nr:.2f}", f"{hid_nr:.2f}", f"{all_nr:.2f}"])
print(f"\nSaved {out}", flush=True)
