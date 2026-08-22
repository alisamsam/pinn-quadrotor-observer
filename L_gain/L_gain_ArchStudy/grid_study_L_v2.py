import sys, os, csv, time
# make project root and phase1a importable (this file lives in L_gain_ArchStudy/)
ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "phase1a"))
import torch
import torch.nn as nn
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

# ------------------------------------------------------------------------------
# Architecture grid WITH the Farkane adaptive gain L (L-ON), controller_v2 dataset.
# Mirrors Layers_Neurons_Simulation/grid_study_v2.py (L-OFF): SAME grid, SAME neutral
# weights (1,1,1), SAME epochs/lr/batch/seed -> the two CSVs are directly comparable.
# Only change vs L-OFF: residual uses  g = dx/dt - f - L*(y - x_hat_meas)  and
# the network is PINNObserverV5 (outputs x_hat AND gain L).
# ------------------------------------------------------------------------------
EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 500
W0, WODE, WY = 1.0, 1.0, 1.0            # neutral weights, to match the L-OFF grid
SEED = 0
LAYERS_LIST = [4, 9, 12]
NEURONS_LIST = [20, 60, 100, 128]
DATASET = "datasets/spiral_v2_dataset.npz"
CSV_PATH = "docs/grid_study_L_v2.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| dataset:", DATASET, "| L-ON", flush=True)
train, test = load_phase4_data(DATASET)
T = train["T"].to(device); X0 = train["X0"].to(device); Y = train["Y"].to(device)
U = train["U"].to(device); X = train["X"].to(device)
N = T.shape[0]
t0 = (T[:, 0] == T[:, 0].min()); T0, X0_0, Xtrue_0 = T[t0], X0[t0], X[t0]
Tt, X0t, Xt = test["T"].to(device), test["X0"].to(device), test["X"].to(device)
HID = [i for i in range(12) if i not in MEAS_IDX]

# Explicit measurement (output) matrix C, size 6 x 12, built from the real MEAS_IDX.
# C selects the six measured states, so C x = y and C x_hat = the measured estimate.
C = torch.zeros(len(MEAS_IDX), 12, device=device)
for _row, _col in enumerate(MEAS_IDX):
    C[_row, _col] = 1.0

def residual_farkane(model, t, x0, u, y_meas):
    # Farkane's observer residual, written in full (no compressed form):
    #   g(t) = x_hat_dot(t) - f(x_hat, u) - L(t) . C . (x - x_hat)
    # For the quadrotor the control is state-dependent, so it stays inside f(x_hat, u)
    # ( f = f0(x_hat) + G(x_hat) u ); a constant B u term would be physically wrong.
    # The true state x is unknown at run time, but C x = y (the measurement), so
    #   C (x - x_hat) = y - C x_hat.
    t = t.clone().requires_grad_(True)
    x_hat, L = model.get_state_and_gain(t, x0)                 # x_hat (B,12), L (B,12,6)
    x_hat_dot = torch.zeros_like(x_hat)
    for i in range(12):
        gi = torch.autograd.grad(x_hat[:, i].sum(), t, create_graph=True)[0]
        x_hat_dot[:, i] = gi[:, 0]                             # d/dt x_hat_i
    f = quadrotor_dynamics_torch(x_hat, u)                     # f(x_hat, u) = f0(x_hat) + G(x_hat) u
    C_x_hat = x_hat @ C.t()                                    # C x_hat            (B,6)
    innovation = (y_meas - C_x_hat).unsqueeze(-1)              # C (x - x_hat)      (B,6,1)
    correction = torch.bmm(L, innovation).squeeze(-1)          # L . C (x - x_hat)  (B,12)
    g = x_hat_dot - f - correction                            # observer residual  (B,12)
    return x_hat, g

def train_one(layers, hidden):
    torch.manual_seed(SEED)
    model = PINNObserverV5(hidden=hidden, n_hidden_layers=layers).to(device)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, BATCH):
            idx = perm[s:s+BATCH]
            tb, x0b, yb, ub = T[idx], X0[idx], Y[idx], U[idx]
            opt.zero_grad()
            x_hat, res = residual_farkane(model, tb, x0b, ub, yb)
            ly = nn.functional.mse_loss(x_hat[:, MEAS_IDX], yb)
            lg = nn.functional.mse_loss(res, torch.zeros_like(res))
            x0p, _ = model.get_state_and_gain(T0, X0_0)
            l0 = nn.functional.mse_loss(x0p, Xtrue_0)
            (WY*ly + WODE*lg + W0*l0).backward()
            opt.step()
        if ep == 1 or ep % LOG_EVERY == 0:
            print(f"    ep {ep:4d} | MSE_0 {l0.item():.3e} | MSE_g {lg.item():.3e} | MSE_y {ly.item():.3e}", flush=True)
    model.eval()
    with torch.no_grad():
        Xp, _ = model.get_state_and_gain(Tt, X0t)
        rmse = torch.sqrt(((Xp - Xt)**2).mean(dim=0))
    return npar, rmse[MEAS_IDX].mean().item(), rmse[HID].mean().item()

os.makedirs("docs", exist_ok=True)
with open(CSV_PATH, "w", newline="") as fp:
    csv.writer(fp).writerow(["layers", "neurons", "params", "rmse_meas", "rmse_hidden", "train_time_s"])
rows = []
for Ly in LAYERS_LIST:
    for H in NEURONS_LIST:
        print(f"\n=== {Ly} layers x {H} neurons (L-ON) ===", flush=True)
        t = time.time()
        p, mr, hr = train_one(Ly, H); dt = time.time() - t
        print(f"  -> params {p} | RMSE meas {mr:.4f} hidden {hr:.4f} | {dt:.0f}s", flush=True)
        row = [Ly, H, p, f"{mr:.4f}", f"{hr:.4f}", f"{dt:.0f}"]; rows.append(row)
        with open(CSV_PATH, "a", newline="") as fp:
            csv.writer(fp).writerow(row)
best = min(rows, key=lambda r: float(r[4]))
print(f"\nBEST (L-ON) by hidden RMSE: {best[0]} layers x {best[1]} neurons -> {best[4]}", flush=True)
print(f"Saved {CSV_PATH}", flush=True)
