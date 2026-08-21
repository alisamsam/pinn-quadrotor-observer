import sys, os, csv, time
# project root and phase1a importable (this file lives in L_gain_ArchStudy/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "phase1a"))
import torch
import torch.nn as nn
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch
from observer_v2_L_gain import build_C, integrate_observer

# ------------------------------------------------------------------------------
# PHASE 1 (L-ON): architecture study evaluated the AUTHENTIC Farkane way.
# For every layers x neurons cell: train the L-observer (residual with the gain,
# weights 1,1,1 -- identical to the L-OFF grid), then EVALUATE BY ODE INTEGRATION
# (observer_v2_L_gain.integrate_observer), not by reading x_hat off the network.
# Per cell we log: ||L||, how many of the 10 test flights stay stable, the hidden
# RMSE where stable, and the direct-output RMSE for reference.
# ------------------------------------------------------------------------------
EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 500
W0, WODE, WY = 1.0, 1.0, 1.0               # neutral weights, matches the L-OFF grid
SEED = 0
LAYERS_LIST = [4, 9, 12]
NEURONS_LIST = [20, 60, 100, 128]
DATASET = "datasets/spiral_v2_dataset.npz"
CSV_PATH = "docs/grid_study_L_integrated.csv"
NSTEPS = 3000                               # steps per flight
N_TEST_FLIGHTS = 10                         # test set = 10 flights (rows in 3000-blocks)
SUBSTEPS = 1                                # RK4 at dt=0.01 (verdict is step-size independent)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| dataset:", DATASET, "| PHASE 1 L-ON (integrated eval)", flush=True)

train, test = load_phase4_data(DATASET)
T = train["T"].to(device); X0 = train["X0"].to(device); Y = train["Y"].to(device)
U = train["U"].to(device); X = train["X"].to(device)
N = T.shape[0]
t0 = (T[:, 0] == T[:, 0].min()); T0, X0_0, Xtrue_0 = T[t0], X0[t0], X[t0]
HID = [i for i in range(12) if i not in MEAS_IDX]
C_dev = build_C(MEAS_IDX, device)           # for training residual
C_cpu = build_C(MEAS_IDX, "cpu")            # for integration (done on CPU)

def residual_farkane(model, t, x0, u, y_meas):
    # g = x_hat_dot - f(x_hat,u) - L . C (x - x_hat),  with C(x - x_hat) = y - C x_hat
    t = t.clone().requires_grad_(True)
    x_hat, L = model.get_state_and_gain(t, x0)
    x_hat_dot = torch.zeros_like(x_hat)
    for i in range(12):
        gi = torch.autograd.grad(x_hat[:, i].sum(), t, create_graph=True)[0]
        x_hat_dot[:, i] = gi[:, 0]
    f = quadrotor_dynamics_torch(x_hat, u)
    C_x_hat = x_hat @ C_dev.t()
    innovation = (y_meas - C_x_hat).unsqueeze(-1)
    correction = torch.bmm(L, innovation).squeeze(-1)
    return x_hat, x_hat_dot - f - correction

def train_one(layers, hidden):
    torch.manual_seed(SEED)
    model = PINNObserverV5(hidden=hidden, n_hidden_layers=layers).to(device)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, BATCH):
            idx = perm[s:s+BATCH]; tb, x0b, yb, ub = T[idx], X0[idx], Y[idx], U[idx]
            opt.zero_grad()
            x_hat, res = residual_farkane(model, tb, x0b, ub, yb)
            ly = nn.functional.mse_loss(x_hat[:, MEAS_IDX], yb)
            lg = nn.functional.mse_loss(res, torch.zeros_like(res))
            x0p, _ = model.get_state_and_gain(T0, X0_0)
            l0 = nn.functional.mse_loss(x0p, Xtrue_0)
            (WY*ly + WODE*lg + W0*l0).backward(); opt.step()
        if ep == 1 or ep % LOG_EVERY == 0:
            print(f"    ep {ep:4d} | MSE_0 {l0.item():.3e} | MSE_g {lg.item():.3e} | MSE_y {ly.item():.3e}", flush=True)
    return model, npar

def eval_all(model):
    """Direct-output RMSE (reference) + ODE-integrated evaluation over the 10 test flights."""
    model = model.to("cpu"); model.eval()
    # direct-output reference on the whole test set
    with torch.no_grad():
        xh_dir, _ = model.get_state_and_gain(test["T"], test["X0"])
        rd = torch.sqrt(((xh_dir - test["X"])**2).mean(dim=0))
    meas_dir = rd[MEAS_IDX].mean().item(); hid_dir = rd[HID].mean().item()
    # integrated evaluation, flight by flight
    n_stable = 0; hid_s, meas_s, divt, Lnorms = [], [], [], []
    for j in range(N_TEST_FLIGHTS):
        s = j*NSTEPS; e = s+NSTEPS
        Tf, Uf, Yf, Xf = test["T"][s:e], test["U"][s:e], test["Y"][s:e], test["X"][s:e]
        x0f = test["X0"][s]
        traj, _, div = integrate_observer(model, quadrotor_dynamics_torch, Tf, Uf, Yf, x0f, C_cpu, "cpu", substeps=SUBSTEPS)
        with torch.no_grad():
            _, L_all = model.get_state_and_gain(Tf, x0f.view(1, 12).repeat(NSTEPS, 1))
        Lnorms.append(L_all.flatten(1).norm(dim=1).mean().item())
        if torch.isfinite(traj).all():
            n_stable += 1
            r = torch.sqrt(((traj - Xf)**2).mean(dim=0))
            meas_s.append(r[MEAS_IDX].mean().item()); hid_s.append(r[HID].mean().item())
        else:
            divt.append(div)
    Ln = sum(Lnorms)/len(Lnorms)
    meas_int = (sum(meas_s)/len(meas_s)) if meas_s else float("nan")
    hid_int = (sum(hid_s)/len(hid_s)) if hid_s else float("nan")
    avgdiv = (sum(divt)/len(divt)) if divt else float("nan")
    model.to(device)
    return Ln, n_stable, meas_int, hid_int, avgdiv, meas_dir, hid_dir

os.makedirs("docs", exist_ok=True)
HEADER = ["layers", "neurons", "params", "L_norm_mean", "n_stable_of_10",
          "meas_rmse_int", "hidden_rmse_int", "avg_diverge_t_s",
          "meas_rmse_direct", "hidden_rmse_direct", "train_time_s"]
with open(CSV_PATH, "w", newline="") as fp:
    csv.writer(fp).writerow(HEADER)

for Ly in LAYERS_LIST:
    for H in NEURONS_LIST:
        print(f"\n=== {Ly} layers x {H} neurons (L-ON, integrated eval) ===", flush=True)
        t = time.time()
        model, npar = train_one(Ly, H)
        Ln, nstab, mint, hint, adiv, mdir, hdir = eval_all(model)
        dt = time.time() - t
        print(f"  -> params {npar} | ||L|| {Ln:.2f} | stable {nstab}/10 | "
              f"int(meas {mint:.4f}, hidden {hint:.4f}) | direct(meas {mdir:.4f}, hidden {hdir:.4f}) | {dt:.0f}s", flush=True)
        row = [Ly, H, npar, f"{Ln:.3f}", nstab, f"{mint:.4f}", f"{hint:.4f}",
               f"{adiv:.2f}", f"{mdir:.4f}", f"{hdir:.4f}", f"{dt:.0f}"]
        with open(CSV_PATH, "a", newline="") as fp:
            csv.writer(fp).writerow(row)

print(f"\nSaved {CSV_PATH}", flush=True)
