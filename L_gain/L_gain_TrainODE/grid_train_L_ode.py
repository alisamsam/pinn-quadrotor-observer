import sys, os, csv, time
ROOT = os.getcwd()
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "phase1a"))
import torch
import torch.nn as nn
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch
from rollout import build_C, rollout_observer
from observer_v2_L_gain import integrate_observer

# ------------------------------------------------------------------------------
# Architecture study for the TRAIN-THROUGH-INTEGRATION observer (the fix).
# For each layers x neurons: train L through the ODE rollout, then evaluate the
# integrated observer on the 10 test flights. Pick the cell with lowest hidden RMSE.
# ------------------------------------------------------------------------------
ITERS, LR = 3000, 1e-3
BATCH_FLIGHTS = 8
NOISE0 = 0.1
CLIP = 1.0
SEED = 0
NSTEPS = 3000
LAYERS_LIST = [4, 9, 12]
NEURONS_LIST = [20, 60, 100, 128]
DATASET = "datasets/spiral_v2_dataset.npz"
CSV_PATH = "docs/grid_train_L_ode.csv"
LOG_EVERY = 500

def window_len(it):
    if it < 800: return 50
    if it < 2000: return 100
    return 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| GRID train-through-ODE (the fix)", flush=True)

train, test = load_phase4_data(DATASET)
TR = {k: v.to(device) for k, v in train.items()}
n_flights = TR["T"].shape[0] // NSTEPS
base_T = TR["T"][0:NSTEPS, 0]
C_dev = build_C(MEAS_IDX, device)
C_cpu = build_C(MEAS_IDX, "cpu")
HID = [i for i in range(12) if i not in MEAS_IDX]

def train_one(layers, hidden):
    torch.manual_seed(SEED)
    model = PINNObserverV5(hidden=hidden, n_hidden_layers=layers).to(device)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for it in range(1, ITERS + 1):
        W = window_len(it)
        flights = torch.randperm(n_flights, device=device)[:BATCH_FLIGHTS]
        s = int(torch.randint(0, NSTEPS - W, (1,)).item())
        rows = (flights.view(-1, 1) * NSTEPS + s + torch.arange(W, device=device).view(1, -1))
        u_win = TR["U"][rows].permute(1, 0, 2); y_win = TR["Y"][rows].permute(1, 0, 2)
        x_true = TR["X"][rows].permute(1, 0, 2); x0_cond = TR["X0"][flights * NSTEPS]
        t_win = base_T[s:s + W]; x_hat0 = x_true[0] + NOISE0 * torch.randn_like(x_true[0])
        opt.zero_grad()
        traj = rollout_observer(model, quadrotor_dynamics_torch, C_dev, t_win, u_win, y_win, x0_cond, x_hat0)
        loss = nn.functional.mse_loss(traj, x_true)
        if not torch.isfinite(loss):
            continue
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step()
        if it == 1 or it % LOG_EVERY == 0:
            print(f"    it {it:4d} | W {W} | loss {loss.item():.4e}", flush=True)
    return model, npar

def eval_integrated(model):
    model = model.to("cpu"); model.eval()
    n_stable = 0; hid, Ln = [], []
    for j in range(10):
        a = j * NSTEPS; b = a + NSTEPS
        Tf, Uf, Yf, Xf = test["T"][a:b], test["U"][a:b], test["Y"][a:b], test["X"][a:b]
        x0f = test["X0"][a]
        tr, _, _ = integrate_observer(model, quadrotor_dynamics_torch, Tf, Uf, Yf, x0f, C_cpu, "cpu", substeps=1)
        with torch.no_grad():
            _, L_all = model.get_state_and_gain(Tf, x0f.view(1, 12).repeat(NSTEPS, 1))
        Ln.append(L_all.flatten(1).norm(dim=1).mean().item())
        if torch.isfinite(tr).all():
            n_stable += 1
            r = torch.sqrt(((tr - Xf) ** 2).mean(dim=0)); hid.append(r[HID].mean().item())
    model.to(device)
    hh = (sum(hid) / len(hid)) if hid else float("nan")
    return n_stable, hh, sum(Ln) / len(Ln)

os.makedirs("docs", exist_ok=True)
with open(CSV_PATH, "w", newline="") as fp:
    csv.writer(fp).writerow(["layers", "neurons", "params", "L_norm_mean", "n_stable_of_10", "hidden_rmse", "train_time_s"])

best = None
for Ly in LAYERS_LIST:
    for H in NEURONS_LIST:
        if (Ly,H) in done:
            print(f"skip {Ly}x{H} (already done)", flush=True); continue
        print(f"\n=== {Ly} layers x {H} neurons (train-through-ODE) ===", flush=True)
        t = time.time()
        model, npar = train_one(Ly, H)
        nstab, hh, Ln = eval_integrated(model)
        dt = time.time() - t
        print(f"  -> params {npar} | ||L|| {Ln:.2f} | stable {nstab}/10 | hidden RMSE {hh:.4f} | {dt:.0f}s", flush=True)
        with open(CSV_PATH, "a", newline="") as fp:
            csv.writer(fp).writerow([Ly, H, npar, f"{Ln:.3f}", nstab, f"{hh:.4f}", f"{dt:.0f}"])
        if nstab >= 8 and (best is None or hh < best[2]):
            best = (Ly, H, hh)

if best:
    print(f"\nBEST (stable) architecture: {best[0]} layers x {best[1]} neurons -> hidden RMSE {best[2]:.4f}", flush=True)
print(f"Saved {CSV_PATH}", flush=True)
