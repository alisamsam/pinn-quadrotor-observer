# train_L_ode_e3.py  --  E3: gentle gain via ||L|| penalty, on top of E1
# Keeps the E1 winner (4x100, per-state weighted rollout loss) UNCHANGED and adds
# one term: LAM * mean(||L||^2), which pushes the gain to stay small so the
# closed-loop estimate drifts less (closer to the accurate no-correction case).
# Sweeps a few LAM values. Everything else identical to E1 -> attributable to LAM.
#
# Lives in L_gain_TrainODE/. Run FROM THE REPO ROOT:
#   module load pytorch/2.0.0/gpu
#   nohup python L_gain_TrainODE/train_L_ode_e3.py >> L_ode_e3_log.txt 2>&1 &

import sys, os, csv, time
ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "phase1a"))
import torch
import torch.nn as nn
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch
from rollout import build_C, rollout_observer
from observer_v2_L_gain import integrate_observer

LAYERS, HIDDEN = 4, 100
ITERS, LR = 4000, 1e-3
BATCH_FLIGHTS = 8
NOISE0 = 0.1
CLIP = 1.0
SEED = 0
NSTEPS = 3000
DATASET = "datasets/spiral_v2_dataset.npz"
CSV_PATH = "docs/train_L_ode_e3.csv"
LOG_EVERY = 500
LAMS = [1e-5, 5e-5, 2e-4]        # gain-size penalty strengths to sweep

def window_len(it):              # same as E1
    if it < 1000: return 50
    if it < 2500: return 100
    return 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| E3: ||L|| penalty on top of E1 (weighted loss, 4x100)", flush=True)

train, test = load_phase4_data(DATASET)
TR = {k: v.to(device) for k, v in train.items()}
n_flights = TR["T"].shape[0] // NSTEPS
base_T = TR["T"][0:NSTEPS, 0]
C = build_C(MEAS_IDX, device)
HID = [i for i in range(12) if i not in MEAS_IDX]
scale = TR["X"].std(dim=0).clamp_min(1e-6)
w = (1.0 / scale).to(device)

os.makedirs("docs", exist_ok=True)
with open(CSV_PATH, "w", newline="") as fp:
    csv.writer(fp).writerow(["lambda", "L_norm_mean", "n_stable_of_10",
                             "hidden_rmse", "meas_rmse", "train_time_s"])

def evaluate(model):
    model = model.to("cpu"); model.eval()
    Cc = build_C(MEAS_IDX, "cpu")
    n_stable = 0; hid, meas = [], []
    for j in range(10):
        a = j * NSTEPS; b = a + NSTEPS
        Tf, Uf, Yf, Xf = test["T"][a:b], test["U"][a:b], test["Y"][a:b], test["X"][a:b]
        x0f = test["X0"][a]
        tr, _, _ = integrate_observer(model, quadrotor_dynamics_torch, Tf, Uf, Yf, x0f, Cc, "cpu", substeps=1)
        if torch.isfinite(tr).all():
            n_stable += 1
            r = torch.sqrt(((tr - Xf) ** 2).mean(dim=0))
            hid.append(r[HID].mean().item()); meas.append(r[MEAS_IDX].mean().item())
    with torch.no_grad():
        _, L0 = model.get_state_and_gain(test["T"][0:NSTEPS], test["X0"][0].view(1, 12).repeat(NSTEPS, 1))
    Lnorm = L0.flatten(1).norm(dim=1).mean().item()
    hh = (sum(hid) / len(hid)) if hid else float("nan")
    mm = (sum(meas) / len(meas)) if meas else float("nan")
    model.to(device)
    return Lnorm, n_stable, hh, mm

for LAM in LAMS:
    print(f"\n=== LAM = {LAM:.0e} ===", flush=True)
    torch.manual_seed(SEED)
    model = PINNObserverV5(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    for it in range(1, ITERS + 1):
        W = window_len(it)
        flights = torch.randperm(n_flights, device=device)[:BATCH_FLIGHTS]
        s = int(torch.randint(0, NSTEPS - W, (1,)).item())
        rows = (flights.view(-1, 1) * NSTEPS + s + torch.arange(W, device=device).view(1, -1))
        u_win = TR["U"][rows].permute(1, 0, 2)
        y_win = TR["Y"][rows].permute(1, 0, 2)
        x_true = TR["X"][rows].permute(1, 0, 2)
        x0_cond = TR["X0"][flights * NSTEPS]
        t_win = base_T[s:s + W]
        x_hat0 = x_true[0] + NOISE0 * torch.randn_like(x_true[0])
        opt.zero_grad()
        traj = rollout_observer(model, quadrotor_dynamics_torch, C, t_win, u_win, y_win, x0_cond, x_hat0)
        wloss = (((traj - x_true) * w) ** 2).mean()
        # ||L|| penalty over the window points (t, x0)
        tw_rep = t_win.view(W, 1, 1).expand(W, BATCH_FLIGHTS, 1).reshape(W * BATCH_FLIGHTS, 1)
        x0_rep = x0_cond.unsqueeze(0).expand(W, BATCH_FLIGHTS, 12).reshape(W * BATCH_FLIGHTS, 12)
        _, Lpen = model.get_state_and_gain(tw_rep, x0_rep)
        lreg = (Lpen ** 2).mean()
        loss = wloss + LAM * lreg
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if it == 1 or it % LOG_EVERY == 0:
            print(f"  it {it:4d} | W {W} | wloss {wloss.item():.4e} | "
                  f"lreg {lreg.item():.3e} | pen {LAM*lreg.item():.3e}", flush=True)
    dt = time.time() - t0
    save_path = f"phase1a/pinn_4x100_v2_L_ode_e3_lam{LAM:.0e}.pth"
    torch.save(model.state_dict(), save_path)
    Lnorm, nstab, hh, mm = evaluate(model)
    print(f"  -> LAM {LAM:.0e} | saved {save_path} | ||L|| {Lnorm:.2f} | stable {nstab}/10 | "
          f"hidden RMSE {hh:.4f} | meas RMSE {mm:.4f} | {dt:.0f}s", flush=True)
    with open(CSV_PATH, "a", newline="") as fp:
        csv.writer(fp).writerow([f"{LAM:.0e}", f"{Lnorm:.3f}", nstab, f"{hh:.4f}", f"{mm:.4f}", f"{dt:.0f}"])

print(f"\nSaved {CSV_PATH}", flush=True)
print("Compare: plain 0.056 | E1 4x100 weighted 0.132 | pick the LAM with lowest hidden RMSE that stays 10/10", flush=True)
