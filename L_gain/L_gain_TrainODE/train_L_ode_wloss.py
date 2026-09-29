# train_L_ode_wloss.py  --  E1: per-state weighted (normalized) rollout loss
# Identical to train_L_ode.py EXCEPT the training loss is normalized per state,
# so hidden velocities/rates are not drowned out by the large-scale positions.
# Everything else (rollout, curriculum, NOISE0, clip, LR, iters, eval) is unchanged,
# so any accuracy change is attributable to the loss weighting alone.
#
# PLACE IN  L_gain_TrainODE/  (next to rollout.py and observer_v2_L_gain.py) and run
# FROM THE REPO ROOT:
#   module load pytorch/2.0.0/gpu
#   nohup python L_gain_TrainODE/train_L_ode_wloss.py >> L_ode_wloss_log.txt 2>&1 &

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

ARCHS = [(4, 20), (4, 100)]          # baseline best (0.178) and the paper's 4x100 (0.253)
ITERS, LR = 4000, 1e-3
BATCH_FLIGHTS = 8
NOISE0 = 0.1
CLIP = 1.0
SEED = 0
NSTEPS = 3000
DATASET = "datasets/spiral_v2_dataset.npz"
CSV_PATH = "docs/train_L_ode_wloss.csv"
LOG_EVERY = 250

def window_len(it):
    if it < 1000: return 50
    if it < 2500: return 100
    return 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| E1: per-state weighted rollout loss", flush=True)

train, test = load_phase4_data(DATASET)
TR = {k: v.to(device) for k, v in train.items()}
n_flights = TR["T"].shape[0] // NSTEPS
base_T = TR["T"][0:NSTEPS, 0]
C = build_C(MEAS_IDX, device)
HID = [i for i in range(12) if i not in MEAS_IDX]

# --- per-state weights = 1 / (std of each state over the training data) ---
scale = TR["X"].std(dim=0).clamp_min(1e-6)      # (12,)
w = (1.0 / scale).to(device)                    # (12,) loss weight per state
print("per-state std:", [round(float(s), 3) for s in scale.tolist()], flush=True)

os.makedirs("docs", exist_ok=True)
with open(CSV_PATH, "w", newline="") as fp:
    csv.writer(fp).writerow(["layers", "neurons", "params", "L_norm_mean",
                             "n_stable_of_10", "hidden_rmse", "meas_rmse", "train_time_s"])

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
    # ||L(t)|| mean on flight 0
    a = 0; b = NSTEPS
    with torch.no_grad():
        _, L0 = model.get_state_and_gain(test["T"][a:b], test["X0"][a].view(1, 12).repeat(NSTEPS, 1))
    Lnorm = L0.flatten(1).norm(dim=1).mean().item()
    hh = (sum(hid) / len(hid)) if hid else float("nan")
    mm = (sum(meas) / len(meas)) if meas else float("nan")
    return Lnorm, n_stable, hh, mm

for (LAYERS, HIDDEN) in ARCHS:
    print(f"\n=== {LAYERS}x{HIDDEN} (weighted-loss rollout) ===", flush=True)
    torch.manual_seed(SEED)
    model = PINNObserverV5(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
    npar = sum(p.numel() for p in model.parameters())
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
        loss = (((traj - x_true) * w) ** 2).mean()      # <-- E1: per-state normalized loss
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if it == 1 or it % LOG_EVERY == 0:
            print(f"  it {it:4d} | W {W} | wloss {loss.item():.4e}", flush=True)
    dt = time.time() - t0
    save_path = f"phase1a/pinn_{LAYERS}x{HIDDEN}_v2_L_ode_wloss.pth"
    torch.save(model.state_dict(), save_path)
    Lnorm, nstab, hh, mm = evaluate(model)
    print(f"  -> saved {save_path} | ||L|| {Lnorm:.2f} | stable {nstab}/10 | "
          f"hidden RMSE {hh:.4f} | meas RMSE {mm:.4f} | {dt:.0f}s", flush=True)
    with open(CSV_PATH, "a", newline="") as fp:
        csv.writer(fp).writerow([LAYERS, HIDDEN, npar, f"{Lnorm:.3f}", nstab,
                                 f"{hh:.4f}", f"{mm:.4f}", f"{dt:.0f}"])
    model.to(device)

print(f"\nSaved {CSV_PATH}", flush=True)
print("Compare hidden RMSE to the unweighted baseline: 4x20=0.178, 4x100=0.253", flush=True)
