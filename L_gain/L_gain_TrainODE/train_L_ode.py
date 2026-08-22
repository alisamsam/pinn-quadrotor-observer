import sys, os, time
ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "phase1a"))
import torch
import torch.nn as nn
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch
from rollout import build_C, rollout_observer
from observer_v2_L_gain import integrate_observer     # for the final stability check

# ------------------------------------------------------------------------------
# Train the gain L "through the integration": start from a WRONG estimate and roll
# the observer ODE forward with the network's L over a short window; the loss is the
# error of that integrated trajectory vs the truth. This forces L to be a STABILISING
# gain (the fix for the diverging L found in the arch study).
# ------------------------------------------------------------------------------
LAYERS, HIDDEN = 4, 100
ITERS, LR = 4000, 1e-3
BATCH_FLIGHTS = 8
NOISE0 = 0.1                 # size of the deliberately-wrong initial estimate
CLIP = 1.0
SEED = 0
NSTEPS = 3000
DATASET = "datasets/spiral_v2_dataset.npz"
SAVE_PATH = "phase1a/pinn_4x100_v2_L_ode.pth"
LOG_EVERY = 100

def window_len(it):          # curriculum: grow the rollout window as training stabilises
    if it < 1000: return 50
    if it < 2500: return 100
    return 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| train L through ODE integration", flush=True)
torch.manual_seed(SEED)

train, test = load_phase4_data(DATASET)
TR = {k: v.to(device) for k, v in train.items()}
n_flights = TR["T"].shape[0] // NSTEPS
base_T = TR["T"][0:NSTEPS, 0]                       # shared time grid within a flight
C = build_C(MEAS_IDX, device)
HID = [i for i in range(12) if i not in MEAS_IDX]

model = PINNObserverV5(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

t0 = time.time()
for it in range(1, ITERS + 1):
    W = window_len(it)
    flights = torch.randperm(n_flights, device=device)[:BATCH_FLIGHTS]
    s = int(torch.randint(0, NSTEPS - W, (1,)).item())
    rows = (flights.view(-1, 1) * NSTEPS + s + torch.arange(W, device=device).view(1, -1))  # (B,W)
    u_win = TR["U"][rows].permute(1, 0, 2)          # (W,B,4)
    y_win = TR["Y"][rows].permute(1, 0, 2)          # (W,B,6)
    x_true = TR["X"][rows].permute(1, 0, 2)         # (W,B,12)
    x0_cond = TR["X0"][flights * NSTEPS]            # (B,12) each flight's x0
    t_win = base_T[s:s + W]                         # (W,)
    x_hat0 = x_true[0] + NOISE0 * torch.randn_like(x_true[0])   # wrong start

    opt.zero_grad()
    traj = rollout_observer(model, quadrotor_dynamics_torch, C, t_win, u_win, y_win, x0_cond, x_hat0)
    loss = nn.functional.mse_loss(traj, x_true)
    if not torch.isfinite(loss):
        print(f"  it {it:4d} | W {W} | non-finite loss, skipped", flush=True)
        continue
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    opt.step()
    if it == 1 or it % LOG_EVERY == 0:
        print(f"  it {it:4d} | W {W} | loss {loss.item():.4e}", flush=True)

torch.save(model.state_dict(), SAVE_PATH)
print(f"\nSaved {SAVE_PATH} | {time.time()-t0:.0f}s", flush=True)

# --- payoff check: does the trained L now give a STABLE integrated observer? ---
model.eval()
Cc = build_C(MEAS_IDX, "cpu"); mc = model.to("cpu")
n_stable = 0; hid = []
for j in range(10):
    a = j * NSTEPS; b = a + NSTEPS
    Tf, Uf, Yf, Xf = test["T"][a:b], test["U"][a:b], test["Y"][a:b], test["X"][a:b]
    x0f = test["X0"][a]
    tr, _, div = integrate_observer(mc, quadrotor_dynamics_torch, Tf, Uf, Yf, x0f, Cc, "cpu", substeps=1)
    if torch.isfinite(tr).all():
        n_stable += 1
        r = torch.sqrt(((tr - Xf) ** 2).mean(dim=0))
        hid.append(r[HID].mean().item())
hh = (sum(hid) / len(hid)) if hid else float("nan")
print(f"AFTER train-through-ODE: stable {n_stable}/10 | hidden RMSE (stable) {hh:.4f}", flush=True)
