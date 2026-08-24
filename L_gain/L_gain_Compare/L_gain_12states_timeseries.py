# -*- coding: utf-8 -*-
# Train the learned-gain observer (4x20, integration) and SAVE true vs estimated
# for all 12 states on one unseen test flight to an .npz (no matplotlib on the cluster).
# Run from the repo root:
#   module load pytorch/2.0.0/gpu
#   nohup python L_gain_12states_timeseries.py >> L_gain_12states_log.txt 2>&1 &
import sys, os, time
ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'phase1a'))
sys.path.insert(0, os.path.join(ROOT, 'L_gain_TrainODE'))
import numpy as np, torch, torch.nn as nn
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch
from rollout import build_C, rollout_observer
from observer_v2_L_gain import integrate_observer

LAYERS, HIDDEN, SEED = 4, 20, 0
DATASET = 'datasets/spiral_v2_dataset.npz'
ITERS, LR_B, BATCH_FLIGHTS, NOISE0, CLIP, NSTEPS = 3000, 1e-3, 8, 0.1, 1.0, 3000
FLIGHT = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, '| L-gain 12-state data @ 4x20', flush=True)
train, test = load_phase4_data(DATASET)

def window_len(it):
    if it < 800: return 50
    if it < 2000: return 100
    return 200

def train_methodB():
    torch.manual_seed(SEED)
    TR = {k: v.to(device) for k, v in train.items()}
    n_fl = TR['T'].shape[0] // NSTEPS
    base_T = TR['T'][0:NSTEPS, 0]; C_dev = build_C(MEAS_IDX, device)
    m = PINNObserverV5(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR_B)
    for it in range(1, ITERS + 1):
        Wl = window_len(it)
        fl = torch.randperm(n_fl, device=device)[:BATCH_FLIGHTS]
        s = int(torch.randint(0, NSTEPS - Wl, (1,)).item())
        rows = (fl.view(-1,1) * NSTEPS + s + torch.arange(Wl, device=device).view(1,-1))
        u = TR['U'][rows].permute(1,0,2); y = TR['Y'][rows].permute(1,0,2)
        xt = TR['X'][rows].permute(1,0,2); x0c = TR['X0'][fl * NSTEPS]
        tw = base_T[s:s+Wl]; xh0 = xt[0] + NOISE0 * torch.randn_like(xt[0])
        opt.zero_grad()
        traj = rollout_observer(m, quadrotor_dynamics_torch, C_dev, tw, u, y, x0c, xh0)
        loss = nn.functional.mse_loss(traj, xt)
        if not torch.isfinite(loss): continue
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP); opt.step()
        if it == 1 or it % 500 == 0:
            print(f'  it {it:4d} | W {Wl} | loss {loss.item():.4e}', flush=True)
    return m

t0 = time.time(); model = train_methodB(); print('trained in %.0fs' % (time.time()-t0), flush=True)

a = FLIGHT * NSTEPS; b = a + NSTEPS
Tf, Uf, Yf, Xf = test['T'][a:b], test['U'][a:b], test['Y'][a:b], test['X'][a:b]
x0f = test['X0'][a]
model = model.to('cpu'); model.eval()
C_cpu = build_C(MEAS_IDX, 'cpu')
traj, _, _ = integrate_observer(model, quadrotor_dynamics_torch, Tf, Uf, Yf, x0f, C_cpu, 'cpu', substeps=1)

os.makedirs('docs', exist_ok=True)
out = 'docs/L_gain_12states_data.npz'
np.savez(out, t=Tf.view(-1).numpy(), true=Xf.numpy(), est=traj.numpy(), meas_idx=np.array(MEAS_IDX))
print('SAVED', out, '| finite:', bool(np.isfinite(traj.numpy()).all()), flush=True)
