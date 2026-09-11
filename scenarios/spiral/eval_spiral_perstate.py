# -*- coding: utf-8 -*-
# SPIRAL: per-state RMSE POOLED over all 10 unseen test flights, best config.
# (Consistency check: should reproduce the report's ~0.044 / 0.056.)
# Run from repo root (cluster):
#   module load pytorch/2.0.0/gpu
#   nohup python3 scenarios/spiral/eval_spiral_perstate.py >> spiral_perstate_log.txt 2>&1 &
import sys, os
ROOT = os.getcwd()
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'phase1a'))
import numpy as np, torch, torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

EPOCHS, LR, BATCH, LOG_EVERY, SEED = 2000, 1e-3, 4096, 100, 0
LAYERS, HIDDEN = 4, 100
W0, WODE, WY = 0.5, 1.5, 1.0                 # best spiral weights (Case 2)
DATASET = 'datasets/spiral_v2_dataset.npz'
TAG = 'spiral'
NAMES = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r']
UNITS = ['m', 'm', 'm', 'm/s', 'm/s', 'm/s', 'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HID = [i for i in range(12) if i not in MEAS_IDX]
print('Device:', device, '| %s per-state 10-flight | %dx%d w=(%.1f,%.1f,%.1f)'
      % (TAG, LAYERS, HIDDEN, W0, WODE, WY), flush=True)

train, test = load_phase4_data(DATASET)
T = train['T'].to(device); X0 = train['X0'].to(device); Y = train['Y'].to(device)
U = train['U'].to(device); X = train['X'].to(device)
N = T.shape[0]; t0 = (T[:, 0] == T[:, 0].min()); T0, X0_0, Xtrue_0 = T[t0], X0[t0], X[t0]


def residual(model, t, x0, u):
    t = t.clone().requires_grad_(True); xh = model(t, x0); dx = torch.zeros_like(xh)
    for i in range(12):
        gi = torch.autograd.grad(xh[:, i].sum(), t, create_graph=True)[0]; dx[:, i] = gi[:, 0]
    return dx - quadrotor_dynamics_torch(xh, u)


torch.manual_seed(SEED)
model = PINNObserverV4(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
for ep in range(1, EPOCHS + 1):
    perm = torch.randperm(N, device=device)
    for s in range(0, N, BATCH):
        idx = perm[s:s + BATCH]; tb, x0b, yb, ub = T[idx], X0[idx], Y[idx], U[idx]
        opt.zero_grad(); xh = model(tb, x0b)
        ly = nn.functional.mse_loss(xh[:, MEAS_IDX], yb); res = residual(model, tb, x0b, ub)
        lg = nn.functional.mse_loss(res, torch.zeros_like(res)); x0p = model(T0, X0_0)
        l0 = nn.functional.mse_loss(x0p, Xtrue_0)
        (WY * ly + WODE * lg + W0 * l0).backward(); opt.step()
    if ep == 1 or ep % LOG_EVERY == 0:
        print('  ep %4d | MSE_0 %.3e | MSE_g %.3e | MSE_y %.3e'
              % (ep, l0.item(), lg.item(), ly.item()), flush=True)

# per-state RMSE pooled over ALL test flights (10 x 3000)
model.eval()
Tt = test['T'].to(device); X0t = test['X0'].to(device); true = test['X'].numpy()
with torch.no_grad():
    est = model(Tt, X0t).cpu().numpy()
rmse = np.sqrt(((est - true) ** 2).mean(axis=0))
nfl = test['T'].shape[0] // 3000
print('pooled over %d unseen test flights' % nfl, flush=True)

os.makedirs('docs', exist_ok=True)
out = 'docs/%s_perstate_10flights.csv' % TAG
with open(out, 'w') as f:
    f.write('state,unit,type,rmse\n')
    for i in range(12):
        f.write('%s,%s,%s,%.4f\n' % (NAMES[i], UNITS[i], 'measured' if i in MEAS_IDX else 'hidden', rmse[i]))
    f.write('avg_measured,,,%.4f\n' % rmse[MEAS_IDX].mean())
    f.write('avg_hidden,,,%.4f\n' % np.array(rmse)[HID].mean())
for i in range(12):
    print('  %-6s %-8s %.4f' % (NAMES[i], 'measured' if i in MEAS_IDX else 'hidden', rmse[i]), flush=True)
print('avg_measured %.4f | avg_hidden %.4f' % (rmse[MEAS_IDX].mean(), np.array(rmse)[HID].mean()), flush=True)
print('SAVED', out, flush=True)
