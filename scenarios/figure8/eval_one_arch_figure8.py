# -*- coding: utf-8 -*-
# Run ONE architecture config for figure-8 (neutral weights), 10-flight pooled RMSE.
# Default 12x128 (the last/missing config). Override:  python3 ... 12 128
# Appends a row to docs/grid_study_figure8_10flight_extra.csv
# Run from repo root:
#   module load pytorch/2.0.0/gpu
#   nohup python3 scenarios/figure8/eval_one_arch_figure8.py 12 128 >> figure8_one_arch_log.txt 2>&1 &
import sys, os, time
ROOT = os.getcwd()
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'phase1a'))
import numpy as np, torch, torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
HIDDEN = int(sys.argv[2]) if len(sys.argv) > 2 else 128
EPOCHS, LR, BATCH, LOG_EVERY, SEED = 2000, 1e-3, 4096, 100, 0
DATASET = 'datasets/figure8_v3_dataset.npz'
HID = [i for i in range(12) if i not in MEAS_IDX]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, '| figure-8 single config %dx%d (neutral weights)' % (LAYERS, HIDDEN), flush=True)

train, test = load_phase4_data(DATASET)
T = train['T'].to(device); X0 = train['X0'].to(device); Y = train['Y'].to(device)
U = train['U'].to(device); X = train['X'].to(device)
N = T.shape[0]; t0 = (T[:, 0] == T[:, 0].min()); T0, X0_0, Xtrue_0 = T[t0], X0[t0], X[t0]
Tt = test['T'].to(device); X0t = test['X0'].to(device); true = test['X'].numpy()


def residual(m, t, x0, u):
    t = t.clone().requires_grad_(True); xh = m(t, x0); dx = torch.zeros_like(xh)
    for i in range(12):
        g = torch.autograd.grad(xh[:, i].sum(), t, create_graph=True)[0]; dx[:, i] = g[:, 0]
    return dx - quadrotor_dynamics_torch(xh, u)


torch.manual_seed(SEED)
model = PINNObserverV4(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
t0w = time.time()
for ep in range(1, EPOCHS + 1):
    perm = torch.randperm(N, device=device)
    for s in range(0, N, BATCH):
        idx = perm[s:s + BATCH]; tb, x0b, yb, ub = T[idx], X0[idx], Y[idx], U[idx]
        opt.zero_grad(); xh = model(tb, x0b)
        ly = nn.functional.mse_loss(xh[:, MEAS_IDX], yb); res = residual(model, tb, x0b, ub)
        lg = nn.functional.mse_loss(res, torch.zeros_like(res)); l0 = nn.functional.mse_loss(model(T0, X0_0), Xtrue_0)
        (ly + lg + l0).backward(); opt.step()
    if ep == 1 or ep % LOG_EVERY == 0:
        print('    [%dx%d] ep %4d | MSE_0 %.2e | MSE_g %.2e | MSE_y %.2e'
              % (LAYERS, HIDDEN, ep, l0.item(), lg.item(), ly.item()), flush=True)

model.eval()
with torch.no_grad():
    est = model(Tt, X0t).cpu().numpy()
rmse = np.sqrt(((est - true) ** 2).mean(axis=0))
mm, hh = rmse[MEAS_IDX].mean(), np.array(rmse)[HID].mean()
pp = sum(p.numel() for p in model.parameters())
dt = time.time() - t0w
print('RESULT %d x %d | params %d | meas %.4f | hidden %.4f | %.0fs' % (LAYERS, HIDDEN, pp, mm, hh, dt), flush=True)

os.makedirs('docs', exist_ok=True)
out = 'docs/grid_study_figure8_10flight_extra.csv'
newfile = not os.path.exists(out)
with open(out, 'a') as f:
    if newfile:
        f.write('layers,neurons,params,rmse_meas,rmse_hidden,train_time_s\n')
    f.write('%d,%d,%d,%.4f,%.4f,%.0f\n' % (LAYERS, HIDDEN, pp, mm, hh, dt))
print('appended to', out, flush=True)
