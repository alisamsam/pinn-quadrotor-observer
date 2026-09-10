# -*- coding: utf-8 -*-
# FIGURE-8 design selection, made CONSISTENT with the 10-flight metric.
#   1) architecture sweep (neutral weights) over LAYERS x WIDTHS
#   2) pick the best architecture (lowest hidden RMSE)
#   3) loss-weight study (7 cases) at that best architecture
# Every RMSE is POOLED over the 10 unseen test flights (same metric as the
# per-state results), so the sweep numbers match the final accuracy.
# Saves docs/grid_study_figure8_10flight.csv and
#       docs/weight_study_figure8_10flight.csv
#
# Run from repo root (cluster):
#   module load pytorch/2.0.0/gpu
#   nohup python3 scenarios/figure8/design_study_figure8_10f.py >> figure8_design_10f_log.txt 2>&1 &
#
# To shorten: edit LAYERS below (e.g. drop 12) or WIDTHS.
import sys, os, time
ROOT = os.getcwd()
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'phase1a'))
import numpy as np, torch, torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

EPOCHS, LR, BATCH, LOG_EVERY, SEED = 2000, 1e-3, 4096, 100, 0
DATASET = 'datasets/figure8_v3_dataset.npz'
LAYERS  = [4, 9, 12]                 # <-- trim here to shorten (e.g. [4, 9])
WIDTHS  = [20, 60, 100, 128]
WEIGHT_CASES = [                      # (case, w0, w_ode, wy)
    (1, 1.0, 1.0, 1.0), (2, 0.5, 1.5, 1.0), (3, 0.5, 0.5, 1.0),
    (4, 1.0, 2.0, 1.0), (5, 2.0, 1.0, 1.0), (6, 2.0, 1.0, 0.5),
    (7, 2.0, 1.5, 1.5),
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HID = [i for i in range(12) if i not in MEAS_IDX]
print('Device:', device, '| figure-8 design study (10-flight)', flush=True)

train, test = load_phase4_data(DATASET)
T = train['T'].to(device); X0 = train['X0'].to(device); Y = train['Y'].to(device)
U = train['U'].to(device); X = train['X'].to(device)
N = T.shape[0]; t0 = (T[:, 0] == T[:, 0].min()); T0, X0_0, Xtrue_0 = T[t0], X0[t0], X[t0]
Tt = test['T'].to(device); X0t = test['X0'].to(device); Xt_true = test['X'].numpy()


def residual(m, t, x0, u):
    t = t.clone().requires_grad_(True); xh = m(t, x0); dx = torch.zeros_like(xh)
    for i in range(12):
        g = torch.autograd.grad(xh[:, i].sum(), t, create_graph=True)[0]; dx[:, i] = g[:, 0]
    return dx - quadrotor_dynamics_torch(xh, u)


def train_eval(layers, hidden, w0, wode, wy, label=''):
    torch.manual_seed(SEED)
    m = PINNObserverV4(hidden=hidden, n_hidden_layers=layers).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]; tb, x0b, yb, ub = T[idx], X0[idx], Y[idx], U[idx]
            opt.zero_grad(); xh = m(tb, x0b)
            ly = nn.functional.mse_loss(xh[:, MEAS_IDX], yb); res = residual(m, tb, x0b, ub)
            lg = nn.functional.mse_loss(res, torch.zeros_like(res))
            l0 = nn.functional.mse_loss(m(T0, X0_0), Xtrue_0)
            (wy * ly + wode * lg + w0 * l0).backward(); opt.step()
        if ep == 1 or ep % LOG_EVERY == 0:
            print('    [%s] ep %4d | MSE_0 %.2e | MSE_g %.2e | MSE_y %.2e'
                  % (label, ep, l0.item(), lg.item(), ly.item()), flush=True)
    m.eval()
    with torch.no_grad():
        est = m(Tt, X0t).cpu().numpy()
    rmse = np.sqrt(((est - Xt_true) ** 2).mean(axis=0))
    params = sum(p.numel() for p in m.parameters())
    return rmse[MEAS_IDX].mean(), np.array(rmse)[HID].mean(), params


os.makedirs('docs', exist_ok=True)

# 1) architecture sweep (neutral weights)
print('\n== architecture sweep (neutral weights, 10-flight) ==', flush=True)
rows = []
for L in LAYERS:
    for W in WIDTHS:
        t = time.time(); mm, hh, pp = train_eval(L, W, 1.0, 1.0, 1.0, label='%dx%d' % (L, W)); dt = time.time() - t
        rows.append((L, W, pp, mm, hh, dt))
        print('  %2d x %3d | params %7d | meas %.4f | hidden %.4f | %.0fs'
              % (L, W, pp, mm, hh, dt), flush=True)
with open('docs/grid_study_figure8_10flight.csv', 'w') as f:
    f.write('layers,neurons,params,rmse_meas,rmse_hidden,train_time_s\n')
    for r in rows:
        f.write('%d,%d,%d,%.4f,%.4f,%.0f\n' % r)
best = min(rows, key=lambda r: r[4]); bestL, bestW = best[0], best[1]
print('  BEST architecture: %d x %d (hidden %.4f)' % (bestL, bestW, best[4]), flush=True)

# 2) loss-weight study at the best architecture
print('\n== loss-weight study at %d x %d (10-flight) ==' % (bestL, bestW), flush=True)
wrows = []
for (c, w0, wode, wy) in WEIGHT_CASES:
    t = time.time(); mm, hh, pp = train_eval(bestL, bestW, w0, wode, wy, label='C%d' % c); dt = time.time() - t
    wrows.append((c, w0, wode, wy, pp, mm, hh, dt))
    print('  C%d (%.1f,%.1f,%.1f) | meas %.4f | hidden %.4f | %.0fs'
          % (c, w0, wode, wy, mm, hh, dt), flush=True)
with open('docs/weight_study_figure8_10flight.csv', 'w') as f:
    f.write('case,w0,w_ode,wy,params,rmse_meas,rmse_hidden,train_time_s\n')
    for r in wrows:
        f.write('%d,%.1f,%.1f,%.1f,%d,%.4f,%.4f,%.0f\n' % r)
bestw = min(wrows, key=lambda r: r[6])
print('  BEST weights: C%d (%.1f,%.1f,%.1f) hidden %.4f'
      % (bestw[0], bestw[1], bestw[2], bestw[3], bestw[6]), flush=True)
print('\nDONE. Saved docs/grid_study_figure8_10flight.csv and docs/weight_study_figure8_10flight.csv', flush=True)
