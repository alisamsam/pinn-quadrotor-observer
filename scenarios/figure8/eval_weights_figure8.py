# -*- coding: utf-8 -*-
# Figure-8 loss-weight study (7 cases) at a FIXED architecture, 10-flight pooled RMSE.
# Pass the best architecture from the grid sweep as args (default 9 100):
#   python3 scenarios/figure8/eval_weights_figure8.py 9 100
# Saves docs/weight_study_figure8_10flight.csv
# Run from repo root:
#   module load pytorch/2.0.0/gpu
#   nohup python3 scenarios/figure8/eval_weights_figure8.py 9 100 >> figure8_weights_10f_log.txt 2>&1 &
import sys, os, time
ROOT = os.getcwd()
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'phase1a'))
import numpy as np, torch, torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 9
HIDDEN = int(sys.argv[2]) if len(sys.argv) > 2 else 100
EPOCHS, LR, BATCH, LOG_EVERY, SEED = 2000, 1e-3, 4096, 100, 0
DATASET = 'datasets/figure8_v3_dataset.npz'
CASES = [(1, 1.0, 1.0, 1.0), (2, 0.5, 1.5, 1.0), (3, 0.5, 0.5, 1.0),
         (4, 1.0, 2.0, 1.0), (5, 2.0, 1.0, 1.0), (6, 2.0, 1.0, 0.5),
         (7, 2.0, 1.5, 1.5)]
HID = [i for i in range(12) if i not in MEAS_IDX]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, '| figure-8 weight study at %dx%d (10-flight)' % (LAYERS, HIDDEN), flush=True)

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


def train_eval(w0, wode, wy, label):
    torch.manual_seed(SEED)
    m = PINNObserverV4(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]; tb, x0b, yb, ub = T[idx], X0[idx], Y[idx], U[idx]
            opt.zero_grad(); xh = m(tb, x0b)
            ly = nn.functional.mse_loss(xh[:, MEAS_IDX], yb); res = residual(m, tb, x0b, ub)
            lg = nn.functional.mse_loss(res, torch.zeros_like(res)); l0 = nn.functional.mse_loss(m(T0, X0_0), Xtrue_0)
            (wy * ly + wode * lg + w0 * l0).backward(); opt.step()
        if ep == 1 or ep % LOG_EVERY == 0:
            print('    [%s] ep %4d | MSE_0 %.2e | MSE_g %.2e | MSE_y %.2e'
                  % (label, ep, l0.item(), lg.item(), ly.item()), flush=True)
    m.eval()
    with torch.no_grad():
        est = m(Tt, X0t).cpu().numpy()
    rmse = np.sqrt(((est - true) ** 2).mean(axis=0))
    return rmse[MEAS_IDX].mean(), np.array(rmse)[HID].mean(), rmse


NAMES = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r']
UNITS = ['m', 'm', 'm', 'm/s', 'm/s', 'm/s', 'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s']
os.makedirs('docs', exist_ok=True)
rows = []; per = {}
for (c, w0, wode, wy) in CASES:
    t = time.time(); mm, hh, rmse = train_eval(w0, wode, wy, 'C%d' % c); dt = time.time() - t
    rows.append((c, w0, wode, wy, mm, hh, dt)); per[c] = rmse
    print('  C%d (%.1f,%.1f,%.1f) | meas %.4f | hidden %.4f | %.0fs' % (c, w0, wode, wy, mm, hh, dt), flush=True)
with open('docs/weight_study_figure8_10flight.csv', 'w') as f:
    f.write('case,w0,w_ode,wy,rmse_meas,rmse_hidden,train_time_s\n')
    for r in rows:
        f.write('%d,%.1f,%.1f,%.1f,%.4f,%.4f,%.0f\n' % r)
best = min(rows, key=lambda r: r[5])
print('BEST weights: C%d (%.1f,%.1f,%.1f) hidden %.4f' % (best[0], best[1], best[2], best[3], best[5]), flush=True)

# per-state accuracy at the best (arch, weights) -> figure8_perstate_10flights.csv
ps = per[best[0]]
with open('docs/figure8_perstate_10flights.csv', 'w') as f:
    f.write('state,unit,type,rmse\n')
    for i in range(12):
        f.write('%s,%s,%s,%.4f\n' % (NAMES[i], UNITS[i], 'measured' if i in MEAS_IDX else 'hidden', ps[i]))
    f.write('avg_measured,,,%.4f\n' % ps[MEAS_IDX].mean())
    f.write('avg_hidden,,,%.4f\n' % np.array(ps)[HID].mean())
print('per-state (best case C%d) saved -> docs/figure8_perstate_10flights.csv (meas %.4f | hidden %.4f)'
      % (best[0], ps[MEAS_IDX].mean(), np.array(ps)[HID].mean()), flush=True)
print('SAVED docs/weight_study_figure8_10flight.csv', flush=True)
