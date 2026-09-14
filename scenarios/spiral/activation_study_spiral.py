# -*- coding: utf-8 -*-
# SPIRAL activation-function study, Farkane-style table:
#   Activation | RMSE | MAE | I-Time (ms) | T-Time (s) | Conv. Iter. | Best Loss
# Neutral setup so no activation is favored: fixed 4x100, weights (1,1,1);
# only the activation sigma changes. 10-flight pooled metrics, saved per row.
# Run from repo root:
#   module load pytorch/2.0.0/gpu
#   nohup python3 scenarios/spiral/activation_study_spiral.py > spiral_activation_log.txt 2>&1 &
import sys, os, time
ROOT = os.getcwd()
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'phase1a'))
import numpy as np, torch, torch.nn as nn
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

EPOCHS, LR, BATCH, LOG_EVERY, SEED = 2000, 1e-3, 4096, 100, 0
LAYERS, HIDDEN = 4, 100
W0, WODE, WY = 1.0, 1.0, 1.0                 # neutral weights (fair to every activation)
DATASET = 'datasets/spiral_v2_dataset.npz'


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


ACTS = [
    ('relu',     nn.ReLU),
    ('sigmoid',  nn.Sigmoid),
    ('tanh',     nn.Tanh),
    ('sine',     Sine),
    ('silu',     nn.SiLU),
    ('gelu',     nn.GELU),
    ('softplus', nn.Softplus),
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HID = [i for i in range(12) if i not in MEAS_IDX]
print('Device:', device, '| spiral activation study | %dx%d w=(%.1f,%.1f,%.1f)'
      % (LAYERS, HIDDEN, W0, WODE, WY), flush=True)


class MLP(nn.Module):
    def __init__(self, act_cls, n_states=12, hidden=HIDDEN, n_layers=LAYERS):
        super().__init__()
        d_in, d_out = 1 + n_states, n_states
        layers = [nn.Linear(d_in, hidden), act_cls()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), act_cls()]
        layers += [nn.Linear(hidden, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, t, x0):
        return self.net(torch.cat([t, x0], dim=1))


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


def train_eval(act_cls, label):
    torch.manual_seed(SEED)
    m = MLP(act_cls).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    best_loss, conv_iter, step = float('inf'), 0, 0
    t_start = time.time()
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, BATCH):
            step += 1
            idx = perm[s:s + BATCH]; tb, x0b, yb, ub = T[idx], X0[idx], Y[idx], U[idx]
            opt.zero_grad(); xh = m(tb, x0b)
            ly = nn.functional.mse_loss(xh[:, MEAS_IDX], yb); res = residual(m, tb, x0b, ub)
            lg = nn.functional.mse_loss(res, torch.zeros_like(res))
            l0 = nn.functional.mse_loss(m(T0, X0_0), Xtrue_0)
            loss = WY * ly + WODE * lg + W0 * l0
            loss.backward(); opt.step()
            lv = loss.item()
            if lv < best_loss:
                best_loss, conv_iter = lv, step
        if ep == 1 or ep % LOG_EVERY == 0:
            print('    [%s] ep %4d | loss %.3e | best %.3e @%d' % (label, ep, lv, best_loss, conv_iter), flush=True)
    t_train = time.time() - t_start
    m.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(5):
            _ = m(Tt, X0t)                      # warmup
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0i = time.time(); REP = 50
        for _ in range(REP):
            est_t = m(Tt, X0t)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        i_time_ms = (time.time() - t0i) / REP * 1000.0
        est = est_t.cpu().numpy()
    err = est - Xt_true
    rmse = float(np.sqrt((err ** 2).mean())); mae = float(np.abs(err).mean())
    per = np.sqrt((err ** 2).mean(axis=0))
    rmse_meas = float(per[MEAS_IDX].mean()); rmse_hid = float(np.array(per)[HID].mean())
    params = sum(p.numel() for p in m.parameters())
    return rmse, mae, i_time_ms, t_train, conv_iter, best_loss, rmse_meas, rmse_hid, params


os.makedirs('docs', exist_ok=True)
out = 'docs/activation_study_spiral.csv'
with open(out, 'w') as f:
    f.write('activation,rmse,mae,i_time_ms,t_time_s,conv_iter,best_loss,rmse_meas,rmse_hidden,params\n')

for name, act in ACTS:
    try:
        rmse, mae, itms, tts, citer, bloss, rm, rh, pp = train_eval(act, name)
        with open(out, 'a') as f:
            f.write('%s,%.4f,%.5f,%.4f,%.2f,%d,%.3e,%.4f,%.4f,%d\n'
                    % (name, rmse, mae, itms, tts, citer, bloss, rm, rh, pp))
        print('  %-9s | RMSE %.4f | MAE %.5f | I %.3fms | T %.0fs | citer %d | best %.2e [saved]'
              % (name, rmse, mae, itms, tts, citer, bloss), flush=True)
    except Exception as e:
        with open(out, 'a') as f:
            f.write('%s,nan,nan,nan,nan,0,nan,nan,nan,0\n' % name)
        print('  %-9s | FAILED: %s' % (name, repr(e)), flush=True)

print('\nDONE -> docs/activation_study_spiral.csv', flush=True)
