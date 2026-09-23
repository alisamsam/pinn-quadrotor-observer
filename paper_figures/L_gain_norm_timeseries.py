# L_gain_norm_timeseries.py
# Train the learned-gain observer (4x100, Farkane residual, neutral weights 1,1,1
# -- IDENTICAL recipe to the L-ON grid study) and save the Frobenius norm
# ||L(t)|| along one unseen spiral test flight. This is the SAME quantity that is
# averaged over time to give the ||L|| column in the L-gain table, now resolved
# in time. Its mean should match the 4x100 row (approx 26.9).
#
# Place this at the REPO ROOT (~/pinn-quadrotor/) and run from there:
#   python3 L_gain_norm_timeseries.py
# (on the cluster: module load pytorch/... ; nohup python3 L_gain_norm_timeseries.py >> L_norm_log.txt 2>&1 &)

import sys, os, time
ROOT = os.getcwd()
for p in [ROOT,
          os.path.join(ROOT, 'phase1a'),
          os.path.join(ROOT, 'L_gain', 'L_gain_ArchStudy')]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np, torch, torch.nn as nn
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch
from observer_v2_L_gain import build_C

LAYERS, HIDDEN = 4, 100
EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 500
W0, WODE, WY = 1.0, 1.0, 1.0                 # neutral weights, matches the L-ON grid
SEED = 0
DATASET = 'datasets/spiral_v2_dataset.npz'
NSTEPS = 3000
FLIGHT = 0
OUT = 'paper_figures/fig_L_norm_timeseries.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, '| L(t) Frobenius norm, 4x100, weights 1,1,1', flush=True)

train, test = load_phase4_data(DATASET)
T = train['T'].to(device); X0 = train['X0'].to(device); Y = train['Y'].to(device)
U = train['U'].to(device); X = train['X'].to(device)
N = T.shape[0]
t0 = (T[:, 0] == T[:, 0].min()); T0, X0_0, Xtrue_0 = T[t0], X0[t0], X[t0]
C_dev = build_C(MEAS_IDX, device)

def residual_farkane(model, t, x0, u, y_meas):
    t = t.clone().requires_grad_(True)
    x_hat, L = model.get_state_and_gain(t, x0)
    x_hat_dot = torch.zeros_like(x_hat)
    for i in range(12):
        gi = torch.autograd.grad(x_hat[:, i].sum(), t, create_graph=True)[0]
        x_hat_dot[:, i] = gi[:, 0]
    f = quadrotor_dynamics_torch(x_hat, u)
    C_x_hat = x_hat @ C_dev.t()
    innovation = (y_meas - C_x_hat).unsqueeze(-1)
    correction = torch.bmm(L, innovation).squeeze(-1)
    return x_hat, x_hat_dot - f - correction

def train_one(layers, hidden):
    torch.manual_seed(SEED)
    model = PINNObserverV5(hidden=hidden, n_hidden_layers=layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, BATCH):
            idx = perm[s:s+BATCH]; tb, x0b, yb, ub = T[idx], X0[idx], Y[idx], U[idx]
            opt.zero_grad()
            x_hat, res = residual_farkane(model, tb, x0b, ub, yb)
            ly = nn.functional.mse_loss(x_hat[:, MEAS_IDX], yb)
            lg = nn.functional.mse_loss(res, torch.zeros_like(res))
            x0p, _ = model.get_state_and_gain(T0, X0_0)
            l0 = nn.functional.mse_loss(x0p, Xtrue_0)
            (WY*ly + WODE*lg + W0*l0).backward(); opt.step()
        if ep == 1 or ep % LOG_EVERY == 0:
            print(f'  ep {ep:4d} | MSE_0 {l0.item():.3e} | MSE_g {lg.item():.3e} | MSE_y {ly.item():.3e}', flush=True)
    return model

t0t = time.time(); model = train_one(LAYERS, HIDDEN)
print('trained in %.0fs' % (time.time() - t0t), flush=True)

# ---- L(t) along one unseen test flight (same convention as the table) ----
model = model.to('cpu'); model.eval()
a = FLIGHT * NSTEPS; b = a + NSTEPS
Tf = test['T'][a:b]; x0f = test['X0'][a]
with torch.no_grad():
    _, L_all = model.get_state_and_gain(Tf, x0f.view(1, 12).repeat(NSTEPS, 1))
L_fro = L_all.flatten(1).norm(dim=1).numpy()     # (3000,) Frobenius norm per timestep
t = Tf.view(-1).numpy()

os.makedirs('paper_figures', exist_ok=True)
np.savetxt(OUT, np.column_stack([t, L_fro]), delimiter=',',
           header='t,L_fro', comments='', fmt='%.6g')
print('SAVED', OUT, '| shape', L_fro.shape,
      '| mean ||L|| = %.2f  (compare with the 4x100 row of the L-gain table)' % L_fro.mean(),
      flush=True)