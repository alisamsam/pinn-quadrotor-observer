import sys, os, csv, time
ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'phase1a'))
import torch
import torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

# ------------------------------------------------------------------------------
# Final head-to-head at the best config: 4 layers x 20 neurons, weights (0.5,1.5,1.0).
# Train the WITHOUT-L observer (V4) and the WITH-L observer (V5, Farkane residual),
# identical everything else, then report per-state RMSE for all 12 states + averages.
# ------------------------------------------------------------------------------
EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 500
LAYERS, HIDDEN = 4, 20
W0, WODE, WY = 0.5, 1.5, 1.0
SEED = 0
DATASET = 'datasets/spiral_v2_dataset.npz'
CSV_PATH = 'docs/compare_L_vs_noL_4x20.csv'
STATES = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, '| best-config head-to-head @ 4x20 w=(0.5,1.5,1.0)', flush=True)
train, test = load_phase4_data(DATASET)
T=train['T'].to(device); X0=train['X0'].to(device); Y=train['Y'].to(device); U=train['U'].to(device); X=train['X'].to(device)
N=T.shape[0]; t0=(T[:,0]==T[:,0].min()); T0,X0_0,Xtrue_0=T[t0],X0[t0],X[t0]
Tt,X0t,Xt=test['T'].to(device),test['X0'].to(device),test['X'].to(device)
HID=[i for i in range(12) if i not in MEAS_IDX]

C = torch.zeros(len(MEAS_IDX), 12, device=device)
for _r,_c in enumerate(MEAS_IDX): C[_r,_c]=1.0

def residual_plain(model, t, x0, u):
    t=t.clone().requires_grad_(True); xh=model(t,x0); dx=torch.zeros_like(xh)
    for i in range(12):
        g=torch.autograd.grad(xh[:,i].sum(), t, create_graph=True)[0]; dx[:,i]=g[:,0]
    return dx - quadrotor_dynamics_torch(xh, u)

def residual_farkane(model, t, x0, u, y_meas):
    t=t.clone().requires_grad_(True)
    x_hat, L = model.get_state_and_gain(t, x0)
    x_hat_dot=torch.zeros_like(x_hat)
    for i in range(12):
        gi=torch.autograd.grad(x_hat[:,i].sum(), t, create_graph=True)[0]; x_hat_dot[:,i]=gi[:,0]
    f=quadrotor_dynamics_torch(x_hat, u)
    C_x_hat=x_hat @ C.t()
    innovation=(y_meas - C_x_hat).unsqueeze(-1)
    correction=torch.bmm(L, innovation).squeeze(-1)
    return x_hat, x_hat_dot - f - correction

def train_noL():
    torch.manual_seed(SEED)
    m=PINNObserverV4(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
    opt=torch.optim.Adam(m.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        perm=torch.randperm(N, device=device)
        for s in range(0, N, BATCH):
            idx=perm[s:s+BATCH]; tb,x0b,yb,ub=T[idx],X0[idx],Y[idx],U[idx]
            opt.zero_grad(); xh=m(tb,x0b)
            ly=nn.functional.mse_loss(xh[:,MEAS_IDX], yb)
            res=residual_plain(m,tb,x0b,ub); lg=nn.functional.mse_loss(res, torch.zeros_like(res))
            l0=nn.functional.mse_loss(m(T0,X0_0), Xtrue_0)
            (WY*ly+WODE*lg+W0*l0).backward(); opt.step()
        if ep==1 or ep%LOG_EVERY==0: print(f'  [noL] ep {ep:4d} | g {lg.item():.3e} | y {ly.item():.3e}', flush=True)
    m.eval()
    with torch.no_grad(): rmse=torch.sqrt(((m(Tt,X0t)-Xt)**2).mean(dim=0))
    return rmse.cpu().numpy()

def train_L():
    torch.manual_seed(SEED)
    m=PINNObserverV5(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
    opt=torch.optim.Adam(m.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        perm=torch.randperm(N, device=device)
        for s in range(0, N, BATCH):
            idx=perm[s:s+BATCH]; tb,x0b,yb,ub=T[idx],X0[idx],Y[idx],U[idx]
            opt.zero_grad()
            xh,res=residual_farkane(m,tb,x0b,ub,yb)
            ly=nn.functional.mse_loss(xh[:,MEAS_IDX], yb); lg=nn.functional.mse_loss(res, torch.zeros_like(res))
            x0p,_=m.get_state_and_gain(T0,X0_0); l0=nn.functional.mse_loss(x0p, Xtrue_0)
            (WY*ly+WODE*lg+W0*l0).backward(); opt.step()
        if ep==1 or ep%LOG_EVERY==0: print(f'  [L]   ep {ep:4d} | g {lg.item():.3e} | y {ly.item():.3e}', flush=True)
    m.eval()
    with torch.no_grad():
        Xp,_=m.get_state_and_gain(Tt,X0t); rmse=torch.sqrt(((Xp-Xt)**2).mean(dim=0))
    return rmse.cpu().numpy()

print('\n== training WITHOUT-L (V4) ==', flush=True); t=time.time(); r_noL=train_noL(); print(f'  done {time.time()-t:.0f}s', flush=True)
print('\n== training WITH-L (V5) ==', flush=True); t=time.time(); r_L=train_L(); print(f'  done {time.time()-t:.0f}s', flush=True)

os.makedirs('docs', exist_ok=True)
with open(CSV_PATH,'w',newline='') as fp:
    wtr=csv.writer(fp); wtr.writerow(['state','type','rmse_without_L','rmse_with_L'])
    for i,s in enumerate(STATES):
        typ='measured' if i in MEAS_IDX else 'hidden'
        wtr.writerow([s, typ, f'{r_noL[i]:.4f}', f'{r_L[i]:.4f}'])
    import numpy as np
    wtr.writerow(['AVG_all','-', f'{r_noL.mean():.4f}', f'{r_L.mean():.4f}'])
    wtr.writerow(['AVG_measured','-', f'{r_noL[MEAS_IDX].mean():.4f}', f'{r_L[MEAS_IDX].mean():.4f}'])
    wtr.writerow(['AVG_hidden','-', f'{r_noL[HID].mean():.4f}', f'{r_L[HID].mean():.4f}'])
print('\nSaved', CSV_PATH, flush=True)
print('AVG all  : noL %.4f | L %.4f' % (r_noL.mean(), r_L.mean()), flush=True)
print('AVG hidden: noL %.4f | L %.4f' % (r_noL[HID].mean(), r_L[HID].mean()), flush=True)
