# -*- coding: utf-8 -*-
# Train the CIRCLE observer at its best config (4 layers x 100 neurons, weights 0.5/0.5/1.0)
# and SAVE true / estimated / noisy for all 12 states on one unseen test flight to an .npz.
# Run from repo root:  module load pytorch/2.0.0/gpu
#   nohup python eval_circle_best.py >> eval_circle_best_log.txt 2>&1 &
import sys, os, time
ROOT = os.getcwd()
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'phase1a'))
import numpy as np, torch, torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 500
LAYERS, HIDDEN = 4, 100
W0, WODE, WY = 0.5, 0.5, 1.0          # best circle weight case (Case 3)
SEED = 0
DATASET = 'datasets/circle_v3_dataset.npz'
NSTEPS = 3000; FLIGHT = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, '| circle best observer 4x100 w=(0.5,0.5,1.0)', flush=True)
train, test = load_phase4_data(DATASET)
T=train['T'].to(device); X0=train['X0'].to(device); Y=train['Y'].to(device); U=train['U'].to(device); X=train['X'].to(device)
N=T.shape[0]; t0=(T[:,0]==T[:,0].min()); T0,X0_0,Xtrue_0=T[t0],X0[t0],X[t0]

def residual(model,t,x0,u):
    t=t.clone().requires_grad_(True); xh=model(t,x0); dx=torch.zeros_like(xh)
    for i in range(12):
        gi=torch.autograd.grad(xh[:,i].sum(),t,create_graph=True)[0]; dx[:,i]=gi[:,0]
    return dx-quadrotor_dynamics_torch(xh,u)

torch.manual_seed(SEED)
model=PINNObserverV4(hidden=HIDDEN,n_hidden_layers=LAYERS).to(device)
opt=torch.optim.Adam(model.parameters(),lr=LR)
for ep in range(1,EPOCHS+1):
    perm=torch.randperm(N,device=device)
    for s in range(0,N,BATCH):
        idx=perm[s:s+BATCH]; tb,x0b,yb,ub=T[idx],X0[idx],Y[idx],U[idx]
        opt.zero_grad(); xh=model(tb,x0b)
        ly=nn.functional.mse_loss(xh[:,MEAS_IDX],yb); res=residual(model,tb,x0b,ub)
        lg=nn.functional.mse_loss(res,torch.zeros_like(res)); x0p=model(T0,X0_0)
        l0=nn.functional.mse_loss(x0p,Xtrue_0); (WY*ly+WODE*lg+W0*l0).backward(); opt.step()
    if ep==1 or ep%LOG_EVERY==0:
        print(f'  ep {ep:4d} | MSE_0 {l0.item():.3e} | MSE_g {lg.item():.3e} | MSE_y {ly.item():.3e}', flush=True)

model.eval()
a=FLIGHT*NSTEPS; b=a+NSTEPS
Tt=test['T'][a:b].to(device); X0t=test['X0'][a:b].to(device)
with torch.no_grad():
    est=model(Tt,X0t).cpu().numpy()
t=test['T'][a:b,0].cpu().numpy(); true=test['X'][a:b].cpu().numpy(); ynoisy=test['Y'][a:b].cpu().numpy()
rmse=np.sqrt(((est-true)**2).mean(axis=0))
HID=[i for i in range(12) if i not in MEAS_IDX]
print('per-state RMSE:', np.round(rmse,4).tolist(), flush=True)
print('avg measured %.4f | avg hidden %.4f' % (rmse[MEAS_IDX].mean(), np.array(rmse)[HID].mean()), flush=True)
os.makedirs('docs', exist_ok=True)
out='docs/eval_circle_best_data.npz'
np.savez(out, t=t, true=true, est=est, ynoisy=ynoisy, meas_idx=np.array(MEAS_IDX))
print('SAVED', out, flush=True)
