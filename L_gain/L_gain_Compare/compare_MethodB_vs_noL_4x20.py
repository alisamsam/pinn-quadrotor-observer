import sys, os, csv, time
ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'phase1a'))
sys.path.insert(0, os.path.join(ROOT, 'L_gain_TrainODE'))   # cluster: flat; has rollout.py, observer_v2_L_gain.py
import torch
import torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch
from rollout import build_C, rollout_observer
from observer_v2_L_gain import integrate_observer

# ------------------------------------------------------------------------------
# Head-to-head at 4 layers x 20 neurons on spiral_v2:
#   (a) no-L observer  (V4), trained the plain way, read directly from the network;
#   (b) Method-B L observer (V5), trained THROUGH the ODE integration, evaluated by
#       integrating the observer forward with the learned gain (the way it is used).
# Reports per-state RMSE for all 12 states + averages.
# ------------------------------------------------------------------------------
LAYERS, HIDDEN, SEED = 4, 20, 0
DATASET = 'datasets/spiral_v2_dataset.npz'
CSV_PATH = 'docs/compare_MethodB_vs_noL_4x20.csv'
STATES = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
# no-L training
EPOCHS, LR, BATCH = 2000, 1e-3, 4096
W0, WODE, WY = 0.5, 1.5, 1.0
# Method-B training
ITERS, LR_B, BATCH_FLIGHTS, NOISE0, CLIP, NSTEPS = 3000, 1e-3, 8, 0.1, 1.0, 3000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, '| Method-B (integration) vs no-L @ 4x20', flush=True)
train, test = load_phase4_data(DATASET)
T=train['T'].to(device); X0=train['X0'].to(device); Y=train['Y'].to(device); U=train['U'].to(device); X=train['X'].to(device)
N=T.shape[0]; t0=(T[:,0]==T[:,0].min()); T0,X0_0,Xtrue_0=T[t0],X0[t0],X[t0]
Tt,X0t,Xt=test['T'].to(device),test['X0'].to(device),test['X'].to(device)
HID=[i for i in range(12) if i not in MEAS_IDX]
Cd = torch.zeros(len(MEAS_IDX),12,device=device)
for _r,_c in enumerate(MEAS_IDX): Cd[_r,_c]=1.0

# ---------------- (a) no-L observer ----------------
def residual_plain(model, t, x0, u):
    t=t.clone().requires_grad_(True); xh=model(t,x0); dx=torch.zeros_like(xh)
    for i in range(12):
        g=torch.autograd.grad(xh[:,i].sum(), t, create_graph=True)[0]; dx[:,i]=g[:,0]
    return dx - quadrotor_dynamics_torch(xh, u)

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
        if ep==1 or ep%500==0: print(f'  [noL] ep {ep:4d} | g {lg.item():.3e} | y {ly.item():.3e}', flush=True)
    m.eval()
    with torch.no_grad(): rmse=torch.sqrt(((m(Tt,X0t)-Xt)**2).mean(dim=0))
    return rmse.cpu().numpy()

# ---------------- (b) Method-B L observer ----------------
def window_len(it):
    if it < 800: return 50
    if it < 2000: return 100
    return 200

def train_L_methodB():
    torch.manual_seed(SEED)
    TR={k:v.to(device) for k,v in train.items()}
    n_flights=TR['T'].shape[0]//NSTEPS
    base_T=TR['T'][0:NSTEPS,0]
    C_dev=build_C(MEAS_IDX, device)
    m=PINNObserverV5(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
    opt=torch.optim.Adam(m.parameters(), lr=LR_B)
    for it in range(1, ITERS+1):
        Wl=window_len(it)
        flights=torch.randperm(n_flights, device=device)[:BATCH_FLIGHTS]
        s=int(torch.randint(0, NSTEPS-Wl, (1,)).item())
        rows=(flights.view(-1,1)*NSTEPS + s + torch.arange(Wl, device=device).view(1,-1))
        u_win=TR['U'][rows].permute(1,0,2); y_win=TR['Y'][rows].permute(1,0,2)
        x_true=TR['X'][rows].permute(1,0,2); x0_cond=TR['X0'][flights*NSTEPS]
        t_win=base_T[s:s+Wl]; x_hat0=x_true[0]+NOISE0*torch.randn_like(x_true[0])
        opt.zero_grad()
        traj=rollout_observer(m, quadrotor_dynamics_torch, C_dev, t_win, u_win, y_win, x0_cond, x_hat0)
        loss=nn.functional.mse_loss(traj, x_true)
        if not torch.isfinite(loss): continue
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP); opt.step()
        if it==1 or it%500==0: print(f'  [L-B] it {it:4d} | W {Wl} | loss {loss.item():.4e}', flush=True)
    # evaluate by integration, per-state RMSE averaged over stable flights
    m=m.to('cpu'); m.eval()
    C_cpu=build_C(MEAS_IDX, 'cpu')
    import numpy as np
    persum=np.zeros(12); nst=0
    for j in range(10):
        a=j*NSTEPS; b=a+NSTEPS
        Tf,Uf,Yf,Xf=test['T'][a:b],test['U'][a:b],test['Y'][a:b],test['X'][a:b]
        x0f=test['X0'][a]
        tr,_,_=integrate_observer(m, quadrotor_dynamics_torch, Tf, Uf, Yf, x0f, C_cpu, 'cpu', substeps=1)
        if torch.isfinite(tr).all():
            r=torch.sqrt(((tr-Xf)**2).mean(dim=0)).numpy(); persum+=r; nst+=1
    per=persum/max(nst,1)
    return per, nst

print('\n== training no-L (V4) ==', flush=True); t=time.time(); r_noL=train_noL(); print(f'  done {time.time()-t:.0f}s', flush=True)
print('\n== training Method-B L (V5, through integration) ==', flush=True); t=time.time(); r_L, nstable=train_L_methodB(); print(f'  done {time.time()-t:.0f}s | stable {nstable}/10', flush=True)

import numpy as np
os.makedirs('docs', exist_ok=True)
with open(CSV_PATH,'w',newline='') as fp:
    wtr=csv.writer(fp); wtr.writerow(['state','type','rmse_no_L','rmse_L_methodB'])
    for i,s in enumerate(STATES):
        typ='measured' if i in MEAS_IDX else 'hidden'
        wtr.writerow([s, typ, f'{r_noL[i]:.4f}', f'{r_L[i]:.4f}'])
    wtr.writerow(['AVG_all','-', f'{r_noL.mean():.4f}', f'{r_L.mean():.4f}'])
    wtr.writerow(['AVG_measured','-', f'{r_noL[MEAS_IDX].mean():.4f}', f'{r_L[MEAS_IDX].mean():.4f}'])
    wtr.writerow(['AVG_hidden','-', f'{r_noL[HID].mean():.4f}', f'{r_L[HID].mean():.4f}'])
print('\nSaved', CSV_PATH, flush=True)
print('AVG hidden: no-L %.4f | Method-B %.4f | stable %d/10' % (r_noL[HID].mean(), r_L[HID].mean(), nstable), flush=True)
print('AVG all   : no-L %.4f | Method-B %.4f' % (r_noL.mean(), r_L.mean()), flush=True)
