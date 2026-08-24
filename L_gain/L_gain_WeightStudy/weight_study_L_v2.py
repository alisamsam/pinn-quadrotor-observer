import sys, os, csv, time
# run from repo root (like grid_study_L_v2.py)
ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'phase1a'))
import torch
import torch.nn as nn
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

# ------------------------------------------------------------------------------
# Loss-weight sweep for the L-ON (Farkane) observer, fixed at the best architecture
# from the L-gain study: 4 layers x 20 neurons. Mirrors Weight_loss_Simulation/
# weight_study_v2.py (L-OFF) -- SAME 7 cases, epochs/lr/batch/seed, spiral_v2 dataset,
# and direct-output evaluation -- so the two CSVs are directly comparable. In addition
# to accuracy we log the mean learned-gain norm ||L|| on the test set. Crash-safe.
# ------------------------------------------------------------------------------
EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 500
LAYERS, HIDDEN = 4, 20
SEED = 0
DATASET = 'datasets/spiral_v2_dataset.npz'
CSV_PATH = 'docs/weight_study_L_v2.csv'
CASES = [(1.0,1.0,1.0),(0.5,1.5,1.0),(0.5,0.5,1.0),(1.0,2.0,1.0),(2.0,1.0,1.0),(2.0,1.0,0.5),(2.0,1.5,1.5)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, '| dataset:', DATASET, '| L-ON weight sweep @ 4x20', flush=True)
train, test = load_phase4_data(DATASET)
T=train['T'].to(device); X0=train['X0'].to(device); Y=train['Y'].to(device); U=train['U'].to(device); X=train['X'].to(device)
N=T.shape[0]; t0=(T[:,0]==T[:,0].min()); T0,X0_0,Xtrue_0=T[t0],X0[t0],X[t0]
Tt,X0t,Xt=test['T'].to(device),test['X0'].to(device),test['X'].to(device)
HID=[i for i in range(12) if i not in MEAS_IDX]

C = torch.zeros(len(MEAS_IDX), 12, device=device)
for _r,_c in enumerate(MEAS_IDX): C[_r,_c]=1.0

def residual_farkane(model, t, x0, u, y_meas):
    # g = x_hat_dot - f(x_hat,u) - L . C (x - x_hat),  with C(x - x_hat) = y - C x_hat
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

def train_one(w0, wode, wy):
    torch.manual_seed(SEED)
    model=PINNObserverV5(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
    npar=sum(p.numel() for p in model.parameters())
    opt=torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        perm=torch.randperm(N, device=device)
        for s in range(0, N, BATCH):
            idx=perm[s:s+BATCH]; tb,x0b,yb,ub=T[idx],X0[idx],Y[idx],U[idx]
            opt.zero_grad()
            x_hat,res=residual_farkane(model, tb, x0b, ub, yb)
            ly=nn.functional.mse_loss(x_hat[:,MEAS_IDX], yb)
            lg=nn.functional.mse_loss(res, torch.zeros_like(res))
            x0p,_=model.get_state_and_gain(T0, X0_0)
            l0=nn.functional.mse_loss(x0p, Xtrue_0)
            (wy*ly + wode*lg + w0*l0).backward(); opt.step()
        if ep==1 or ep%LOG_EVERY==0:
            print(f'    ep {ep:4d} | MSE_0 {l0.item():.3e} | MSE_g {lg.item():.3e} | MSE_y {ly.item():.3e}', flush=True)
    model.eval()
    with torch.no_grad():
        Xp, Lall = model.get_state_and_gain(Tt, X0t)
        rmse=torch.sqrt(((Xp - Xt)**2).mean(dim=0))
        Lnorm=Lall.flatten(1).norm(dim=1).mean().item()
    return npar, rmse[MEAS_IDX].mean().item(), rmse[HID].mean().item(), Lnorm

HEADER=['case','w0','w_ode','wy','params','rmse_meas','rmse_hidden','L_norm_mean','train_time_s']
os.makedirs('docs', exist_ok=True)
done=set()
if os.path.exists(CSV_PATH):
    with open(CSV_PATH) as fp:
        rd=csv.reader(fp); next(rd, None)
        for r in rd:
            if r and r[0].isdigit(): done.add(int(r[0]))
    print('resuming; already done cases:', sorted(done), flush=True)
else:
    with open(CSV_PATH,'w',newline='') as fp: csv.writer(fp).writerow(HEADER)

rows=[]
for ci,(w0,wode,wy) in enumerate(CASES, 1):
    if ci in done:
        print(f'skip case {ci} (already done)', flush=True); continue
    print(f'\n=== Case {ci}: w=({w0},{wode},{wy}) ===', flush=True); t=time.time()
    p,mr,hr,Ln=train_one(w0,wode,wy); dt=time.time()-t
    print(f'  -> meas {mr:.4f} | hidden {hr:.4f} | ||L|| {Ln:.2f} | {dt:.0f}s', flush=True)
    row=[ci,w0,wode,wy,p,f'{mr:.4f}',f'{hr:.4f}',f'{Ln:.3f}',f'{dt:.0f}']; rows.append(row)
    with open(CSV_PATH,'a',newline='') as fp: csv.writer(fp).writerow(row)

# best over the FULL csv (so it is correct even on a resumed run)
allrows=[r for r in csv.reader(open(CSV_PATH))][1:]
allrows=[r for r in allrows if r and r[0].isdigit()]
best=min(allrows, key=lambda r: float(r[6]))
print(f'\nBEST by hidden RMSE: case {best[0]} w=({best[1]},{best[2]},{best[3]}) -> hidden {best[6]} (meas {best[5]}, ||L|| {best[7]})', flush=True)
print('Saved', CSV_PATH, flush=True)
