import sys, os, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))
import numpy as np
import torch
import torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

# Stability characterisation of the 12x60 config: train it under several
# independent seeds and record how often it diverges vs converges.
EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 500
W0, WODE, WY = 1.0, 1.0, 1.0
LAYERS, HIDDEN = 12, 60
SEEDS = [0, 1, 2, 3, 4]
DATASET = "datasets/spiral_v2_dataset.npz"
NAMES = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
CSV_PATH = "docs/grid_12x60_seeds.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| dataset:", DATASET, flush=True)
train, test = load_phase4_data(DATASET)
T=train["T"].to(device); X0=train["X0"].to(device); Y=train["Y"].to(device); U=train["U"].to(device); X=train["X"].to(device)
N=T.shape[0]; t0=(T[:,0]==T[:,0].min()); T0,X0_0,Xtrue_0=T[t0],X0[t0],X[t0]
Tt,X0t,Xt=test["T"].to(device),test["X0"].to(device),test["X"].to(device)
HID=[i for i in range(12) if i not in MEAS_IDX]
Xt_np = Xt.cpu().numpy(); RANGE = Xt_np.max(axis=0) - Xt_np.min(axis=0)

def residual(model,t,x0,u):
    t=t.clone().requires_grad_(True); xh=model(t,x0); dx=torch.zeros_like(xh)
    for i in range(12):
        g=torch.autograd.grad(xh[:,i].sum(),t,create_graph=True)[0]; dx[:,i]=g[:,0]
    return dx-quadrotor_dynamics_torch(xh,u)

def train_one(seed):
    torch.manual_seed(seed); model=PINNObserverV4(hidden=HIDDEN,n_hidden_layers=LAYERS).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    for ep in range(1,EPOCHS+1):
        perm=torch.randperm(N,device=device)
        for s in range(0,N,BATCH):
            idx=perm[s:s+BATCH]; tb,x0b,yb,ub=T[idx],X0[idx],Y[idx],U[idx]
            opt.zero_grad(); xh=model(tb,x0b)
            ly=nn.functional.mse_loss(xh[:,MEAS_IDX],yb); res=residual(model,tb,x0b,ub)
            lg=nn.functional.mse_loss(res,torch.zeros_like(res))
            l0=nn.functional.mse_loss(model(T0,X0_0),Xtrue_0); (WY*ly+WODE*lg+W0*l0).backward(); opt.step()
        if ep==1 or ep%LOG_EVERY==0:
            print(f"    ep {ep:4d} | MSE_y {ly.item():.3e}", flush=True)
    model.eval()
    with torch.no_grad():
        rmse=torch.sqrt(((model(Tt,X0t)-Xt)**2).mean(dim=0)).cpu().numpy()
    return rmse

os.makedirs("docs",exist_ok=True)
with open(CSV_PATH,"w",newline="") as fp:
    csv.writer(fp).writerow(["seed"]+[f"RMSE_{n}" for n in NAMES]+
        ["meas_RMSE","hidden_RMSE","meas_NRMSE_pct","hidden_NRMSE_pct","overall_NRMSE_pct","diverged"])
for sd in SEEDS:
    print(f"\n=== seed {sd} (12x60) ===",flush=True); t=time.time()
    rmse=train_one(sd); dt=time.time()-t
    nr=rmse/RANGE; mr,hr=rmse[MEAS_IDX].mean(),rmse[HID].mean()
    mnr,hnr,anr=100*nr[MEAS_IDX].mean(),100*nr[HID].mean(),100*nr.mean()
    diverged = bool(mr>1.0 or hr>1.0 or not np.isfinite(mr) or not np.isfinite(hr))
    print(f"  -> seed {sd} | meas {mr:.4f} hid {hr:.4f} | NRMSE all {anr:.2f}% | diverged={diverged} | {dt:.0f}s",flush=True)
    row=[sd]+[f"{v:.4f}" for v in rmse]+[f"{mr:.4f}",f"{hr:.4f}",f"{mnr:.2f}",f"{hnr:.2f}",f"{anr:.2f}",str(diverged)]
    with open(CSV_PATH,"a",newline="") as fp: csv.writer(fp).writerow(row)
print(f"\nSaved {CSV_PATH}",flush=True)
