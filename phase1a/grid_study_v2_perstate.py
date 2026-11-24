import sys, os, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))
import numpy as np
import torch
import torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

# Architecture grid, PER-STATE + NRMSE version (reviewer comment [2]).
# layers {4,9,12} x neurons {20,60,100,128}, neutral weights (1,1,1), spiral_v2.
# Crash-safe: appends each row as it finishes.
EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 500
W0, WODE, WY = 1.0, 1.0, 1.0
SEED = 0
LAYERS_LIST = [4, 9, 12]; NEURONS_LIST = [20, 60, 100, 128]
DATASET = "datasets/spiral_v2_dataset.npz"
NAMES = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
CSV_PATH = "docs/grid_study_v2_perstate.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| dataset:", DATASET, flush=True)
train, test = load_phase4_data(DATASET)
T=train["T"].to(device); X0=train["X0"].to(device); Y=train["Y"].to(device); U=train["U"].to(device); X=train["X"].to(device)
N=T.shape[0]; t0=(T[:,0]==T[:,0].min()); T0,X0_0,Xtrue_0=T[t0],X0[t0],X[t0]
Tt,X0t,Xt=test["T"].to(device),test["X0"].to(device),test["X"].to(device)
HID=[i for i in range(12) if i not in MEAS_IDX]
Xt_np = Xt.cpu().numpy(); RANGE = Xt_np.max(axis=0) - Xt_np.min(axis=0)   # (12,)

def residual(model,t,x0,u):
    t=t.clone().requires_grad_(True); xh=model(t,x0); dx=torch.zeros_like(xh)
    for i in range(12):
        g=torch.autograd.grad(xh[:,i].sum(),t,create_graph=True)[0]; dx[:,i]=g[:,0]
    return dx-quadrotor_dynamics_torch(xh,u)

def train_one(layers,hidden):
    torch.manual_seed(SEED); model=PINNObserverV4(hidden=hidden,n_hidden_layers=layers).to(device)
    npar=sum(p.numel() for p in model.parameters()); opt=torch.optim.Adam(model.parameters(),lr=LR)
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
        rmse=torch.sqrt(((model(Tt,X0t)-Xt)**2).mean(dim=0)).cpu().numpy()   # (12,)
    return npar, rmse

os.makedirs("docs",exist_ok=True)
with open(CSV_PATH,"w",newline="") as fp:
    csv.writer(fp).writerow(["layers","neurons","params"] +
        [f"RMSE_{n}" for n in NAMES] +
        ["meas_RMSE","hidden_RMSE","meas_NRMSE_pct","hidden_NRMSE_pct","overall_NRMSE_pct"])
for L in LAYERS_LIST:
    for H in NEURONS_LIST:
        print(f"\n=== {L} layers x {H} neurons ===",flush=True); t=time.time()
        p,rmse=train_one(L,H); dt=time.time()-t
        nr=rmse/RANGE; mr,hr=rmse[MEAS_IDX].mean(),rmse[HID].mean()
        mnr,hnr,anr=100*nr[MEAS_IDX].mean(),100*nr[HID].mean(),100*nr.mean()
        print(f"  -> params {p} | meas {mr:.4f} hid {hr:.4f} | NRMSE meas {mnr:.2f}% hid {hnr:.2f}% all {anr:.2f}% | {dt:.0f}s",flush=True)
        row=[L,H,p]+[f"{v:.4f}" for v in rmse]+[f"{mr:.4f}",f"{hr:.4f}",f"{mnr:.2f}",f"{hnr:.2f}",f"{anr:.2f}"]
        with open(CSV_PATH,"a",newline="") as fp: csv.writer(fp).writerow(row)
print(f"\nSaved {CSV_PATH}",flush=True)
