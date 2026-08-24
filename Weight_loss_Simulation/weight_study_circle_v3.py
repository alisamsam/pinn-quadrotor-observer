import sys, os, csv, time
ROOT = os.getcwd()                      # run from the repo root
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "phase1a"))
import torch
import torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

# Loss-weight sweep (7 cases) for the CIRCLE observer at its best architecture: 4 x 100.
EPOCHS, LR, BATCH, LOG_EVERY = 2000, 1e-3, 4096, 500
LAYERS, HIDDEN = 4, 100
SEED = 0
DATASET = "datasets/circle_v3_dataset.npz"
CSV_PATH = "docs/weight_study_circle_v3.csv"
CASES = [(1.0,1.0,1.0),(0.5,1.5,1.0),(0.5,0.5,1.0),(1.0,2.0,1.0),(2.0,1.0,1.0),(2.0,1.0,0.5),(2.0,1.5,1.5)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| dataset:", DATASET, "| weight sweep @ 4x100", flush=True)
train, test = load_phase4_data(DATASET)
T=train["T"].to(device); X0=train["X0"].to(device); Y=train["Y"].to(device); U=train["U"].to(device); X=train["X"].to(device)
N=T.shape[0]; t0=(T[:,0]==T[:,0].min()); T0,X0_0,Xtrue_0=T[t0],X0[t0],X[t0]
Tt,X0t,Xt=test["T"].to(device),test["X0"].to(device),test["X"].to(device)
HID=[i for i in range(12) if i not in MEAS_IDX]

def residual(model,t,x0,u):
    t=t.clone().requires_grad_(True); xh=model(t,x0); dx=torch.zeros_like(xh)
    for i in range(12):
        g=torch.autograd.grad(xh[:,i].sum(),t,create_graph=True)[0]; dx[:,i]=g[:,0]
    return dx-quadrotor_dynamics_torch(xh,u)

def train_one(w0,wode,wy):
    torch.manual_seed(SEED); model=PINNObserverV4(hidden=HIDDEN,n_hidden_layers=LAYERS).to(device)
    npar=sum(p.numel() for p in model.parameters()); opt=torch.optim.Adam(model.parameters(),lr=LR)
    for ep in range(1,EPOCHS+1):
        perm=torch.randperm(N,device=device)
        for s in range(0,N,BATCH):
            idx=perm[s:s+BATCH]; tb,x0b,yb,ub=T[idx],X0[idx],Y[idx],U[idx]
            opt.zero_grad(); xh=model(tb,x0b)
            ly=nn.functional.mse_loss(xh[:,MEAS_IDX],yb); res=residual(model,tb,x0b,ub)
            lg=nn.functional.mse_loss(res,torch.zeros_like(res)); x0p=model(T0,X0_0)
            l0=nn.functional.mse_loss(x0p,Xtrue_0); (wy*ly+wode*lg+w0*l0).backward(); opt.step()
        if ep==1 or ep%LOG_EVERY==0:
            print(f"    ep {ep:4d} | MSE_0 {l0.item():.3e} | MSE_g {lg.item():.3e} | MSE_y {ly.item():.3e}", flush=True)
    model.eval()
    with torch.no_grad():
        rmse=torch.sqrt(((model(Tt,X0t)-Xt)**2).mean(dim=0))
    return npar, rmse[MEAS_IDX].mean().item(), rmse[HID].mean().item()

HEADER=["case","w0","w_ode","wy","params","rmse_meas","rmse_hidden","train_time_s"]
os.makedirs("docs",exist_ok=True)
done=set()
if os.path.exists(CSV_PATH):
    with open(CSV_PATH) as fp:
        rd=csv.reader(fp); next(rd,None)
        for r in rd:
            if r and r[0].isdigit(): done.add(int(r[0]))
    print("resuming; done cases:", sorted(done), flush=True)
else:
    with open(CSV_PATH,"w",newline="") as fp: csv.writer(fp).writerow(HEADER)

for ci,(w0,wode,wy) in enumerate(CASES,1):
    if ci in done:
        print(f"skip case {ci} (done)", flush=True); continue
    print(f"\n=== Case {ci}: w=({w0},{wode},{wy}) ===",flush=True); t=time.time()
    p,mr,hr=train_one(w0,wode,wy); dt=time.time()-t
    print(f"  -> meas {mr:.4f} | hidden {hr:.4f} | {dt:.0f}s",flush=True)
    with open(CSV_PATH,"a",newline="") as fp:
        csv.writer(fp).writerow([ci,w0,wode,wy,p,f"{mr:.4f}",f"{hr:.4f}",f"{dt:.0f}"])

# BEST over the FULL csv (correct even on a resumed run)
rows=[r for r in csv.reader(open(CSV_PATH))][1:]
rows=[r for r in rows if r and r[0].isdigit()]
best=min(rows,key=lambda r:float(r[6]))
print(f"\nBEST by hidden RMSE: case {best[0]} w=({best[1]},{best[2]},{best[3]}) -> hidden {best[6]} (meas {best[5]})",flush=True)
print("Saved",CSV_PATH,flush=True)
