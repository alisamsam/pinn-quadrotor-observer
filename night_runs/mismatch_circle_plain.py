# -*- coding: utf-8 -*-
"""CIRCLE model-mismatch study (plain PINN observer, 4x100, w=0.5/0.5/1.0).
Protocol (same as spiral run_mismatch_v2): perturb TRUE plant params, keep the
controller nominal (unaware), evaluate the frozen circle observer vs true states.
Resume-safe: caches the trained model (CKPT) and appends CSV rows one condition
at a time, skipping conditions already present.
Run from repo root:  module load pytorch/2.0.0/gpu
  nohup python3 night_runs/mismatch_circle_plain.py >> night_runs/log_circle_mm.txt 2>&1 &
Quick check:  SMOKE=1 python3 night_runs/mismatch_circle_plain.py
"""
import sys, os, csv, time
ROOT = os.getcwd()
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "phase1a"))
import numpy as np, torch, torch.nn as nn
from scipy.integrate import solve_ivp
from phase_mismatch.generate_perturbed import quadrotor_dynamics, NOMINAL
from core.controller_v3 import full_control_v3
from scenarios.circle.circle_reference import circular_reference, yaw_reference
from pinn_observer_v4 import PINNObserverV4
from dynamics_torch import quadrotor_dynamics_torch
from data_phase4 import load_phase4_data, MEAS_IDX

SMOKE   = bool(os.environ.get("SMOKE"))
LAYERS, HIDDEN = 4, 100
W0, WODE, WY   = 0.5, 0.5, 1.0
EPOCHS, LR, BATCH = (2 if SMOKE else 2000), 1e-3, 4096
DATASET = "datasets/circle_v3_dataset.npz"
CKPT    = "phase1a/pinn_circle_4x100.pth"
CSVOUT  = "docs/circle_mismatch_summary.csv"
HID = [i for i in range(12) if i not in MEAS_IDX]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| circle mismatch | SMOKE=", SMOKE, flush=True)

# ---------- train or load the plain observer ----------
model = PINNObserverV4(hidden=HIDDEN, n_hidden_layers=LAYERS).to(device)
if os.path.exists(CKPT) and not SMOKE:
    model.load_state_dict(torch.load(CKPT, map_location=device)); print("loaded", CKPT, flush=True)
else:
    train, _ = load_phase4_data(DATASET)
    T=train["T"].to(device); X0=train["X0"].to(device); Y=train["Y"].to(device)
    U=train["U"].to(device); X=train["X"].to(device)
    N=T.shape[0]; t0=(T[:,0]==T[:,0].min()); T0,X0_0,Xtrue0=T[t0],X0[t0],X[t0]
    def residual(m,t,x0,u):
        t=t.clone().requires_grad_(True); xh=m(t,x0); dx=torch.zeros_like(xh)
        for i in range(12): dx[:,i]=torch.autograd.grad(xh[:,i].sum(),t,create_graph=True)[0][:,0]
        return dx-quadrotor_dynamics_torch(xh,u)
    torch.manual_seed(0); opt=torch.optim.Adam(model.parameters(),lr=LR)
    for ep in range(1,EPOCHS+1):
        perm=torch.randperm(N,device=device)
        for s in range(0,N,BATCH):
            idx=perm[s:s+BATCH]; tb,x0b,yb,ub=T[idx],X0[idx],Y[idx],U[idx]
            opt.zero_grad(); xh=model(tb,x0b)
            ly=nn.functional.mse_loss(xh[:,MEAS_IDX],yb); res=residual(model,tb,x0b,ub)
            lg=nn.functional.mse_loss(res,torch.zeros_like(res)); x0p=model(T0,X0_0)
            l0=nn.functional.mse_loss(x0p,Xtrue0); (WY*ly+WODE*lg+W0*l0).backward(); opt.step()
        if ep==1 or ep%500==0: print(f"  ep {ep:4d} | L0 {l0.item():.2e} Lg {lg.item():.2e} Ly {ly.item():.2e}", flush=True)
    if not SMOKE: torch.save(model.state_dict(), CKPT); print("saved", CKPT, flush=True)
model.eval()

# ---------- 10 unseen starts + nominal (unaware) controller ----------
base=np.load(DATASET); x0s=[base["X"][i,0,:] for i in range(40,50)]
if SMOKE: x0s=x0s[:1]
t_eval=np.arange(0.0,30.0,0.01)
ctrl=lambda t,x: full_control_v3(t,x,circular_reference,yaw_reference)

def eval_set(P):
    me,he=[],[]
    for x0 in x0s:
        sol=solve_ivp(lambda t,x: quadrotor_dynamics(t,x,ctrl(t,x),P),(0.0,30.0),x0,
                      t_eval=t_eval,max_step=0.02)
        xt=sol.y.T
        Tt=torch.tensor(sol.t,dtype=torch.float32).unsqueeze(1).to(device)
        X0t=torch.tensor(np.tile(x0,(len(sol.t),1)),dtype=torch.float32).to(device)
        with torch.no_grad(): xh=model(Tt,X0t).cpu().numpy()
        r=np.sqrt(((xh-xt)**2).mean(axis=0)); me.append(r[MEAS_IDX].mean()); he.append(r[HID].mean())
    return float(np.mean(me)), float(np.mean(he))

# ---------- resume-safe sweep ----------
devs=[-0.20,-0.10,0.0,0.10,0.20]
params={"mass":["m"],"inertia":["Ix","Iy","Iz"],"arm":["l"]}
conds=[(p,d) for p in params for d in devs]+[("combined",0.20)]
if SMOKE: conds=[("mass",0.0)]
os.makedirs("docs",exist_ok=True)
newfile=not os.path.exists(CSVOUT)
done=set()
if not newfile:
    for row in csv.reader(open(CSVOUT)):
        if row and row[0]!="parameter": done.add((row[0],row[1]))
f=open(CSVOUT,"a",newline=""); w=csv.writer(f)
if newfile: w.writerow(["parameter","deviation","meas_RMSE","hidden_RMSE"]); f.flush()
for pname,dev in conds:
    key=(pname,f"{dev:+.2f}")
    if key in done: print("skip",key,flush=True); continue
    P=dict(NOMINAL)
    if pname=="combined":
        for k in ["m","Ix","Iy","Iz","l"]: P[k]=NOMINAL[k]*1.20
    else:
        for k in params[pname]: P[k]=NOMINAL[k]*(1+dev)
    t0=time.time(); mr,hr=eval_set(P)
    w.writerow([pname,f"{dev:+.2f}",f"{mr:.4f}",f"{hr:.4f}"]); f.flush()
    print(f"{pname:>8} {dev:+.0%}: meas {mr:.4f} | hidden {hr:.4f}  ({time.time()-t0:.0f}s)",flush=True)
f.close()
print("DONE circle mismatch ->",CSVOUT,flush=True)
