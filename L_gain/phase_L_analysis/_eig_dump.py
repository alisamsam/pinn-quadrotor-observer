import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "phase1a"))
import numpy as np, torch
from pinn_observer_v5 import PINNObserverV5
from dynamics_torch import quadrotor_dynamics_torch

MEAS_IDX=[0,1,2,6,7,8]; FLIGHT=45; STEP=25
d=np.load("datasets/spiral_dataset.npz"); T,X,U=d["T"],d["X"],d["U"]
x_true=X[FLIGHT]; u_true=U[FLIGHT]; x0=x_true[0]
m=PINNObserverV5(); m.load_state_dict(torch.load("phase1a/pinn_phase5.pth",map_location="cpu")); m.eval()
Tt=torch.tensor(T,dtype=torch.float32).unsqueeze(1); X0t=torch.tensor(np.tile(x0,(len(T),1)),dtype=torch.float32)
with torch.no_grad(): _,L_all=m.get_state_and_gain(Tt,X0t)
C=np.zeros((6,12)); C[np.arange(6),MEAS_IDX]=1.0
rows=[]
for k in range(0,len(T),STEP):
    xk=torch.tensor(x_true[k],dtype=torch.float32); uk=torch.tensor(u_true[k],dtype=torch.float32)
    f=lambda x: quadrotor_dynamics_torch(x.unsqueeze(0),uk.unsqueeze(0)).squeeze(0)
    A=torch.autograd.functional.jacobian(f,xk).numpy(); L=L_all[k].numpy()
    ev=np.linalg.eigvals(A-L@C)
    rows.append([float(T[k]), float(ev.real.max()), float(ev.real.min()),
                 float(np.abs(ev.imag).max()), "UNSTABLE" if ev.real.max()>0 else "stable"])
import csv
os.makedirs("docs",exist_ok=True)
with open("docs/eig_analysis_phase5.csv","w",newline="") as fp:
    w=csv.writer(fp); w.writerow(["time_s","max_real_part","min_real_part","max_abs_imag","verdict"]); w.writerows(rows)
print("rows:",len(rows)); print("wrote docs/eig_analysis_phase5.csv")