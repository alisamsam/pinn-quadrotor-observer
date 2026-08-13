"""Luenberger under model mismatch (fair comparison vs PINN mismatch study).
Observer integrates the NOMINAL Singha model; the true drone (controller_v2) has
perturbed parameters. Clean measurements. Evaluated vs the true states, 10 unseen starts.
"""
import sys, os, csv
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, "phase1a"))
import numpy as np
from scipy.integrate import solve_ivp
from phase_mismatch.generate_perturbed import quadrotor_dynamics as true_dyn, NOMINAL
from core.generate_quadrotor_data import quadrotor_dynamics as nom_dyn
from core.controller_v2 import full_control_v2
from scenarios.spiral.spiral_reference import spiral_reference, yaw_reference

MEAS=[0,1,2,6,7,8]; PART=[3,4,5,9,10,11]; HID=[3,4,5,9,10,11]
W=10.0
L=np.zeros((12,6))
for j in range(6): L[MEAS[j],j]=2*W; L[PART[j],j]=W*W
ctrl = lambda t,x: full_control_v2(t,x,spiral_reference,yaw_reference)

def luen_rhs(xh,u,y): return nom_dyn(0.0,xh,u) + L@(y - xh[MEAS])
def luen_run(Xtrue,U,dt):
    N=Xtrue.shape[0]; xh=Xtrue[0].copy(); est=np.zeros_like(Xtrue); est[0]=xh
    for k in range(N-1):
        u,y=U[k],Xtrue[k,MEAS]
        k1=luen_rhs(xh,u,y);k2=luen_rhs(xh+0.5*dt*k1,u,y);k3=luen_rhs(xh+0.5*dt*k2,u,y);k4=luen_rhs(xh+dt*k3,u,y)
        xh=xh+dt/6*(k1+2*k2+2*k3+k4); est[k+1]=xh
    return est

Xb=np.load("datasets/spiral_v2_dataset.npz")["X"]; x0s=[Xb[i,0,:] for i in range(40,50)]
t_eval=np.arange(0.0,30.0,0.01); dt=0.01

def eval_set(P_true):
    me,he=[],[]
    for x0 in x0s:
        sol=solve_ivp(lambda t,x: true_dyn(t,x,ctrl(t,x),P_true),(0.0,30.0),x0,t_eval=t_eval,max_step=0.02)
        Xt=sol.y.T; U=np.array([ctrl(sol.t[k],Xt[k]) for k in range(len(sol.t))])
        est=luen_run(Xt,U,dt); rmse=np.sqrt(((est-Xt)**2).mean(axis=0))
        me.append(rmse[MEAS].mean()); he.append(rmse[HID].mean())
    return float(np.mean(me)), float(np.mean(he))

devs=[-0.20,-0.10,0.0,0.10,0.20]; params={"mass":["m"],"inertia":["Ix","Iy","Iz"],"arm":["l"]}
res={}
for p,keys in params.items():
    for dv in devs:
        P=dict(NOMINAL)
        for k in keys: P[k]=NOMINAL[k]*(1+dv)
        res[(p,dv)]=eval_set(P); print(f"{p:>8} {dv:+.0%}: meas {res[(p,dv)][0]:.4f} | hidden {res[(p,dv)][1]:.4f}", flush=True)
Pw=dict(NOMINAL); Pw["m"]*=0.8; Pw["Ix"]*=1.2; Pw["Iy"]*=1.2; Pw["Iz"]*=1.2; Pw["l"]*=0.8
cw=eval_set(Pw); print(f"combined worst: meas {cw[0]:.4f} | hidden {cw[1]:.4f}", flush=True)

cols=["-20%","-10%","nominal","+10%","+20%"]
with open("docs/luenberger_mismatch_v2.csv","w",newline="") as fp:
    wr=csv.writer(fp); wr.writerow(["parameter","metric"]+cols)
    for p in params:
        wr.writerow([p,"measured RMSE"]+[f"{res[(p,d)][0]:.4f}" for d in devs])
        wr.writerow([p,"hidden RMSE"]+[f"{res[(p,d)][1]:.4f}" for d in devs])
    wr.writerow(["combined worst","measured RMSE","","","%.4f"%res[("mass",0.0)][0],"","%.4f"%cw[0]])
    wr.writerow(["combined worst","hidden RMSE","","","%.4f"%res[("mass",0.0)][1],"","%.4f"%cw[1]])
print("saved -> docs/luenberger_mismatch_v2.csv", flush=True)
