"""EKF under model mismatch: nominal model, true drone has perturbed parameters.
Perturbed flights (controller_v2) are generated once and SAVED to datasets/mismatch_v2/
so the UKF can reuse them. Clean measurements. Jacobian F = I + dt*A (standard).
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

MEAS=[0,1,2,6,7,8]; HID=[3,4,5,9,10,11]
C=np.zeros((6,12));
for i,m in enumerate(MEAS): C[i,m]=1.0
Q=1e-5*np.eye(12); P0=1e-6*np.eye(12); R=1e-6*np.eye(6)
ctrl=lambda t,x: full_control_v2(t,x,spiral_reference,yaw_reference)

def f_step(x,u,dt):
    k1=nom_dyn(0.,x,u);k2=nom_dyn(0.,x+0.5*dt*k1,u);k3=nom_dyn(0.,x+0.5*dt*k2,u);k4=nom_dyn(0.,x+dt*k3,u)
    return x+dt/6.*(k1+2*k2+2*k3+k4)
def Fmat(x,u,dt,eps=1e-6):
    f0=nom_dyn(0.,x,u); A=np.empty((12,12))
    for i in range(12):
        dx=np.zeros(12);dx[i]=eps;A[:,i]=(nom_dyn(0.,x+dx,u)-f0)/eps
    return np.eye(12)+dt*A
def ekf(Xt,U,dt):
    xh=Xt[0].copy();P=P0.copy();est=np.zeros_like(Xt);est[0]=xh;I=np.eye(12)
    for k in range(Xt.shape[0]-1):
        u=U[k];xp=f_step(xh,u,dt);F=Fmat(xh,u,dt);P=F@P@F.T+Q
        y=Xt[k+1,MEAS];S=C@P@C.T+R;K=P@C.T@np.linalg.inv(S)
        xh=xp+K@(y-C@xp);P=(I-K@C)@P;est[k+1]=xh
    return est

Xb=np.load("datasets/spiral_v2_dataset.npz")["X"]; x0s=[Xb[i,0,:] for i in range(40,50)]
t_eval=np.arange(0.0,30.0,0.01); dt=0.01
os.makedirs("datasets/mismatch_v2", exist_ok=True)

def flights(tag,P):
    fp=f"datasets/mismatch_v2/{tag}.npz"
    if os.path.exists(fp):
        d=np.load(fp); return d["X"],d["U"]
    Xs,Us=[],[]
    for x0 in x0s:
        sol=solve_ivp(lambda t,x: true_dyn(t,x,ctrl(t,x),P),(0.0,30.0),x0,t_eval=t_eval,max_step=0.02)
        Xt=sol.y.T; U=np.array([ctrl(sol.t[k],Xt[k]) for k in range(len(sol.t))])
        Xs.append(Xt);Us.append(U)
    Xs=np.array(Xs);Us=np.array(Us); np.savez(fp,X=Xs,U=Us); return Xs,Us

def eval_set(tag,P):
    Xs,Us=flights(tag,P); per=[]
    for i in range(len(Xs)):
        est=ekf(Xs[i],Us[i],dt); per.append(np.sqrt(((est-Xs[i])**2).mean(axis=0)))
    per=np.array(per).mean(axis=0); return per[MEAS].mean(), per[HID].mean()


devs=[-0.20,-0.10,0.0,0.10,0.20]; params={"mass":["m"],"inertia":["Ix","Iy","Iz"],"arm":["l"]}
res={}
for p,keys in params.items():
    for dv in devs:
        P=dict(NOMINAL)
        for k in keys: P[k]=NOMINAL[k]*(1+dv)
        tag=f"{p}_{int(round(dv*100)):+d}"
        res[(p,dv)]=eval_set(tag,P)
        print(f"{p:>8} {dv:+.0%}: meas {res[(p,dv)][0]:.4f} | hidden {res[(p,dv)][1]:.4f}", flush=True)
Pw=dict(NOMINAL); Pw["m"]*=0.8; Pw["Ix"]*=1.2; Pw["Iy"]*=1.2; Pw["Iz"]*=1.2; Pw["l"]*=0.8
cw=eval_set("combined_worst",Pw); print(f"combined worst: meas {cw[0]:.4f} | hidden {cw[1]:.4f}", flush=True)

# worst deviation per parameter (by hidden RMSE)
print("\n=== worst condition per parameter (by hidden RMSE) ===", flush=True)
for p in params:
    wd=max(devs, key=lambda d: res[(p,d)][1])
    print(f"  {p:>8}: worst at {wd:+.0%}  ->  meas {res[(p,wd)][0]:.4f} | hidden {res[(p,wd)][1]:.4f}", flush=True)
print(f"  combined: meas {cw[0]:.4f} | hidden {cw[1]:.4f}", flush=True)

cols=["-20%","-10%","nominal","+10%","+20%"]
with open("docs/ekf_mismatch_v2.csv","w",newline="") as fp:
    wr=csv.writer(fp); wr.writerow(["parameter","metric"]+cols)
    for p in params:
        wr.writerow([p,"measured RMSE"]+[f"{res[(p,d)][0]:.4f}" for d in devs])
        wr.writerow([p,"hidden RMSE"]+[f"{res[(p,d)][1]:.4f}" for d in devs])
    wr.writerow(["combined worst","measured RMSE","","","%.4f"%res[("mass",0.0)][0],"","%.4f"%cw[0]])
    wr.writerow(["combined worst","hidden RMSE","","","%.4f"%res[("mass",0.0)][1],"","%.4f"%cw[1]])
print("saved -> docs/ekf_mismatch_v2.csv", flush=True)
