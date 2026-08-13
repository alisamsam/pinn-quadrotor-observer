"""UKF under model mismatch, reusing the SAVED perturbed flights (datasets/mismatch_v2/).
Nominal model, clean measurements. Same UKF as ukf_noise_v2 (Euler sigma propagation)."""
import sys, os, csv
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import numpy as np
from core.generate_quadrotor_data import quadrotor_dynamics

MEAS=[0,1,2,6,7,8]; HID=[3,4,5,9,10,11]
C=np.zeros((6,12));
for i,m in enumerate(MEAS): C[i,m]=1.0
n=12; alpha=1e-3; beta=2.0; kappa=0.0; lam=alpha**2*(n+kappa)-n
Wm=np.full(2*n+1,1.0/(2*(n+lam))); Wc=Wm.copy(); Wm[0]=lam/(n+lam); Wc[0]=Wm[0]+(1-alpha**2+beta)
Q=1e-5*np.eye(12); P0=1e-6*np.eye(12); R=1e-6*np.eye(6)

def f_step(x,u,dt): return x + dt*quadrotor_dynamics(0.,x,u)
def sigma(x,P):
    S=np.linalg.cholesky((n+lam)*(P+1e-12*np.eye(12)))
    return np.array([x]+[x+S[:,i] for i in range(n)]+[x-S[:,i] for i in range(n)])
def ukf(Xt,U,dt):
    xh=Xt[0].copy(); P=P0.copy(); est=np.zeros_like(Xt); est[0]=xh
    for k in range(Xt.shape[0]-1):
        u=U[k]; X=sigma(xh,P); Xp=np.array([f_step(s,u,dt) for s in X])
        xm=Wm@Xp; dX=Xp-xm; Pm=(dX.T*Wc)@dX+Q
        Yp=Xp@C.T; ym=Wm@Yp; dY=Yp-ym; Pyy=(dY.T*Wc)@dY+R; Pxy=(dX.T*Wc)@dY
        K=Pxy@np.linalg.inv(Pyy); xh=xm+K@(Xt[k+1,MEAS]-ym); P=Pm-K@Pyy@K.T; est[k+1]=xh
    return est

dt=0.01
def eval_tag(tag):
    d=np.load(f"datasets/mismatch_v2/{tag}.npz"); Xs,Us=d["X"],d["U"]; per=[]
    for i in range(len(Xs)):
        est=ukf(Xs[i],Us[i],dt); per.append(np.sqrt(((est-Xs[i])**2).mean(axis=0)))
    per=np.array(per).mean(axis=0); return per[MEAS].mean(), per[HID].mean()

devs=[-20,-10,0,10,20]; params=["mass","inertia","arm"]; res={}
for p in params:
    for dv in devs:
        tag=f"{p}_{dv:+d}"; res[(p,dv)]=eval_tag(tag)
        print(f"{p:>8} {dv:+d}%: meas {res[(p,dv)][0]:.4f} | hidden {res[(p,dv)][1]:.4f}", flush=True)
cw=eval_tag("combined_worst"); print(f"combined worst: meas {cw[0]:.4f} | hidden {cw[1]:.4f}", flush=True)
print("\nworst per parameter (by hidden):", flush=True)
for p in params:
    wd=max(devs,key=lambda d:res[(p,d)][1]); print(f"  {p:>8}: {wd:+d}% -> meas {res[(p,wd)][0]:.4f} | hidden {res[(p,wd)][1]:.4f}", flush=True)

cols=["-20%","-10%","nominal","+10%","+20%"]
with open("docs/ukf_mismatch_v2.csv","w",newline="") as fp:
    wr=csv.writer(fp); wr.writerow(["parameter","metric"]+cols)
    for p in params:
        wr.writerow([p,"measured RMSE"]+[f"{res[(p,d)][0]:.4f}" for d in devs])
        wr.writerow([p,"hidden RMSE"]+[f"{res[(p,d)][1]:.4f}" for d in devs])
    wr.writerow(["combined worst","measured RMSE","","","%.4f"%res[("mass",0)][0],"","%.4f"%cw[0]])
    wr.writerow(["combined worst","hidden RMSE","","","%.4f"%res[("mass",0)][1],"","%.4f"%cw[1]])
print("saved -> docs/ukf_mismatch_v2.csv", flush=True)
