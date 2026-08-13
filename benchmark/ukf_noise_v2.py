"""Unscented Kalman Filter (UKF) on spiral_v2: clean + noise.
Sigma-point propagation through the nonlinear model (no Jacobian). Linear measurement
C (6 of 12). R = true noise per case; fixed Q. Start = true x0. vs PINN/Luen/EKF.
"""
import sys, os, csv
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import numpy as np
from core.generate_quadrotor_data import quadrotor_dynamics

MEAS=[0,1,2,6,7,8]; HID=[3,4,5,9,10,11]
C=np.zeros((6,12));
for i,m in enumerate(MEAS): C[i,m]=1.0
n=12; alpha=1e-3; beta=2.0; kappa=0.0
lam=alpha**2*(n+kappa)-n
Wm=np.full(2*n+1, 1.0/(2*(n+lam))); Wc=Wm.copy()
Wm[0]=lam/(n+lam); Wc[0]=Wm[0]+(1-alpha**2+beta)
Q=1e-5*np.eye(12); P0=1e-6*np.eye(12)

def f_step(x,u,dt):        # Euler propagation (fast; dt=0.01)
    return x + dt*quadrotor_dynamics(0.,x,u)

def sigma(x,P):
    S=np.linalg.cholesky((n+lam)*(P+1e-12*np.eye(12)))
    pts=[x]+[x+S[:,i] for i in range(n)]+[x-S[:,i] for i in range(n)]
    return np.array(pts)

def ukf(Xt,U,Ym,dt,R):     # standard UKF: one sigma draw, reuse for measurement
    xh=Xt[0].copy(); P=P0.copy(); est=np.zeros_like(Xt); est[0]=xh
    for k in range(Xt.shape[0]-1):
        u=U[k]; X=sigma(xh,P)
        Xp=np.array([f_step(s,u,dt) for s in X])
        xm=Wm@Xp; dX=Xp-xm; Pm=(dX.T*Wc)@dX+Q
        Yp=Xp@C.T; ym=Wm@Yp
        dY=Yp-ym; Pyy=(dY.T*Wc)@dY+R; Pxy=(dX.T*Wc)@dY
        K=Pxy@np.linalg.inv(Pyy); xh=xm+K@(Ym[k+1]-ym); P=Pm-K@Pyy@K.T
        est[k+1]=xh
    return est

d=np.load("datasets/spiral_v2_dataset.npz"); T,X,U=d["T"],d["X"],d["U"]
dt=float(T[1]-T[0]); test=list(range(40,50))
CASES=[("clean",0.0,0.0),("moderate",0.05,0.01),("strong",0.10,0.02)]
def R_of(sp,sa): return 1e-6*np.eye(6) if sp==0 else np.diag([sp*sp]*3+[sa*sa]*3)

rows=[]
print(f"  {'case':<9}{'UKF meas':>10}{'UKF hid':>9}", flush=True)
for name,sp,sa in CASES:
    rng=np.random.default_rng(0); R=R_of(sp,sa); per=[]
    for f in test:
        Ym=X[f][:,MEAS].copy()
        if sp>0: Ym[:,0:3]+=sp*rng.standard_normal(Ym[:,0:3].shape)
        if sa>0: Ym[:,3:6]+=sa*rng.standard_normal(Ym[:,3:6].shape)
        est=ukf(X[f],U[f],Ym,dt,R); per.append(np.sqrt(((est-X[f])**2).mean(axis=0)))
    per=np.array(per).mean(axis=0); em,eh=per[MEAS].mean(),per[HID].mean()
    print(f"  {name:<9}{em:>10.4f}{eh:>9.4f}", flush=True); rows.append([name,sp,sa,f"{em:.4f}",f"{eh:.4f}"])

os.makedirs("docs",exist_ok=True)
with open("docs/ukf_noise_v2.csv","w",newline="") as fp:
    wr=csv.writer(fp); wr.writerow(["case","sigma_pos","sigma_att","UKF_meas_RMSE","UKF_hidden_RMSE"]); wr.writerows(rows)
print("saved -> docs/ukf_noise_v2.csv", flush=True)
