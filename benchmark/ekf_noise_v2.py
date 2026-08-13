"""Extended Kalman Filter (EKF) on spiral_v2 test flights: clean + noise.
Discrete EKF: predict with RK4 of the nominal Singha model, numerical Jacobian for
covariance, linear measurement C (6 of 12). R set to the true noise per case; fixed Q.
Start = true x0. Evaluated vs true states. Compared to PINN and Luenberger.
"""
import sys, os, csv
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import numpy as np
from core.generate_quadrotor_data import quadrotor_dynamics

MEAS = [0,1,2,6,7,8]; HID = [3,4,5,9,10,11]
C = np.zeros((6,12));
for i,m in enumerate(MEAS): C[i,m] = 1.0
Q = 1e-5*np.eye(12)          # process-noise (model trust); fixed
P0 = 1e-6*np.eye(12)         # start confident (true x0)

def f_step(x,u,dt):
    k1=quadrotor_dynamics(0.,x,u); k2=quadrotor_dynamics(0.,x+0.5*dt*k1,u)
    k3=quadrotor_dynamics(0.,x+0.5*dt*k2,u); k4=quadrotor_dynamics(0.,x+dt*k3,u)
    return x + dt/6.*(k1+2*k2+2*k3+k4)

def jac(x,u,dt,eps=1e-6):
    f0=f_step(x,u,dt); F=np.empty((12,12))
    for i in range(12):
        dx=np.zeros(12); dx[i]=eps
        F[:,i]=(f_step(x+dx,u,dt)-f0)/eps
    return F

def ekf_run(Xtrue,U,Ymeas,dt,R):
    xh=Xtrue[0].copy(); P=P0.copy(); est=np.zeros_like(Xtrue); est[0]=xh; I=np.eye(12)
    for k in range(Xtrue.shape[0]-1):
        u=U[k]
        xp=f_step(xh,u,dt); F=jac(xh,u,dt); P=F@P@F.T+Q
        y=Ymeas[k+1]; S=C@P@C.T+R; K=P@C.T@np.linalg.inv(S)
        xh=xp+K@(y-C@xp); P=(I-K@C)@P; est[k+1]=xh
    return est

d=np.load("datasets/spiral_v2_dataset.npz"); T,X,U=d["T"],d["X"],d["U"]
dt=float(T[1]-T[0]); test=list(range(40,50))
CASES=[("clean",0.0,0.0),("moderate",0.05,0.01),("strong",0.10,0.02)]
PINN={"clean":(0.0442,0.0559),"moderate":(0.0775,0.0699),"strong":(0.0850,0.0720)}
LUEN={"clean":(0.0100,0.0094),"moderate":(0.0127,0.0213),"strong":(0.0183,0.0368)}  # w=5

def R_of(sp,sa):
    if sp==0: return 1e-6*np.eye(6)
    return np.diag([sp*sp]*3+[sa*sa]*3)

rows=[]
print(f"  {'case':<9}{'EKF meas':>10}{'EKF hid':>9}{'Luen meas':>11}{'Luen hid':>10}{'PINN meas':>11}{'PINN hid':>10}", flush=True)
for name,sp,sa in CASES:
    rng=np.random.default_rng(0); R=R_of(sp,sa); per=[]
    for f in test:
        Ym=X[f][:,MEAS].copy()
        if sp>0: Ym[:,0:3]+=sp*rng.standard_normal(Ym[:,0:3].shape)
        if sa>0: Ym[:,3:6]+=sa*rng.standard_normal(Ym[:,3:6].shape)
        est=ekf_run(X[f],U[f],Ym,dt,R); per.append(np.sqrt(((est-X[f])**2).mean(axis=0)))
    per=np.array(per).mean(axis=0); em,eh=per[MEAS].mean(),per[HID].mean()
    lm,lh=LUEN[name]; pm,ph=PINN[name]
    print(f"  {name:<9}{em:>10.4f}{eh:>9.4f}{lm:>11.4f}{lh:>10.4f}{pm:>11.4f}{ph:>10.4f}", flush=True)
    rows.append([name,sp,sa,f"{em:.4f}",f"{eh:.4f}"])

os.makedirs("docs",exist_ok=True)
with open("docs/ekf_noise_v2.csv","w",newline="") as fp:
    wr=csv.writer(fp); wr.writerow(["case","sigma_pos","sigma_att","EKF_meas_RMSE","EKF_hidden_RMSE"]); wr.writerows(rows)
print("saved -> docs/ekf_noise_v2.csv", flush=True)
