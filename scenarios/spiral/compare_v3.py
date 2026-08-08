"""
Controller v3 ablation at FIXED aggressive spiral (R=10, omega=0.6).

Isolated: imports only pure modules; writes ONLY docs/spiral_v3_ablation.png.
Does NOT touch datasets, .pth models, or the cluster observer.

Levers tested on top of v2 (which already has acceleration feedforward):
  #1 retuned gains (faster outer + inner loops)
  #3 thrust tilt-compensation  U1 /= cos(phi)cos(theta)
Reports RMSE, normalized RMSE, peak tilt, and thrust range (feasibility).
"""
import sys, os
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJ)
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.generate_quadrotor_data import quadrotor_dynamics
from core.controller import (M, G, KP_Z, KD_Z, KP_XY, KD_XY,
                             KP_ATT, KD_ATT, IX, IY, IZ, L_ARM)

R, CLIMB, PSI_D, W = 10.0, 2.0, 0.7853, 0.6


def spiral_ref(t):
    x_d, y_d, z_d = R*np.sin(W*t), R*np.cos(W*t), CLIMB*t
    xd_d, yd_d, zd_d = R*W*np.cos(W*t), -R*W*np.sin(W*t), CLIMB
    xdd_d, ydd_d = -R*W**2*np.sin(W*t), -R*W**2*np.cos(W*t)
    return (np.array([x_d, y_d, z_d]),
            np.array([xd_d, yd_d, zd_d]),
            np.array([xdd_d, ydd_d, 0.0]))


def _tilt(Uex, Uey, phi, psi):
    Uex = np.clip(Uex, -0.99, 0.99); Uey = np.clip(Uey, -0.99, 0.99)
    phi_d = np.arcsin(Uex*np.sin(psi) - Uey*np.cos(psi))
    theta_d = np.arcsin(np.clip(
        (Uex - np.sin(phi)*np.sin(psi)) / (np.cos(phi)*np.cos(psi)), -0.99, 0.99))
    return phi_d, theta_d


def _inner(x, phi_d, theta_d, g):
    phi, theta, psi = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    U2 = (IX/L_ARM)*(g['kpatt']*(phi_d-phi) - g['kdatt']*p - q*r*(IY-IZ)/IX)
    U3 = (IY/L_ARM)*(g['kpatt']*(theta_d-theta) - g['kdatt']*q - p*r*(IZ-IX)/IY)
    U4 = IZ*(g['kpyaw']*(PSI_D-psi) - g['kdyaw']*r - p*q*(IX-IY)/IZ)
    return U2, U3, U4


def control(t, x, accel_ff, tilt_comp, g):
    pos_d, vel_d, acc_d = spiral_ref(t)
    z, zdot = x[2], x[5]
    phi, theta = x[6], x[7]
    az_pd = g['kpz']*(pos_d[2]-z) + g['kdz']*(vel_d[2]-zdot)
    if tilt_comp:
        den = np.cos(phi)*np.cos(theta)
        den = np.sign(den)*max(abs(den), 0.25)
        U1 = M*(G + (acc_d[2] if accel_ff else 0.0) + az_pd)/den
    else:
        U1 = M*G + az_pd
    px, py, vx, vy = x[0], x[1], x[3], x[4]
    ax = g['kpxy']*(pos_d[0]-px) + g['kdxy']*(vel_d[0]-vx)
    ay = g['kpxy']*(pos_d[1]-py) + g['kdxy']*(vel_d[1]-vy)
    if accel_ff:
        ax += acc_d[0]; ay += acc_d[1]
    Uex, Uey = (M/U1)*ax, (M/U1)*ay
    phi_d, theta_d = _tilt(Uex, Uey, phi, PSI_D)
    U2, U3, U4 = _inner(x, phi_d, theta_d, g)
    return np.array([U1, U2, U3, U4])


base = dict(kpxy=KP_XY, kdxy=KD_XY, kpz=KP_Z, kdz=KD_Z,
            kpatt=KP_ATT, kdatt=KD_ATT, kpyaw=2.0, kdyaw=2.0)
hi   = dict(kpxy=3.0, kdxy=4.0, kpz=6.0, kdz=5.0,
            kpatt=25.0, kdatt=8.0, kpyaw=2.0, kdyaw=2.0)


def run(accel_ff, tilt_comp, g, t_end=30.0, dt=0.01):
    x0 = np.zeros(12); x0[0], x0[1], x0[2] = 4.0, 5.0, 0.0
    t_eval = np.arange(0.0, t_end, dt)
    sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, control(t, x, accel_ff, tilt_comp, g)),
                    (0.0, t_end), x0, t_eval=t_eval, max_step=0.02)
    P = np.array([spiral_ref(t)[0] for t in sol.t])
    rmse = np.sqrt(((sol.y[:3].T - P)**2).mean(axis=0))
    U = np.array([control(sol.t[k], sol.y[:, k], accel_ff, tilt_comp, g) for k in range(len(sol.t))])
    tilt = np.degrees(np.sqrt(sol.y[6]**2 + sol.y[7]**2)).max()
    return sol.t, sol.y, P, rmse, U, tilt


variants = [
    ("OLD (vel FF)",     False, False, base),
    ("V2 (+accel FF)",   True,  False, base),
    ("V2 +tilt-comp",    True,  True,  base),
    ("V2 +hi-gains",     True,  False, hi),
    ("V3 (all)",         True,  True,  hi),
]
scale = np.array([R, R, CLIMB*30.0])
res = {}
print(f"\n================ ABLATION @ R={R}, omega={W} ================", flush=True)
print(f"  {'variant':<18}{'x':>8}{'y':>8}{'z':>8}{'overall':>9}{'nRMSE%':>9}{'tilt°':>7}{'U1max':>7}", flush=True)
for name, aff, tc, g in variants:
    t, Y, P, rmse, U, tilt = run(aff, tc, g)
    res[name] = (t, Y, P, rmse)
    overall = np.linalg.norm(rmse)
    nrmse = 100*np.linalg.norm(rmse/scale)/np.sqrt(3)
    print(f"  {name:<18}{rmse[0]:>8.3f}{rmse[1]:>8.3f}{rmse[2]:>8.3f}"
          f"{overall:>9.3f}{nrmse:>9.2f}{tilt:>7.1f}{U[:,0].max():>7.1f}", flush=True)


fig, ax = plt.subplots(1, 3, figsize=(15, 4))
styles = {"OLD (vel FF)": ('r--', 1.4), "V2 (+accel FF)": ('g-.', 1.4), "V3 (all)": ('b-', 1.6)}
for i, nm in enumerate(['x', 'y', 'z']):
    t, Y, P, _ = res["V3 (all)"]
    ax[i].plot(t, P[:, i], 'k-', lw=2, label='desired')
    for name, (st, lw) in styles.items():
        tt, YY, _, rr = res[name]
        ax[i].plot(tt, YY[i], st, lw=lw, label=f"{name} ({rr[i]:.2f} m)")
    ax[i].set_title(nm); ax[i].set_xlabel("t (s)"); ax[i].grid(alpha=0.3); ax[i].legend(fontsize=7)
fig.suptitle(f"Spiral R={R}, omega={W}: OLD vs V2 vs V3 (accel FF + tilt-comp + retuned gains)")
out = os.path.join(PROJ, "docs", "spiral_v3_ablation.png")
plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"\n  saved -> {out}", flush=True)
