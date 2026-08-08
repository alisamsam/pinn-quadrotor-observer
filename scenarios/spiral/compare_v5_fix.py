"""
v5: EXACT tilt inversion (fixes arcsin domain bug) + accel FF + tilt-comp + sat + steady.
R=10, omega=0.6. Isolated: pure imports; writes ONLY docs/spiral_v5_fix.png.

Exact inversion (rotate accel eqs by yaw psi):
    sin(phi_d)             = Uex*sin(psi) - Uey*cos(psi)
    sin(theta_d)*cos(phi_d)= Uex*cos(psi) + Uey*sin(psi)
Clamp each arcsin argument to [-1,1] -> never NaN, and it is exact.
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
SKIP = 5.0
U1_MIN, U1_MAX = 1.0, 35.0
TILT_MAX = np.radians(30.0)
TAU_MAX = 5.0


def spiral_ref(t):
    return (np.array([R*np.sin(W*t), R*np.cos(W*t), CLIMB*t]),
            np.array([R*W*np.cos(W*t), -R*W*np.sin(W*t), CLIMB]),
            np.array([-R*W**2*np.sin(W*t), -R*W**2*np.cos(W*t), 0.0]))


def tilt_exact(Uex, Uey, psi):
    b2 = np.clip(Uex*np.sin(psi) - Uey*np.cos(psi), -1.0, 1.0)   # = sin(phi_d)
    phi_d = np.arcsin(b2)
    b1 = Uex*np.cos(psi) + Uey*np.sin(psi)                        # = sin(theta_d)cos(phi_d)
    theta_d = np.arcsin(np.clip(b1/np.cos(phi_d), -1.0, 1.0))
    return phi_d, theta_d


def control(t, x, g):
    pos_d, vel_d, acc_d = spiral_ref(t)
    z, zdot, phi, theta, psi = x[2], x[5], x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    az_pd = g['kpz']*(pos_d[2]-z) + g['kdz']*(vel_d[2]-zdot)
    den = np.sign(np.cos(phi)*np.cos(theta))*max(abs(np.cos(phi)*np.cos(theta)), 0.25)
    U1 = np.clip(M*(G + acc_d[2] + az_pd)/den, U1_MIN, U1_MAX)
    ax = acc_d[0] + g['kpxy']*(pos_d[0]-x[0]) + g['kdxy']*(vel_d[0]-x[3])
    ay = acc_d[1] + g['kpxy']*(pos_d[1]-x[1]) + g['kdxy']*(vel_d[1]-x[4])
    phi_d, theta_d = tilt_exact((M/U1)*ax, (M/U1)*ay, PSI_D)
    phi_d = np.clip(phi_d, -TILT_MAX, TILT_MAX)
    theta_d = np.clip(theta_d, -TILT_MAX, TILT_MAX)
    U2 = (IX/L_ARM)*(g['kpatt']*(phi_d-phi) - g['kdatt']*p - q*r*(IY-IZ)/IX)
    U3 = (IY/L_ARM)*(g['kpatt']*(theta_d-theta) - g['kdatt']*q - p*r*(IZ-IX)/IY)
    U4 = IZ*(g['kpyaw']*(PSI_D-psi) - g['kdyaw']*r - p*q*(IX-IY)/IZ)
    U2, U3, U4 = (np.clip(v, -TAU_MAX, TAU_MAX) for v in (U2, U3, U4))
    return np.array([U1, U2, U3, U4])


def run(g, t_end=30.0, dt=0.01):
    x0 = np.zeros(12); x0[0], x0[1], x0[2] = 4.0, 5.0, 0.0
    t_eval = np.arange(0.0, t_end, dt)
    sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, control(t, x, g)),
                    (0.0, t_end), x0, t_eval=t_eval, max_step=0.02)
    P = np.array([spiral_ref(t)[0] for t in sol.t])
    m = sol.t >= SKIP
    rmse = np.sqrt(((sol.y[:3, m].T - P[m])**2).mean(axis=0))
    U = np.array([control(sol.t[k], sol.y[:, k], g) for k in range(len(sol.t))])
    tilt = np.degrees(np.sqrt(sol.y[6, m]**2 + sol.y[7, m]**2)).max()
    return sol.t, sol.y, P, rmse, U[m, 0].max(), tilt


print(f"\n====== GAIN SWEEP with EXACT inversion (steady, saturated) ======", flush=True)
print(f"  steady overall RMSE (m); rows KP_XY, cols KD_XY", flush=True)
KPS, KDS = [1.0, 1.5, 3.0, 4.5], [3.0, 4.0, 5.0]
best = (1e9, None)
print("  KP\\KD " + "".join(f"{kd:>9.1f}" for kd in KDS), flush=True)
for kp in KPS:
    row = f"  {kp:>5.1f} "
    for kd in KDS:
        g = dict(kpxy=kp, kdxy=kd, kpz=6.0, kdz=5.0, kpatt=25.0, kdatt=8.0, kpyaw=2.0, kdyaw=2.0)
        _, _, _, rmse, u1, tilt = run(g)
        ov = np.linalg.norm(rmse)
        row += f"{ov:>9.3f}"
        if np.isfinite(ov) and ov < best[0]:
            best = (ov, (kp, kd, u1, tilt))
    print(row, flush=True)
kp, kd, u1, tilt = best[1]
print(f"\n  BEST: KP_XY={kp}, KD_XY={kd} -> steady RMSE {best[0]:.3f} m "
      f"(peak tilt {tilt:.1f}deg, peak U1 {u1:.1f} N)", flush=True)
print(f"  (was NaN/3.126 at KP_XY=4.5,KD_XY=5.0 before the fix)", flush=True)


gbest = dict(kpxy=kp, kdxy=kd, kpz=6.0, kdz=5.0, kpatt=25.0, kdatt=8.0, kpyaw=2.0, kdyaw=2.0)
t, Y, P, rmse, u1, tilt = run(gbest)
print(f"\n  BEST per-axis steady RMSE:  x={rmse[0]:.3f}  y={rmse[1]:.3f}  z={rmse[2]:.3f} m", flush=True)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, nm in enumerate(['x', 'y', 'z']):
    ax[i].plot(t, P[:, i], 'k-', lw=2, label='desired')
    ax[i].plot(t, Y[i], 'b-', lw=1.5, label=f'v5 exact (ss {rmse[i]:.2f} m)')
    ax[i].axvspan(0, SKIP, color='gray', alpha=0.15)
    ax[i].set_title(nm); ax[i].set_xlabel("t (s)"); ax[i].grid(alpha=0.3); ax[i].legend(fontsize=8)
fig.suptitle(f"v5 exact tilt inversion  KP_XY={kp}, KD_XY={kd}  R={R}, omega={W}")
out = os.path.join(PROJ, "docs", "spiral_v5_fix.png")
plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"\n  saved -> {out}", flush=True)
