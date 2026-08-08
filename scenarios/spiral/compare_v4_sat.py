"""
v4: steady-state metrics + realistic saturation + small gain sweep.  R=10, omega=0.6.

Isolated: pure imports only; writes ONLY docs/spiral_v4_sat.png. No datasets/models/observer touched.

Realistic limits (from Singha params, m=1.8 kg):
  thrust U1 in [1, 35] N  (thrust-to-weight ~= 2)
  tilt |phi_d|,|theta_d| <= 30 deg
  torques U2,U3,U4 in [-5, 5]
Steady state = t >= 5 s (skip launch transient; start (4,5,0) far from spiral@t=0).
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


def _tilt(Uex, Uey, phi, psi):
    Uex = np.clip(Uex, -0.99, 0.99); Uey = np.clip(Uey, -0.99, 0.99)
    phi_d = np.arcsin(Uex*np.sin(psi) - Uey*np.cos(psi))
    theta_d = np.arcsin(np.clip(
        (Uex - np.sin(phi)*np.sin(psi)) / (np.cos(phi)*np.cos(psi)), -0.99, 0.99))
    return phi_d, theta_d


def control(t, x, accel_ff, tilt_comp, g, sat):
    pos_d, vel_d, acc_d = spiral_ref(t)
    z, zdot, phi, theta, psi = x[2], x[5], x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    az_pd = g['kpz']*(pos_d[2]-z) + g['kdz']*(vel_d[2]-zdot)
    if tilt_comp:
        den = np.sign(np.cos(phi)*np.cos(theta))*max(abs(np.cos(phi)*np.cos(theta)), 0.25)
        U1 = M*(G + (acc_d[2] if accel_ff else 0.0) + az_pd)/den
    else:
        U1 = M*G + az_pd
    if sat:
        U1 = np.clip(U1, U1_MIN, U1_MAX)
    ax = g['kpxy']*(pos_d[0]-x[0]) + g['kdxy']*(vel_d[0]-x[3])
    ay = g['kpxy']*(pos_d[1]-x[1]) + g['kdxy']*(vel_d[1]-x[4])
    if accel_ff:
        ax += acc_d[0]; ay += acc_d[1]
    phi_d, theta_d = _tilt((M/U1)*ax, (M/U1)*ay, phi, PSI_D)
    if sat:
        phi_d = np.clip(phi_d, -TILT_MAX, TILT_MAX)
        theta_d = np.clip(theta_d, -TILT_MAX, TILT_MAX)
    U2 = (IX/L_ARM)*(g['kpatt']*(phi_d-phi) - g['kdatt']*p - q*r*(IY-IZ)/IX)
    U3 = (IY/L_ARM)*(g['kpatt']*(theta_d-theta) - g['kdatt']*q - p*r*(IZ-IX)/IY)
    U4 = IZ*(g['kpyaw']*(PSI_D-psi) - g['kdyaw']*r - p*q*(IX-IY)/IZ)
    if sat:
        U2, U3, U4 = (np.clip(v, -TAU_MAX, TAU_MAX) for v in (U2, U3, U4))
    return np.array([U1, U2, U3, U4])


base = dict(kpxy=KP_XY, kdxy=KD_XY, kpz=KP_Z, kdz=KD_Z,
            kpatt=KP_ATT, kdatt=KD_ATT, kpyaw=2.0, kdyaw=2.0)
hi   = dict(kpxy=3.0, kdxy=4.0, kpz=6.0, kdz=5.0,
            kpatt=25.0, kdatt=8.0, kpyaw=2.0, kdyaw=2.0)


def run(accel_ff, tilt_comp, g, sat, t_end=30.0, dt=0.01):
    x0 = np.zeros(12); x0[0], x0[1], x0[2] = 4.0, 5.0, 0.0
    t_eval = np.arange(0.0, t_end, dt)
    sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, control(t, x, accel_ff, tilt_comp, g, sat)),
                    (0.0, t_end), x0, t_eval=t_eval, max_step=0.02)
    P = np.array([spiral_ref(t)[0] for t in sol.t])
    m = sol.t >= SKIP
    rmse = np.sqrt(((sol.y[:3, m].T - P[m])**2).mean(axis=0))
    U = np.array([control(sol.t[k], sol.y[:, k], accel_ff, tilt_comp, g, sat) for k in range(len(sol.t))])
    tilt = np.degrees(np.sqrt(sol.y[6, m]**2 + sol.y[7, m]**2)).max()
    return sol.t, sol.y, P, rmse, U[m, 0].max(), tilt


print(f"\n====== STEADY-STATE (t>={SKIP}s), SATURATED @ R={R}, omega={W} ======", flush=True)
print(f"  {'variant':<20}{'x':>7}{'y':>7}{'z':>7}{'overall':>9}{'tilt°':>7}{'U1max':>7}", flush=True)
res = {}
for name, aff, tc, g in [("OLD",False,False,base),("V2",True,False,base),("V3 (all)",True,True,hi)]:
    t, Y, P, rmse, u1, tilt = run(aff, tc, g, sat=True)
    res[name] = (t, Y, P, rmse)
    print(f"  {name:<20}{rmse[0]:>7.3f}{rmse[1]:>7.3f}{rmse[2]:>7.3f}"
          f"{np.linalg.norm(rmse):>9.3f}{tilt:>7.1f}{u1:>7.1f}", flush=True)


print(f"\n====== GAIN SWEEP (accel FF + tilt-comp + sat, kpatt=25/kdatt=8) ======", flush=True)
print(f"  steady overall RMSE (m); rows KP_XY, cols KD_XY", flush=True)
KPS, KDS = [1.5, 3.0, 4.5], [3.0, 4.0, 5.0]
best = (1e9, None)
header = "  KP\\KD " + "".join(f"{kd:>9.1f}" for kd in KDS)
print(header, flush=True)
for kp in KPS:
    row = f"  {kp:>5.1f} "
    for kd in KDS:
        g = dict(kpxy=kp, kdxy=kd, kpz=6.0, kdz=5.0, kpatt=25.0, kdatt=8.0, kpyaw=2.0, kdyaw=2.0)
        _, _, _, rmse, u1, tilt = run(True, True, g, sat=True)
        ov = np.linalg.norm(rmse)
        row += f"{ov:>9.3f}"
        if ov < best[0]:
            best = (ov, (kp, kd, u1, tilt))
    print(row, flush=True)
kp, kd, u1, tilt = best[1]
print(f"\n  BEST: KP_XY={kp}, KD_XY={kd} -> steady RMSE {best[0]:.3f} m "
      f"(peak tilt {tilt:.1f}deg, peak U1 {u1:.1f} N)", flush=True)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, nm in enumerate(['x', 'y', 'z']):
    t, Y, P, _ = res["V3 (all)"]
    ax[i].plot(t, P[:, i], 'k-', lw=2, label='desired')
    for name, st in [("OLD", 'r--'), ("V2", 'g-.'), ("V3 (all)", 'b-')]:
        tt, YY, _, rr = res[name]
        ax[i].plot(tt, YY[i], st, lw=1.4, label=f"{name} (ss {rr[i]:.2f} m)")
    ax[i].axvspan(0, SKIP, color='gray', alpha=0.15)
    ax[i].set_title(nm); ax[i].set_xlabel("t (s)"); ax[i].grid(alpha=0.3); ax[i].legend(fontsize=7)
fig.suptitle(f"Saturated, steady-state (gray=skipped {SKIP}s)  R={R}, omega={W}")
out = os.path.join(PROJ, "docs", "spiral_v4_sat.png")
plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"\n  saved -> {out}", flush=True)
