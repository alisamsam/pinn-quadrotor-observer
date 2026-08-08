"""
Controller v2 — add ACCELERATION feedforward to the spiral tracker (isolated).

Imports only pure modules (core.generate_quadrotor_data, core.controller, both
__main__-guarded). Writes ONLY new files docs/spiral_v2_compare_*.png. Does NOT touch
existing datasets, .pth models, or the cluster observer.

OLD law = faithful copy of scenarios/spiral/simulate_spiral.full_control_spiral
(position + VELOCITY feedforward). V2 adds the reference ACCELERATION that
spiral_reference already computes but the old law threw away.
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

KP_YAW, KD_YAW = 2.0, 2.0
R, CLIMB, PSI_D = 10.0, 2.0, 0.7853     # Singha spiral (matches spiral_reference.py)


def spiral_ref(t, w):
    x_d, y_d, z_d = R*np.sin(w*t), R*np.cos(w*t), CLIMB*t
    xd_d, yd_d, zd_d = R*w*np.cos(w*t), -R*w*np.sin(w*t), CLIMB
    xdd_d, ydd_d, zdd_d = -R*w**2*np.sin(w*t), -R*w**2*np.cos(w*t), 0.0
    return (np.array([x_d, y_d, z_d]),
            np.array([xd_d, yd_d, zd_d]),
            np.array([xdd_d, ydd_d, zdd_d]))


def _tilt(Uex, Uey, phi, psi):
    Uex = np.clip(Uex, -0.99, 0.99); Uey = np.clip(Uey, -0.99, 0.99)
    phi_d = np.arcsin(Uex*np.sin(psi) - Uey*np.cos(psi))
    theta_d = np.arcsin(np.clip(
        (Uex - np.sin(phi)*np.sin(psi)) / (np.cos(phi)*np.cos(psi)), -0.99, 0.99))
    return phi_d, theta_d


def _inner(x, phi_d, theta_d, psi_d, psi_d_dot):
    phi, theta, psi = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    U2 = (IX/L_ARM)*(KP_ATT*(phi_d-phi) - KD_ATT*p - q*r*(IY-IZ)/IX)
    U3 = (IY/L_ARM)*(KP_ATT*(theta_d-theta) - KD_ATT*q - p*r*(IZ-IX)/IY)
    U4 = IZ*(KP_YAW*(psi_d-psi) + KD_YAW*(psi_d_dot-r) - p*q*(IX-IY)/IZ)
    return U2, U3, U4


def old_control(t, x, w):
    """Velocity feedforward only — faithful copy of full_control_spiral."""
    pos_d, vel_d, _ = spiral_ref(t, w)
    z, zdot = x[2], x[5]
    U1 = M*G + KP_Z*(pos_d[2]-z) + KD_Z*(vel_d[2]-zdot)
    px, py, vx, vy = x[0], x[1], x[3], x[4]
    Uex = (M/U1)*(KP_XY*(pos_d[0]-px) + KD_XY*(vel_d[0]-vx))
    Uey = (M/U1)*(KP_XY*(pos_d[1]-py) + KD_XY*(vel_d[1]-vy))
    phi_d, theta_d = _tilt(Uex, Uey, x[6], PSI_D)
    U2, U3, U4 = _inner(x, phi_d, theta_d, PSI_D, 0.0)
    return np.array([U1, U2, U3, U4])


def v2_control(t, x, w):
    """v2: adds reference ACCELERATION feedforward to the horizontal loop."""
    pos_d, vel_d, acc_d = spiral_ref(t, w)
    z, zdot = x[2], x[5]
    U1 = M*G + KP_Z*(pos_d[2]-z) + KD_Z*(vel_d[2]-zdot)   # z: acc_d=0, identical to old
    px, py, vx, vy = x[0], x[1], x[3], x[4]
    ax = acc_d[0] + KP_XY*(pos_d[0]-px) + KD_XY*(vel_d[0]-vx)
    ay = acc_d[1] + KP_XY*(pos_d[1]-py) + KD_XY*(vel_d[1]-vy)
    Uex, Uey = (M/U1)*ax, (M/U1)*ay
    phi_d, theta_d = _tilt(Uex, Uey, x[6], PSI_D)
    U2, U3, U4 = _inner(x, phi_d, theta_d, PSI_D, 0.0)
    return np.array([U1, U2, U3, U4])


def run(ctrl, w, t_end=30.0, dt=0.01):
    x0 = np.zeros(12); x0[0], x0[1], x0[2] = 4.0, 5.0, 0.0
    t_eval = np.arange(0.0, t_end, dt)
    sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, ctrl(t, x, w)),
                    (0.0, t_end), x0, t_eval=t_eval, max_step=0.02)
    P = np.array([spiral_ref(t, w)[0] for t in sol.t])
    rmse = np.sqrt(((sol.y[:3].T - P)**2).mean(axis=0))
    return sol.t, sol.y, P, rmse


def compare(w):
    to, Yo, P, r_old = run(old_control, w)
    tv, Yv, P2, r_v2 = run(v2_control, w)
    scale = np.array([R, R, CLIMB*30.0])   # nRMSE denominators (amplitudes/span)
    print(f"\n=== omega = {w} (spiral R={R}, climb={CLIMB}) ===", flush=True)
    print(f"  {'axis':<5}{'OLD (m)':>10}{'V2 (m)':>10}{'OLD nRMSE%':>13}{'V2 nRMSE%':>12}", flush=True)
    for i, nm in enumerate(['x', 'y', 'z']):
        print(f"  {nm:<5}{r_old[i]:>10.4f}{r_v2[i]:>10.4f}"
              f"{100*r_old[i]/scale[i]:>13.2f}{100*r_v2[i]/scale[i]:>12.2f}", flush=True)
    print(f"  overall pos RMSE:  OLD {np.linalg.norm(r_old):.4f} m   "
          f"V2 {np.linalg.norm(r_v2):.4f} m   "
          f"(-{100*(1-np.linalg.norm(r_v2)/np.linalg.norm(r_old)):.1f}%)", flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, nm in enumerate(['x', 'y', 'z']):
        ax[i].plot(to, P[:, i], 'k-', lw=2, label='desired')
        ax[i].plot(to, Yo[i], 'r--', lw=1.6, label='OLD (vel FF)')
        ax[i].plot(tv, Yv[i], 'b-', lw=1.4, label='V2 (+accel FF)')
        ax[i].set_title(f"{nm}   OLD {r_old[i]:.3f} m  ->  V2 {r_v2[i]:.3f} m")
        ax[i].set_xlabel("t (s)"); ax[i].grid(alpha=0.3); ax[i].legend(fontsize=8)
    fig.suptitle(f"Spiral tracking: acceleration feedforward, omega={w}")
    out = os.path.join(PROJ, "docs", f"spiral_v2_compare_w{int(round(w*100)):03d}.png")
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  saved -> {out}", flush=True)


if __name__ == "__main__":
    compare(0.6)      # your current aggressive spiral
    compare(0.35)     # gentler, per Prof. Chaibet's advice (isolates trajectory-speed effect)
