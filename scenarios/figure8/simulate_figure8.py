"""T.2: fly Singha's Figure8 trajectory with the full controller + feedforward."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from core.generate_quadrotor_data import quadrotor_dynamics
from core.controller import (M, G, KP_Z, KD_Z, KP_XY, KD_XY,
                             KP_ATT, KD_ATT, IX, IY, IZ, L_ARM, outer_loop)
from figure8_reference import figure8_reference, yaw_reference, OMEGA

KP_YAW, KD_YAW = 2.0, 2.0


def full_control_figure8(t, x):
    """Time-aware controller tracking the moving figure-8 (with feedforward)."""
    pos_d, vel_d, acc_d = figure8_reference(t)     # moving target + derivatives
    psi_d, psi_d_dot = yaw_reference(t)

    # --- altitude (z) with feedforward velocity ---
    z, zdot = x[2], x[5]
    U1 = M*G + KP_Z*(pos_d[2] - z) + KD_Z*(vel_d[2] - zdot)

    # --- outer loop: position error -> desired tilt, WITH feedforward velocity ---
    # we pass the moving target into the outer loop
    phi_d, theta_d = outer_loop_ff(x, U1, pos_d, vel_d, psi_d)

    # --- inner loop: attitude -> torques (time-varying yaw) ---
    phi, theta, psi = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    U2 = (IX/L_ARM)*(KP_ATT*(phi_d - phi) - KD_ATT*p - q*r*(IY-IZ)/IX)
    U3 = (IY/L_ARM)*(KP_ATT*(theta_d - theta) - KD_ATT*q - p*r*(IZ-IX)/IY)
    U4 = IZ*(KP_YAW*(psi_d - psi) + KD_YAW*(psi_d_dot - r) - p*q*(IX-IY)/IZ)
    return np.array([U1, U2, U3, U4])


def outer_loop_ff(x, U1, pos_d, vel_d, psi):
    """Outer loop WITH feedforward: track moving (pos_d, vel_d)."""
    px, py = x[0], x[1]
    vx, vy = x[3], x[4]
    # PD on position error + feedforward on desired velocity
    Uex = (M/U1) * (KP_XY*(pos_d[0]-px) + KD_XY*(vel_d[0]-vx))
    Uey = (M/U1) * (KP_XY*(pos_d[1]-py) + KD_XY*(vel_d[1]-vy))
    Uex = np.clip(Uex, -0.99, 0.99)
    Uey = np.clip(Uey, -0.99, 0.99)
    phi_d = np.arcsin(Uex*np.sin(psi) - Uey*np.cos(psi))
    phi = x[6]
    theta_d = np.arcsin(np.clip(
        (Uex - np.sin(phi)*np.sin(psi)) / (np.cos(phi)*np.cos(psi)), -0.99, 0.99))
    return phi_d, theta_d


# ===== simulate =====

x0 = np.zeros(12)
# Table 5: figure-8 trajectory initial position (0,0,0), initial angles (0.087, 0.1042, 0.209)
x0[0], x0[1], x0[2] = 0.0, 0.0, 0.0          # start at (0,0,0.0) per Table 5
x0[3], x0[4], x0[5] = 0.0, 0.0, 0.0          # at rest (drone placed, not moving)
x0[6], x0[7], x0[8] = 0.087, 0.1042, 0.209   # Table 5 initial angles

t_end = 2*np.pi/OMEGA                               # ~21, one full figure-8
t_eval = np.arange(0.0, t_end, 0.01)

sol = solve_ivp(
    lambda t, x: quadrotor_dynamics(t, x, full_control_figure8(t, x)),
    (0.0, t_end), x0, t_eval=t_eval, max_step=0.02,
)
x, y, z = sol.y[0], sol.y[1], sol.y[2]
print(f"Start: ({x[0]:.2f},{y[0]:.2f},{z[0]:.2f})")
print(f"End:   ({x[-1]:.2f},{y[-1]:.2f},{z[-1]:.2f})")

# desired figure-8 for reference
P = np.array([figure8_reference(t)[0] for t in t_eval])

# 3D plot: ACTUAL connected flight path + desired figure-8
# DIAGNOSTIC: every key variable vs time
# use the solver's ACTUAL returned times (handles early-stop safely)
tt = sol.t
P = np.array([figure8_reference(t)[0] for t in tt])

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0,0].plot(tt, x, 'b', label='actual'); ax[0,0].plot(tt, P[:,0], 'r--', label='desired'); ax[0,0].set_title("x"); ax[0,0].legend()
ax[0,1].plot(tt, y, 'b'); ax[0,1].plot(tt, P[:,1], 'r--'); ax[0,1].set_title("y")
ax[0,2].plot(tt, z, 'b'); ax[0,2].plot(tt, P[:,2], 'r--'); ax[0,2].set_title("z: actual vs desired")
ax[1,0].plot(tt, sol.y[6], 'g'); ax[1,0].set_title("phi (roll)")
ax[1,1].plot(tt, sol.y[7], 'g'); ax[1,1].set_title("theta (pitch)")
ax[1,2].plot(tt, z, 'b'); ax[1,2].plot(tt, P[:,2], 'r--'); ax[1,2].set_title("z (zoom)"); ax[1,2].set_xlim(0, 5)
for a in ax.flat: a.grid(alpha=0.3); a.set_xlabel("t(s)")
plt.tight_layout(); plt.savefig("docs/figure8_iso_debug.png", dpi=120, bbox_inches="tight")
print("Saved -> docs/figure8_iso_debug.png")

# also print NUMERIC symptoms at key times
for tc in [0.0, 0.5, 1.0, 2.0]:
    i = int(tc/0.01)
    if i < len(z):
        print(f"t={tc}: z={z[i]:.2f}, phi={sol.y[6][i]:.3f}, theta={sol.y[7][i]:.3f}")