"""Singha Fig 2 scenario: start (6,0.2,0) -> target (0.6,0.6,10), yaw=0.1 sin(t)."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from data.generate_quadrotor_data import quadrotor_dynamics
from data.controller import (M, G, KP_Z, KD_Z, KP_XY, KD_XY,
                             KP_ATT, KD_ATT, IX, IY, IZ, L_ARM, outer_loop)

# gentler gains for tracking the smooth yaw sine (vs aggressive setpoint gains)
KP_YAW = 2.0
KD_YAW = 2.0

# ===== Singha scenario targets =====
X_TGT, Y_TGT, Z_TGT = 0.6, 0.6, 10.0


def full_control_singha(t, x):
    """Time-aware controller: tracks (0.6,0.6,10) with yaw = 0.1 sin(t)."""
    # --- time-varying yaw command + its feedforward derivative ---
    psi_d    = 0.1 * np.sin(t)
    psi_d_dot = 0.1 * np.cos(t)        # feedforward

    # --- thrust (altitude) toward z=10 ---
    z, zdot = x[2], x[5]
    U1 = M * G + KP_Z * (Z_TGT - z) - KD_Z * zdot

    # --- outer loop: position -> desired tilt ---
    phi_d, theta_d = outer_loop(x, U1, X_TGT, Y_TGT)

    # --- inner loop: attitude, now with time-varying yaw target + feedforward ---
    phi, theta, psi = x[6], x[7], x[8]
    p, q, r         = x[9], x[10], x[11]
    U2 = (IX/L_ARM) * (KP_ATT*(phi_d - phi) - KD_ATT*p - q*r*(IY-IZ)/IX)
    U3 = (IY/L_ARM) * (KP_ATT*(theta_d - theta) - KD_ATT*q - p*r*(IZ-IX)/IY)
    # yaw: track psi_d(t), with feedforward psi_d_dot
    U4 = IZ * (KP_YAW*(psi_d - psi) + KD_YAW*(psi_d_dot - r) - p*q*(IX-IY)/IZ)

    return np.array([U1, U2, U3, U4])


# ===== simulate =====
x0 = np.zeros(12)
x0[0], x0[1], x0[2] = 6.0, 0.2, 0.0        # Singha start position

t_end = 18.0                                 # longer: big climb needs time
t_eval = np.arange(0.0, t_end, 0.01)

sol = solve_ivp(
    lambda t, x: quadrotor_dynamics(t, x, full_control_singha(t, x)),
    (0.0, t_end), x0, t_eval=t_eval, max_step=0.01,
)

x, y, z, psi = sol.y[0], sol.y[1], sol.y[2], sol.y[8]
print(f"Start:  ({x[0]:.2f}, {y[0]:.2f}, {z[0]:.2f})")
print(f"Final:  ({x[-1]:.3f}, {y[-1]:.3f}, {z[-1]:.3f})")
print(f"Target: ({X_TGT}, {Y_TGT}, {Z_TGT})")
reached = abs(x[-1]-X_TGT)<0.2 and abs(y[-1]-Y_TGT)<0.2 and abs(z[-1]-Z_TGT)<0.2
print("Reached setpoint?", reached)

# plot position + yaw
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
ax[0,0].plot(sol.t, x, 'b'); ax[0,0].axhline(X_TGT, color='r', ls='--'); ax[0,0].set_title("x (m)")
ax[0,1].plot(sol.t, y, 'b'); ax[0,1].axhline(Y_TGT, color='r', ls='--'); ax[0,1].set_title("y (m)")
ax[1,0].plot(sol.t, z, 'b'); ax[1,0].axhline(Z_TGT, color='r', ls='--'); ax[1,0].set_title("z (m)")
ax[1,1].plot(sol.t, psi, 'b', label='actual')
ax[1,1].plot(sol.t, 0.1*np.sin(sol.t), 'r--', label='desired 0.1 sin t'); ax[1,1].set_title("yaw psi (rad)"); ax[1,1].legend()
for a in ax.flat: a.grid(alpha=0.3); a.set_xlabel("t (s)")
fig.suptitle("Singha Fig 2 scenario: (6,0.2,0) -> (0.6,0.6,10), yaw=0.1 sin t")
fig.tight_layout()
os.makedirs("docs", exist_ok=True)
plt.savefig("docs/singha_scenario.png", dpi=120, bbox_inches="tight")
print("Saved -> docs/singha_scenario.png")