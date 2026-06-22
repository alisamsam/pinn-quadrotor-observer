"""3D flight path for the Singha Fig 2 scenario."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from core.generate_quadrotor_data import quadrotor_dynamics
from core.controller import (M, G, KP_Z, KD_Z, KP_ATT, KD_ATT,
                             IX, IY, IZ, L_ARM, outer_loop)

KP_YAW, KD_YAW = 2.0, 2.0
X_TGT, Y_TGT, Z_TGT = 0.6, 0.6, 10.0
X_START, Y_START, Z_START = 6.0, 0.2, 0.0


def full_control_singha(t, x):
    psi_d, psi_d_dot = 0.1*np.sin(t), 0.1*np.cos(t)
    z, zdot = x[2], x[5]
    U1 = M*G + KP_Z*(Z_TGT - z) - KD_Z*zdot
    phi_d, theta_d = outer_loop(x, U1, X_TGT, Y_TGT)
    phi, theta, psi = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    U2 = (IX/L_ARM)*(KP_ATT*(phi_d-phi) - KD_ATT*p - q*r*(IY-IZ)/IX)
    U3 = (IY/L_ARM)*(KP_ATT*(theta_d-theta) - KD_ATT*q - p*r*(IZ-IX)/IY)
    U4 = IZ*(KP_YAW*(psi_d-psi) + KD_YAW*(psi_d_dot-r) - p*q*(IX-IY)/IZ)
    return np.array([U1, U2, U3, U4])


# simulate
x0 = np.zeros(12)
x0[0], x0[1], x0[2] = X_START, Y_START, Z_START
t_eval = np.arange(0.0, 18.0, 0.01)
sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, full_control_singha(t, x)),
                (0.0, 18.0), x0, t_eval=t_eval, max_step=0.01)
x, y, z = sol.y[0], sol.y[1], sol.y[2]

# 3D plot
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

# the actual flight path, colored by time so you can see direction of travel
ax.plot(x, y, z, 'b-', linewidth=2, label="Actual path", alpha=0.8)

# key markers
ax.scatter(X_START, Y_START, Z_START, color='black', s=120, marker='o',
           label=f"Start ({X_START}, {Y_START}, {Z_START})")
ax.scatter(X_TGT, Y_TGT, Z_TGT, color='green', s=200, marker='*',
           label=f"Desired ({X_TGT}, {Y_TGT}, {Z_TGT})")
ax.scatter(x[-1], y[-1], z[-1], color='red', s=120, marker='X',
           label=f"Actual end ({x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f})")

# faint vertical line from ground to target, to anchor the eye
ax.plot([X_TGT, X_TGT], [Y_TGT, Y_TGT], [0, Z_TGT], 'g:', alpha=0.4)

ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
ax.set_title("Singha Fig 2: 3D flight path  (6,0.2,0) -> (0.6,0.6,10)")
ax.legend(loc='upper left', fontsize=9)
ax.view_init(elev=20, azim=-60)   # a viewing angle that shows the climb + curve

os.makedirs("docs", exist_ok=True)
plt.savefig("docs/singha_3d_path.png", dpi=120, bbox_inches="tight")
print("Saved -> docs/singha_3d_path.png")
print(f"Start: ({X_START},{Y_START},{Z_START})  End: ({x[-1]:.2f},{y[-1]:.2f},{z[-1]:.2f})  Target: ({X_TGT},{Y_TGT},{Z_TGT})")