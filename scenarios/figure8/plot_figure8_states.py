"""Plot all 12 TRUE states of the figure-8 flight (no observer yet)."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from core.generate_quadrotor_data import quadrotor_dynamics
from simulate_figure8 import full_control_figure8
from figure8_reference import figure8_reference, yaw_reference, OMEGA

# simulate the flight (Table 5 start: 0,0,0)
x0 = np.zeros(12)
x0[6], x0[7], x0[8] = 0.087, 0.1042, 0.209
t_end = 2*np.pi/OMEGA
t_eval = np.arange(0.0, t_end, 0.01)
sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, full_control_figure8(t, x)),
                (0.0, t_end), x0, t_eval=t_eval, max_step=0.02)
X = sol.y.T
tt = sol.t

# desired position + yaw (the only states that HAVE a desired)
P = np.array([figure8_reference(t)[0] for t in tt])
V = np.array([figure8_reference(t)[1] for t in tt])
psi_d = np.array([yaw_reference(t)[0] for t in tt])

STATE = [("x","m"),("y","m"),("z","m"),
    ("x_dot","m/s"),("y_dot","m/s"),("z_dot","m/s"),
    ("phi (roll)","rad"),("theta (pitch)","rad"),("psi (yaw)","rad"),
    ("phi_dot","rad/s"),("theta_dot","rad/s"),("psi_dot","rad/s")]

fig, axes = plt.subplots(4, 3, figsize=(15, 12)); axes = axes.flatten()
for i,(name,unit) in enumerate(STATE):
    ax = axes[i]
    ax.plot(tt, X[:,i], 'b-', lw=1.3, label="actual")
    # overlay desired ONLY where one exists
    if i < 3:                              # x,y,z position
        ax.plot(tt, P[:,i], 'r--', lw=1.1, label="desired")
    elif i < 6:                            # velocities
        ax.plot(tt, V[:,i-3], 'r--', lw=1.1, label="desired")
    elif i == 8:                           # yaw (only commanded angle)
        ax.plot(tt, psi_d, 'r--', lw=1.1, label="desired")
    ax.set_title(f"{name} [{unit}]", fontsize=11); ax.grid(alpha=0.3)
    if i==0: ax.legend(fontsize=8)
fig.suptitle("Figure-8 flight: all 12 TRUE states (red dashed = desired where commanded)", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.98])
os.makedirs("docs", exist_ok=True)
plt.savefig("docs/figure8_states.png", dpi=110, bbox_inches="tight")
print("Saved -> docs/figure8_states.png")