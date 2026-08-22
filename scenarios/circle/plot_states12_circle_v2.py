"""controller_v2 on the circle: (1) standalone 3D figure, (2) all-12-states grid.
Mirror of scenarios/spiral/plot_states12_v2.py with the circle reference."""
import sys, os
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJ)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from core.generate_quadrotor_data import quadrotor_dynamics
from core.controller_v2 import full_control_v2, tilt_from_accel, DEFAULT_GAINS, LIMITS, M, G
from circle_reference import circular_reference, yaw_reference

g, lim = DEFAULT_GAINS, LIMITS
x0 = np.zeros(12); x0[0], x0[1], x0[2] = 4.0, 5.0, 0.0
t_eval = np.arange(0.0, 30.0, 0.01)
ctrl = lambda t, x: full_control_v2(t, x, circular_reference, yaw_reference)
sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, ctrl(t, x)),
                (0.0, 30.0), x0, t_eval=t_eval, max_step=0.02)
t, Y = sol.t, sol.y
pos_d = np.array([circular_reference(tt)[0] for tt in t])
vel_d = np.array([circular_reference(tt)[1] for tt in t])
psi_d = np.array([yaw_reference(tt)[0] for tt in t])

# recompute desired roll/pitch (phi_d, theta_d) along the flight (mirrors controller_v2)
phi_d = np.zeros_like(t); theta_d = np.zeros_like(t)
for k in range(len(t)):
    x = Y[:, k]; p_d, v_d, a_d = circular_reference(t[k])
    az = a_d[2] + g['kp_z']*(p_d[2]-x[2]) + g['kd_z']*(v_d[2]-x[5])
    den = np.cos(x[6])*np.cos(x[7]); den = np.sign(den)*max(abs(den), 0.25)
    U1 = float(np.clip(M*(G+az)/den, lim['u1_min'], lim['u1_max']))
    ax = a_d[0] + g['kp_xy']*(p_d[0]-x[0]) + g['kd_xy']*(v_d[0]-x[3])
    ay = a_d[1] + g['kp_xy']*(p_d[1]-x[1]) + g['kd_xy']*(v_d[1]-x[4])
    pd_, td_ = tilt_from_accel((M/U1)*ax, (M/U1)*ay, x[8])
    phi_d[k] = np.clip(pd_, -lim['tilt_max'], lim['tilt_max'])
    theta_d[k] = np.clip(td_, -lim['tilt_max'], lim['tilt_max'])

# ---------- Figure 1: standalone 3D ----------
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos_d[:, 0], pos_d[:, 1], pos_d[:, 2], 'b-', lw=2.3, label='desired')
ax.plot(Y[0], Y[1], Y[2], 'r--', lw=1.8, label='actual')
ax.scatter([x0[0]], [x0[1]], [x0[2]], c='k', s=45, label='start (4,5,0)')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
ax.set_title('Circle tracking: desired vs actual')
ax.legend(); ax.view_init(elev=18, azim=-60)
out3d = os.path.join(PROJ, "docs", "circle_3d_v2.png")
plt.tight_layout(); plt.savefig(out3d, dpi=120, bbox_inches="tight"); plt.close(fig)
print("saved ->", out3d)

# ---------- Figure 2: all 12 states ----------
labels = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r']
units = ['m', 'm', 'm', 'm/s', 'm/s', 'm/s', 'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s']
desired = [pos_d[:, 0], pos_d[:, 1], pos_d[:, 2],
           vel_d[:, 0], vel_d[:, 1], vel_d[:, 2],
           phi_d, theta_d, psi_d,
           None, None, None]

fig, axes = plt.subplots(4, 3, figsize=(15, 11))
for i, ax in enumerate(axes.flat):
    ax.plot(t, Y[i], 'r-', lw=1.3, label='actual')
    if desired[i] is not None:
        ax.plot(t, desired[i], 'b--', lw=1.6, label='desired')
    ax.set_title(labels[i], fontsize=10); ax.grid(alpha=0.3)
    ax.set_xlabel('time  t  (s)', fontsize=8)
    ax.set_ylabel(f'{labels[i]}  ({units[i]})', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
fig.suptitle('All 12 states on the circle trajectory: actual vs desired '
             '(p, q, r have no reference).', fontsize=12)
out12 = os.path.join(PROJ, "docs", "circle_12states_v2.png")
plt.tight_layout(rect=[0, 0, 1, 0.98]); plt.savefig(out12, dpi=120, bbox_inches="tight")
print("saved ->", out12)
