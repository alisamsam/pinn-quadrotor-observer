"""Actual vs desired spiral using core/controller_v2 (defense figure)."""
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
from core.controller_v2 import full_control_v2
from spiral_reference import spiral_reference, yaw_reference

SKIP = 5.0
x0 = np.zeros(12); x0[0], x0[1], x0[2] = 4.0, 5.0, 0.0
t_eval = np.arange(0.0, 30.0, 0.01)
ctrl = lambda t, x: full_control_v2(t, x, spiral_reference, yaw_reference)
sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, ctrl(t, x)),
                (0.0, 30.0), x0, t_eval=t_eval, max_step=0.02)
t = sol.t
xa, ya, za = sol.y[0], sol.y[1], sol.y[2]
P = np.array([spiral_reference(tt)[0] for tt in t])
m = t >= SKIP
rmse = np.sqrt(((sol.y[:3, m].T - P[m])**2).mean(axis=0))


fig = plt.figure(figsize=(14, 9))

ax3d = fig.add_subplot(2, 2, 1, projection='3d')
ax3d.plot(P[:, 0], P[:, 1], P[:, 2], 'b-', lw=2.2, label='desired')
ax3d.plot(xa, ya, za, 'r--', lw=1.8, label='actual')
ax3d.scatter([x0[0]], [x0[1]], [x0[2]], c='k', s=40, label='start (4,5,0)')
ax3d.set_xlabel('x (m)'); ax3d.set_ylabel('y (m)'); ax3d.set_zlabel('z (m)')
ax3d.set_title('3D trajectory'); ax3d.legend(fontsize=8); ax3d.view_init(elev=18, azim=-60)

for (idx, nm, ax) in [(0, 'x', fig.add_subplot(2, 2, 2)),
                      (1, 'y', fig.add_subplot(2, 2, 3)),
                      (2, 'z', fig.add_subplot(2, 2, 4))]:
    ax.plot(t, P[:, idx], 'b-', lw=2, label='desired')
    ax.plot(t, sol.y[idx], 'r--', lw=1.5, label='actual')
    ax.axvspan(0, SKIP, color='gray', alpha=0.15)
    ax.set_xlabel('t (s)'); ax.set_ylabel(f'{nm} (m)')
    ax.set_title(f'{nm}:  steady RMSE = {rmse[idx]:.3f} m'); ax.grid(alpha=0.3); ax.legend(fontsize=8)

fig.suptitle('Spiral tracking with controller_v2  (R=10, omega=0.6)  |  '
             f'overall steady RMSE = {np.linalg.norm(rmse):.3f} m  (gray = first {SKIP:.0f}s skipped)',
             fontsize=12)
out = os.path.join(PROJ, "docs", "spiral_actual_vs_desired_v2.png")
plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(out, dpi=120, bbox_inches="tight")
print("saved ->", out)
print("steady RMSE x=%.3f y=%.3f z=%.3f overall=%.3f" % (rmse[0], rmse[1], rmse[2], np.linalg.norm(rmse)))
