"""Compare controller_v2 vs controller_v3 on the circle: tracking RMSE + v3 figures."""
import sys, os
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJ); sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from core.generate_quadrotor_data import quadrotor_dynamics
from core.controller_v2 import full_control_v2
from core.controller_v3 import full_control_v3
from circle_reference import circular_reference, yaw_reference

t_eval = np.arange(0.0, 30.0, 0.01)
x0 = np.zeros(12); x0[0], x0[1], x0[2] = 4.0, 5.0, 0.0
pos_d = np.array([circular_reference(tt)[0] for tt in t_eval])
vel_d = np.array([circular_reference(tt)[1] for tt in t_eval])

def run(ctrl_fn):
    ctrl = lambda t, x: ctrl_fn(t, x, circular_reference, yaw_reference)
    sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, ctrl(t, x)),
                    (0.0, 30.0), x0, t_eval=t_eval, max_step=0.02)
    return sol.y  # (12, N)

def rmse_after(Y, t0=3.0):
    """tracking RMSE ignoring the first t0 seconds (start-up transient off-path)."""
    m = t_eval >= t0
    pos = np.sqrt(((Y[0:3, m].T - pos_d[m]) ** 2).mean())
    vel = np.sqrt(((Y[3:6, m].T - vel_d[m]) ** 2).mean())
    return pos, vel

for name, fn in [("controller_v2", full_control_v2), ("controller_v3", full_control_v3)]:
    Y = run(fn)
    p_all = np.sqrt(((Y[0:3].T - pos_d) ** 2).mean()); v_all = np.sqrt(((Y[3:6].T - vel_d) ** 2).mean())
    p3, v3 = rmse_after(Y, 3.0)
    print(f"{name}: pos RMSE full {p_all:.4f} m | vel RMSE full {v_all:.4f} m/s | "
          f"(after 3s) pos {p3:.4f} | vel {v3:.4f}", flush=True)
    if name == "controller_v3":
        Yv3 = Y

# --- v3 figures (12 states + 3D) ---
t = t_eval; Y = Yv3
psi_d = np.array([yaw_reference(tt)[0] for tt in t])
labels = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
units = ['m','m','m','m/s','m/s','m/s','rad','rad','rad','rad/s','rad/s','rad/s']
desired = [pos_d[:,0],pos_d[:,1],pos_d[:,2],vel_d[:,0],vel_d[:,1],vel_d[:,2],
           None,None,psi_d,None,None,None]
fig, axes = plt.subplots(4, 3, figsize=(15, 11))
for i, ax in enumerate(axes.flat):
    ax.plot(t, Y[i], 'r-', lw=1.3, label='actual')
    if desired[i] is not None: ax.plot(t, desired[i], 'b--', lw=1.6, label='desired')
    ax.set_title(labels[i]); ax.grid(alpha=0.3); ax.set_xlabel('t (s)')
    ax.set_ylabel(f'{labels[i]} ({units[i]})'); ax.legend(fontsize=7)
fig.suptitle('controller_v3 on the circle: all 12 states (actual vs desired)')
plt.tight_layout(rect=[0,0,1,0.98]); plt.savefig(os.path.join(PROJ,"docs","circle_12states_v3.png"), dpi=120, bbox_inches="tight")
print("saved -> docs/circle_12states_v3.png", flush=True)

fig = plt.figure(figsize=(8,7)); ax = fig.add_subplot(111, projection='3d')
ax.plot(pos_d[:,0],pos_d[:,1],pos_d[:,2],'b-',lw=2.3,label='desired')
ax.plot(Y[0],Y[1],Y[2],'r--',lw=1.8,label='actual')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
ax.set_title('Circle tracking (controller_v3)'); ax.legend(); ax.view_init(elev=18, azim=-60)
plt.tight_layout(); plt.savefig(os.path.join(PROJ,"docs","circle_3d_v3.png"), dpi=120, bbox_inches="tight")
print("saved -> docs/circle_3d_v3.png", flush=True)
