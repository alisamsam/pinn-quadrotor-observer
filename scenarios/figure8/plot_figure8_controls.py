"""Plot U1-U4 control inputs over the figure-8 flight (validation vs Singha)."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from core.generate_quadrotor_data import quadrotor_dynamics
from simulate_figure8 import full_control_figure8
from figure8_reference import figure8_reference, OMEGA

# simulate one clean flight (Table 5 start: 0,0,0)
x0 = np.zeros(12)
x0[0], x0[1], x0[2] = 0.0, 0.0, 0.0
x0[6], x0[7], x0[8] = 0.087, 0.1042, 0.209

t_end = 2*np.pi/OMEGA
t_eval = np.arange(0.0, t_end, 0.01)
sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, full_control_figure8(t, x)),
                (0.0, t_end), x0, t_eval=t_eval, max_step=0.02)

# recompute U at each step
U = np.array([full_control_figure8(t_eval[k], sol.y[:,k]) for k in range(len(t_eval))])

# plot the four control inputs
fig, ax = plt.subplots(2, 2, figsize=(13, 7))
labels = ["U1 (thrust) [N]", "U2 (roll torque) [N·m]", "U3 (pitch torque) [N·m]", "U4 (yaw torque) [N·m]"]
for i, a in enumerate(ax.flat):
    a.plot(t_eval, U[:, i], 'b-', lw=1.4)
    a.set_title(labels[i]); a.set_xlabel("t (s)"); a.grid(alpha=0.3)
ax[0,0].axhline(17.658, color='r', ls='--', alpha=0.5, label='hover (mg)')
ax[0,0].legend()
fig.suptitle("Figure8 trajectory — control inputs U1–U4", fontsize=14)
fig.tight_layout()
os.makedirs("docs", exist_ok=True)
plt.savefig("docs/figure8_controls.png", dpi=120, bbox_inches="tight")
print("Saved -> docs/figure8_controls.png")
print(f"U1 range: [{U[:,0].min():.2f}, {U[:,0].max():.2f}]  (hover ~17.66 N)")
print(f"U2 range: [{U[:,1].min():.3f}, {U[:,1].max():.3f}]")
print(f"U3 range: [{U[:,2].min():.3f}, {U[:,2].max():.3f}]")
print(f"U4 range: [{U[:,3].min():.3f}, {U[:,3].max():.3f}]")