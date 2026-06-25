"""3D flight path: actual Figure8-tracking vs desired Figure8."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from core.generate_quadrotor_data import quadrotor_dynamics
from simulate_figure8 import full_control_figure8
from figure8_reference import figure8_reference, OMEGA

# simulate one clean flight (start on the figure8)

x0 = np.zeros(12)
x0[0], x0[1], x0[2] = 0.0, 0.0, 0.0  # Table 5 initial position (0,0,0) 
x0[6], x0[7], x0[8] = 0.087, 0.1042, 0.209 # Table 5 initial angles (roll, pitch, yaw)    
 
t_end = 2*np.pi/OMEGA            # one full loop
t_eval = np.arange(0.0, t_end, 0.01)
sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, full_control_figure8(t, x)),
                (0.0, t_end), x0, t_eval=t_eval, max_step=0.02)
x, y, z = sol.y[0], sol.y[1], sol.y[2]

# desired figure8
P = np.array([figure8_reference(t)[0] for t in t_eval])

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(P[:,0], P[:,1], P[:,2], 'r--', lw=1.5, label="desired figure8")
ax.plot(x, y, z, 'b-', lw=1.6, label="actual flight")
ax.scatter(x[0], y[0], z[0], color='black', s=90, marker='o', label="start")
ax.scatter(x[-1], y[-1], z[-1], color='green', s=120, marker='*', label="end")

ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
ax.set_title("Figure8 trajectory tracking (R=10, ω=0.3) — actual vs desired")
ax.set_xticks(np.arange(-10,11,5)); ax.set_yticks(np.arange(-10,11,5)); ax.set_zticks(np.arange(0,5,1))
ax.legend(); ax.view_init(elev=18, azim=-72)

os.makedirs("docs", exist_ok=True)
plt.savefig("docs/figure8_flight_3d.png", dpi=120, bbox_inches="tight")
print("Saved -> docs/figure8_flight_3d.png")