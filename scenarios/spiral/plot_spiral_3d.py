"""3D flight path: actual circle-tracking vs desired circle."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from core.generate_quadrotor_data import quadrotor_dynamics
from simulate_circle import full_control_circle
from circle_reference import circular_reference, OMEGA

# simulate one clean flight (start on the circle)

x0 = np.zeros(12)
x0[0], x0[1], x0[2] = 0.0, 0.0, 4.0  # Table 5 initial position (0,0,4) 
x0[6], x0[7], x0[8] = 0.087, 0.1042, 0.209 # Table 5 initial angles (roll, pitch, yaw)    
 
t_end = 2*np.pi/OMEGA            # one full loop
t_eval = np.arange(0.0, t_end, 0.01)
sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, full_control_circle(t, x)),
                (0.0, t_end), x0, t_eval=t_eval, max_step=0.02)
x, y, z = sol.y[0], sol.y[1], sol.y[2]

# desired circle
P = np.array([circular_reference(t)[0] for t in t_eval])

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(P[:,0], P[:,1], P[:,2], 'r--', lw=1.5, label="desired circle")
ax.plot(x, y, z, 'b-', lw=1.6, label="actual flight")
ax.scatter(x[0], y[0], z[0], color='black', s=90, marker='o', label="start")
ax.scatter(x[-1], y[-1], z[-1], color='green', s=120, marker='*', label="end")

ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
ax.set_title("Circle trajectory tracking (R=10, ω=0.3) — actual vs desired")
ax.set_xticks(np.arange(-10,11,5)); ax.set_yticks(np.arange(-10,11,5)); ax.set_zticks(np.arange(0,5,1))
ax.legend(); ax.view_init(elev=18, azim=-72)

os.makedirs("docs", exist_ok=True)
plt.savefig("docs/circle_flight_3d.png", dpi=120, bbox_inches="tight")
print("Saved -> docs/circle_flight_3d.png")