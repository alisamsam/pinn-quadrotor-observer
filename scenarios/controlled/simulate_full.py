"""B.5: test the full controller — does the drone hold a 3D setpoint?"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from core.generate_quadrotor_data import quadrotor_dynamics
from core.controller import full_control, X_TARGET, Y_TARGET, Z_TARGET

# start OFF-target AND tilted -- the hard case that drifted before
x0 = np.zeros(12)
x0[0] = 1.0       # 1m off in x
x0[1] = -0.5      # 0.5m off in y
x0[6] = 0.1       # initial roll tilt
x0[7] = 0.1       # initial pitch tilt

t_end = 12.0
t_eval = np.arange(0.0, t_end, 0.01)

sol = solve_ivp(
    lambda t, x: quadrotor_dynamics(t, x, full_control(x)),
    (0.0, t_end), x0, t_eval=t_eval,
)

x, y, z = sol.y[0], sol.y[1], sol.y[2]

# checks
print(f"Start:  x={x[0]:.2f}, y={y[0]:.2f}, z={z[0]:.2f}")
print(f"Final:  x={x[-1]:.3f}, y={y[-1]:.3f}, z={z[-1]:.3f}")
print(f"Target: x={X_TARGET}, y={Y_TARGET}, z={Z_TARGET}")
reached = abs(x[-1])<0.1 and abs(y[-1])<0.1 and abs(z[-1]-Z_TARGET)<0.1
print("Held the 3D setpoint?", reached)

# plot x, y, z over time
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, (d, name, tgt) in enumerate([(x,"x",X_TARGET),(y,"y",Y_TARGET),(z,"z",Z_TARGET)]):
    ax[i].plot(sol.t, d, 'b-', lw=2)
    ax[i].axhline(tgt, color='r', ls='--', label='target')
    ax[i].set_title(f"{name} (m)"); ax[i].set_xlabel("t (s)"); ax[i].grid(alpha=0.3); ax[i].legend()
fig.suptitle("B.5: full controller — position vs time")
fig.tight_layout()
os.makedirs("docs", exist_ok=True)
plt.savefig("docs/full_control_test.png", dpi=120, bbox_inches="tight")
print("Saved -> docs/full_control_test.png")