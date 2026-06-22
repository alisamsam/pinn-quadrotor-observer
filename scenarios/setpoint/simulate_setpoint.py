"""Phase 2 Step 2.3-2.4: closed-loop altitude-hold simulation."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from data.generate_quadrotor_data import quadrotor_dynamics
from data.controller import altitude_pd_control, Z_TARGET, M, G

# ----- simulate -----
t_end = 8.0
t_eval = np.arange(0.0, t_end, 0.01)
x0 = np.zeros(12)                       # start at rest, z=0 (below target)

sol = solve_ivp(
    lambda t, x: quadrotor_dynamics(t, x, altitude_pd_control(x)),
    (0.0, t_end), x0, t_eval=t_eval,
)

z = sol.y[2]                            # altitude over time

# ----- 2.4 sanity checks -----
print("Start z:      ", round(z[0], 3))
print("Final z:      ", round(z[-1], 3), " (target =", Z_TARGET, ")")
print("Reached target?", abs(z[-1] - Z_TARGET) < 0.05)
print("Max overshoot: ", round(max(z) - Z_TARGET, 3) if max(z) > Z_TARGET else 0.0)

# ----- plot -----
plt.figure(figsize=(9, 5))
plt.plot(sol.t, z, 'b-', linewidth=2, label="altitude z(t)")
plt.axhline(Z_TARGET, color='g', linestyle='--', label=f"target = {Z_TARGET} m")
plt.xlabel("Time (s)")
plt.ylabel("z (m)")
plt.title("Closed-loop altitude hold (PD controller)")
plt.legend(); plt.grid(True, alpha=0.3)
os.makedirs("docs", exist_ok=True)
plt.savefig("docs/setpoint_altitude.png", dpi=120, bbox_inches="tight")
print("Saved -> docs/setpoint_altitude.png")