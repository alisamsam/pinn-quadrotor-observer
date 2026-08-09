"""Verify core/controller_v2 reproduces the v5 spiral result (~0.18 m steady)."""
import sys, os
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJ)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.integrate import solve_ivp
from core.generate_quadrotor_data import quadrotor_dynamics
from core.controller_v2 import full_control_v2
from spiral_reference import spiral_reference, yaw_reference

SKIP = 5.0
x0 = np.zeros(12); x0[0], x0[1], x0[2] = 4.0, 5.0, 0.0
t_eval = np.arange(0.0, 30.0, 0.01)
ctrl = lambda t, x: full_control_v2(t, x, spiral_reference, yaw_reference)
sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, ctrl(t, x)),
                (0.0, 30.0), x0, t_eval=t_eval, max_step=0.02)
P = np.array([spiral_reference(t)[0] for t in sol.t])
m = sol.t >= SKIP
rmse = np.sqrt(((sol.y[:3, m].T - P[m])**2).mean(axis=0))
print("steady RMSE  x=%.3f y=%.3f z=%.3f  overall=%.3f m"
      % (rmse[0], rmse[1], rmse[2], float(np.linalg.norm(rmse))))
print("expected (v5 best): x~0.148 y~0.103 z~0.000  overall~0.180 m")
