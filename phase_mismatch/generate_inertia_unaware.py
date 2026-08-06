"""Inertia mismatch, CONTROLLER-UNAWARE variant.
True dynamics use the PERTURBED inertia; the controller keeps the NOMINAL
inertia (it does not know the drone changed). This breaks the cancellation,
so the inertia change actually alters the flight and the observer's
sensitivity becomes visible.
Reuses the validated dynamics + controller functions, but passes them
DIFFERENT parameter dicts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.integrate import solve_ivp
from phase_mismatch.generate_perturbed import (
    quadrotor_dynamics, full_control_spiral, NOMINAL)

def generate_set_unaware(P_true, P_ctrl, init_conditions, dt=0.01, t_end=30.0):
    """Dynamics use P_true; controller uses P_ctrl (different dicts)."""
    t_eval = np.arange(0.0, t_end, dt)
    trajs, us = [], []
    for x0 in init_conditions:
        sol = solve_ivp(
            lambda t, x: quadrotor_dynamics(t, x, full_control_spiral(t, x, P_ctrl), P_true),
            (0.0, t_end), x0, t_eval=t_eval, max_step=0.02)
        traj = sol.y.T
        u_traj = np.array([full_control_spiral(t_eval[k], traj[k], P_ctrl)
                           for k in range(traj.shape[0])])
        trajs.append(traj); us.append(u_traj)
    return t_eval, np.array(trajs), np.array(us)

base = np.load("datasets/spiral_dataset.npz")
X = base["X"]
test_x0 = [X[i, 0, :] for i in range(40, 50)]

os.makedirs("datasets/mismatch", exist_ok=True)
deviations = [-0.20, -0.10, 0.0, 0.10, 0.20]
for dev in deviations:
    P_true = dict(NOMINAL)                       # true drone: perturbed inertia
    P_true["Ix"] = NOMINAL["Ix"] * (1 + dev)
    P_true["Iy"] = NOMINAL["Iy"] * (1 + dev)
    P_true["Iz"] = NOMINAL["Iz"] * (1 + dev)
    P_ctrl = dict(NOMINAL)                        # controller: stays nominal
    tag = "inertia_unaware_%+d" % int(round(dev*100))
    if dev == 0.0:
        tag = "inertia_unaware_0"
    T, Xp, Up = generate_set_unaware(P_true, P_ctrl, test_x0)
    np.savez(f"datasets/mismatch/{tag}.npz", T=T, X=Xp, U=Up,
             Ix_true=P_true["Ix"], dev=dev)
    print("inertia(unaware) %+.0f%% (true Ix=%.4f, ctrl Ix=%.4f) -> %s  %s" %
          (dev*100, P_true["Ix"], P_ctrl["Ix"], tag, str(Xp.shape)), flush=True)
print("done", flush=True)