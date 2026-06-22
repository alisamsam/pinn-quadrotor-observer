"""Phase 2 Step 2.5: generate controlled (setpoint-hold) dataset."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import solve_ivp

from core.generate_quadrotor_data import quadrotor_dynamics
from core.controller import altitude_pd_control, Z_TARGET


def generate_setpoint_dataset(N=50, t_end=8.0, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t_eval = np.arange(0.0, t_end, dt)

    all_traj = []
    all_u = []                          # we ALSO save the control input now
    for i in range(N):
        # start near the ground, slightly perturbed
        x0 = np.zeros(12)
        x0[:3] += rng.normal(0, 0.1, 3)        # small position scatter
        x0[6:9] += rng.normal(0, 0.02, 3)      # tiny initial tilt

        sol = solve_ivp(
            lambda t, x: quadrotor_dynamics(t, x, altitude_pd_control(x)),
            (0.0, t_end), x0, t_eval=t_eval,
        )
        traj = sol.y.T                         # (n_steps, 12)

        # recompute the control input at each saved step (for the dataset)
        u_traj = np.array([altitude_pd_control(traj[k]) for k in range(traj.shape[0])])

        all_traj.append(traj)
        all_u.append(u_traj)

    X = np.array(all_traj)              # (N, n_steps, 12)
    U = np.array(all_u)                 # (N, n_steps, 4)
    return t_eval, X, U


if __name__ == "__main__":
    T, X, U = generate_setpoint_dataset(N=50, t_end=8.0, dt=0.01, seed=0)
    print("Time shape:   ", T.shape)         # (800,)
    print("States shape: ", X.shape)         # (50, 800, 12)
    print("Control shape:", U.shape)         # (50, 800, 4)

    # quick check: did trajectories reach the target on average?
    final_z = X[:, -1, 2]
    print(f"Mean final z: {final_z.mean():.3f}  (target {Z_TARGET})")

    np.savez("datasets/setpoint_dataset.npz", T=T, X=X, U=U)
    print("Saved -> data/setpoint_dataset.npz")