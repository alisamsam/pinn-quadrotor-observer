"""Phase 2 Step C: clean controlled dataset (zero initial tilt, no drift)."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import solve_ivp

from core.generate_quadrotor_data import quadrotor_dynamics
from core.controller import altitude_pd_control, Z_TARGET


def generate_clean_dataset(N=50, t_end=8.0, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t_eval = np.arange(0.0, t_end, dt)

    all_traj, all_u = [], []
    for i in range(N):
        x0 = np.zeros(12)
        # vary ONLY the starting height (and a little x,y position) -- NO tilt
        x0[2] = rng.uniform(0.0, 0.5)          # start somewhere low, random
        x0[0] = rng.normal(0, 0.05)            # tiny x,y position scatter
        x0[1] = rng.normal(0, 0.05)
        # angles (indices 6,7,8) stay exactly ZERO -> no tilt -> no drift

        sol = solve_ivp(
            lambda t, x: quadrotor_dynamics(t, x, altitude_pd_control(x)),
            (0.0, t_end), x0, t_eval=t_eval,
        )
        traj = sol.y.T
        u_traj = np.array([altitude_pd_control(traj[k]) for k in range(traj.shape[0])])
        all_traj.append(traj)
        all_u.append(u_traj)

    X = np.array(all_traj)
    U = np.array(all_u)
    return t_eval, X, U


if __name__ == "__main__":
    T, X, U = generate_clean_dataset(N=50, t_end=8.0, dt=0.01, seed=0)
    print("States shape: ", X.shape)
    print("Control shape:", U.shape)

    # checks: final altitude near target, AND no horizontal drift
    final_z = X[:, -1, 2]
    final_x = X[:, -1, 0]
    print(f"Mean final z: {final_z.mean():.3f}  (target {Z_TARGET})")
    print(f"Max |final x|: {np.abs(final_x).max():.4f}  (should be tiny ~0)")

    np.savez("datasets/setpoint_clean.npz", T=T, X=X, U=U)
    print("Saved -> data/setpoint_clean.npz")