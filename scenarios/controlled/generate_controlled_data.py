"""B.6: rich controlled dataset using the FULL controller."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import solve_ivp

from core.generate_quadrotor_data import quadrotor_dynamics
from core.controller import full_control, X_TARGET, Y_TARGET, Z_TARGET


def generate_controlled_dataset(N=50, t_end=12.0, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t_eval = np.arange(0.0, t_end, dt)

    all_traj, all_u = [], []
    for i in range(N):
        x0 = np.zeros(12)
        # vary the START: off-target position + small tilt (so each flight differs)
        x0[0] = rng.uniform(-1.0, 1.0)      # x offset
        x0[1] = rng.uniform(-1.0, 1.0)      # y offset
        x0[2] = rng.uniform(0.0, 0.5)       # start low
        x0[6] = rng.normal(0, 0.1)          # initial roll tilt
        x0[7] = rng.normal(0, 0.1)          # initial pitch tilt

        sol = solve_ivp(
            lambda t, x: quadrotor_dynamics(t, x, full_control(x)),
            (0.0, t_end), x0, t_eval=t_eval,
        )
        traj = sol.y.T
        u_traj = np.array([full_control(traj[k]) for k in range(traj.shape[0])])
        all_traj.append(traj)
        all_u.append(u_traj)

    X = np.array(all_traj)
    U = np.array(all_u)
    return t_eval, X, U


if __name__ == "__main__":
    T, X, U = generate_controlled_dataset(N=50, t_end=12.0, dt=0.01, seed=0)
    print("States shape: ", X.shape)         # (50, 1200, 12)
    print("Control shape:", U.shape)          # (50, 1200, 4)

    # check: did they all converge to the setpoint?
    final = X[:, -1, :3]                       # final x,y,z of each
    print(f"Mean final x: {final[:,0].mean():.3f}  (target {X_TARGET})")
    print(f"Mean final y: {final[:,1].mean():.3f}  (target {Y_TARGET})")
    print(f"Mean final z: {final[:,2].mean():.3f}  (target {Z_TARGET})")

    np.savez("datasets/controlled_dataset.npz", T=T, X=X, U=U)
    print("Saved -> data/controlled_dataset.npz")