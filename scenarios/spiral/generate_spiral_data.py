"""Generate dataset from spiral-tracking flights (for the observer)."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import solve_ivp
from core.generate_quadrotor_data import quadrotor_dynamics
from simulate_spiral import  full_control_spiral
from spiral_reference import spiral_reference, OMEGA


def generate_spiral_dataset(N=50, dt=0.01, seed=0):
    t_end = 30.0          ## full spiral flight (3 loops, matches simulate_spiral)
    t_eval = np.arange(0.0, t_end, dt)
    rng = np.random.default_rng(seed)

    all_traj, all_u = [], []
    for i in range(N):
        x0= np.zeros(12)
        x0[0] = 4.0 + rng.uniform(-1.0, 1.0)      # x near 4
        x0[1] = 5.0 + rng.uniform(-1.0, 1.0)      # y near 5
        x0[2] = 0.0 + rng.uniform(-0.5, 0.5)      # z near 0 (ground start, climbs)
        x0[3:6] = 0.0                              # at rest
        x0[6] = 0.0 + rng.normal(0, 0.02)         # zero initial angles + scatter
        x0[7] = 0.0 + rng.normal(0, 0.02)
        x0[8] = 0.0 + rng.normal(0, 0.02)
        sol = solve_ivp(
            lambda t, x: quadrotor_dynamics(t, x, full_control_spiral(t, x)),
            (0.0, t_end), x0, t_eval=t_eval, max_step=0.02)
        traj = sol.y.T
        u_traj = np.array([full_control_spiral(t_eval[k], traj[k]) for k in range(traj.shape[0])])
        all_traj.append(traj)
        all_u.append(u_traj)

    return t_eval, np.array(all_traj), np.array(all_u)


if __name__ == "__main__":
    T, X, U = generate_spiral_dataset(N=50)
    print("States shape:", X.shape)
    print("Control shape:", U.shape)
    print(f"x range: [{X[:,:,0].min():.1f}, {X[:,:,0].max():.1f}]  (spiral radius ~10)")
    print(f"z range: [{X[:,:,2].min():.1f}, {X[:,:,2].max():.1f}]  (climbs 0 to ~60)")
    print(f"yaw mean: {X[:,:,8].mean():.4f}  (should settle ~0.785)")
    np.savez("datasets/spiral_dataset.npz", T=T, X=X, U=U)