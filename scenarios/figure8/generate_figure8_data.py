"""Generate dataset from figure-8 tracking flights (for the observer)."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import solve_ivp
from core.generate_quadrotor_data import quadrotor_dynamics
from simulate_figure8 import full_control_figure8
from figure8_reference import figure8_reference, OMEGA


def generate_figure8_dataset(N=50, dt=0.01, seed=0):
    t_end = 2*np.pi/OMEGA          # one full loop
    t_eval = np.arange(0.0, t_end, dt)
    rng = np.random.default_rng(seed)

    all_traj, all_u = [], []
    for i in range(N):
        # start near the figure-8's t=0 point, with small random scatter
        x0 = np.zeros(12)
        # Table 5 start (0,0,0) with small scatter for dataset variety  
        x0[0] = 0.0 + rng.uniform(-1.0, 1.0)      # x near 0
        x0[1] = 0.0 + rng.uniform(-1.0, 1.0)      # y near 0
        x0[2] = 0.0 + rng.uniform(0.0, 1.0)      # z near 0 (climb to figure8)
        x0[3:6] = 0.0                              # at rest (drone placed, not moving)
        # Table 5 initial angles, with small scatter
        x0[6] = 0.087  + rng.normal(0, 0.02)
        x0[7] = 0.1042 + rng.normal(0, 0.02)
        x0[8] = 0.209  + rng.normal(0, 0.02)

        sol = solve_ivp(
            lambda t, x: quadrotor_dynamics(t, x, full_control_figure8(t, x)),
            (0.0, t_end), x0, t_eval=t_eval, max_step=0.02)
        traj = sol.y.T
        u_traj = np.array([full_control_figure8(t_eval[k], traj[k]) for k in range(traj.shape[0])])
        all_traj.append(traj)
        all_u.append(u_traj)

    return t_eval, np.array(all_traj), np.array(all_u)


if __name__ == "__main__":
    T, X, U = generate_figure8_dataset(N=50)
    print("States shape:", X.shape)
    print("Control shape:", U.shape)
    print(f"x range: [{X[:,:,0].min():.1f}, {X[:,:,0].max():.1f}]  (figure-8, ~±10)")
    print(f"z mean: {X[:,:,2].mean():.2f}  (figure-8 z bobs 0.5-2.5, mean ~1.5)")
    print(f"z range: [{X[:,:,2].min():.1f}, {X[:,:,2].max():.1f}]  (should span ~0.5 to 2.5)")
    print(f"yaw mean: {X[:,:,8].mean():.3f}  (constant ~0.785)")
    np.savez("datasets/figure8_dataset.npz", T=T, X=X, U=U)
    print("Saved -> datasets/figure8_dataset.npz")