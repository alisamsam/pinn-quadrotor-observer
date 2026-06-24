"""Generate dataset from circle-tracking flights (for the observer)."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import solve_ivp
from core.generate_quadrotor_data import quadrotor_dynamics
from simulate_circle import full_control_circle
from circle_reference import circular_reference, OMEGA


def generate_circle_dataset(N=50, dt=0.01, seed=0):
    t_end = 2*np.pi/OMEGA          # one full loop
    t_eval = np.arange(0.0, t_end, dt)
    rng = np.random.default_rng(seed)

    all_traj, all_u = [], []
    for i in range(N):
        # start near the circle's t=0 point, with small random scatter
        pos0, vel0, _ = circular_reference(0.0)
        x0 = np.zeros(12)
        x0[0:3] = pos0 + rng.uniform(-0.5, 0.5, 3)   # small position scatter
        x0[3:6] = vel0                                # match circle velocity
        x0[6] = rng.normal(0, 0.05)                   # small tilt scatter
        x0[7] = rng.normal(0, 0.05)

        sol = solve_ivp(
            lambda t, x: quadrotor_dynamics(t, x, full_control_circle(t, x)),
            (0.0, t_end), x0, t_eval=t_eval, max_step=0.02)
        traj = sol.y.T
        u_traj = np.array([full_control_circle(t_eval[k], traj[k]) for k in range(traj.shape[0])])
        all_traj.append(traj)
        all_u.append(u_traj)

    return t_eval, np.array(all_traj), np.array(all_u)


if __name__ == "__main__":
    T, X, U = generate_circle_dataset(N=50)
    print("States shape:", X.shape)
    print("Control shape:", U.shape)
    print(f"x range: [{X[:,:,0].min():.1f}, {X[:,:,0].max():.1f}]  (circle radius ~10)")
    print(f"z mean: {X[:,:,2].mean():.2f}  (should be ~2)")
    print(f"yaw std: {X[:,:,8].std():.4f}  (excited by 0.2 sin)")
    np.savez("datasets/circle_dataset.npz", T=T, X=X, U=U)
    print("Saved -> datasets/circle_dataset.npz")