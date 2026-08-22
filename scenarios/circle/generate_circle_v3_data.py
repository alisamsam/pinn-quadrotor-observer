"""Generate observer dataset from controller_v3 CIRCLE flights.
Same 50-flight IC scatter as the v2 generators; only the controller changes
(controller_v3, principled gains). Output: datasets/circle_v3_dataset.npz.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.integrate import solve_ivp
from core.generate_quadrotor_data import quadrotor_dynamics
from core.controller_v3 import full_control_v3
from circle_reference import circular_reference, yaw_reference


def ctrl(t, x):
    return full_control_v3(t, x, circular_reference, yaw_reference)


def generate(N=50, dt=0.01, t_end=30.0, seed=0):
    t_eval = np.arange(0.0, t_end, dt)
    rng = np.random.default_rng(seed)
    all_traj, all_u = [], []
    for i in range(N):
        x0 = np.zeros(12)
        x0[0] = 4.0 + rng.uniform(-1.0, 1.0)
        x0[1] = 5.0 + rng.uniform(-1.0, 1.0)
        x0[2] = 0.0 + rng.uniform(-0.5, 0.5)
        x0[6] = rng.normal(0, 0.02); x0[7] = rng.normal(0, 0.02); x0[8] = rng.normal(0, 0.02)
        sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, ctrl(t, x)),
                        (0.0, t_end), x0, t_eval=t_eval, max_step=0.02)
        traj = sol.y.T
        u_traj = np.array([ctrl(t_eval[k], traj[k]) for k in range(traj.shape[0])])
        all_traj.append(traj); all_u.append(u_traj)
    return t_eval, np.array(all_traj), np.array(all_u)


if __name__ == "__main__":
    T, X, U = generate(N=50)
    print("States shape:", X.shape, " Control shape:", U.shape)
    print(f"x range [{X[:,:,0].min():.1f}, {X[:,:,0].max():.1f}]  z range [{X[:,:,2].min():.1f}, {X[:,:,2].max():.1f}]")
    os.makedirs("datasets", exist_ok=True)
    np.savez("datasets/circle_v3_dataset.npz", T=T, X=X, U=U)
    print("saved -> datasets/circle_v3_dataset.npz")
