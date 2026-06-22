"""Generate dataset from the Singha Fig 2 scenario (all states excited, incl. yaw)."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import solve_ivp

from core.generate_quadrotor_data import quadrotor_dynamics
from core.controller import (M, G, KP_Z, KD_Z, KP_ATT, KD_ATT,
                             IX, IY, IZ, L_ARM, outer_loop)

KP_YAW, KD_YAW = 2.0, 2.0
X_TGT, Y_TGT, Z_TGT = 0.6, 0.6, 10.0


def full_control_singha(t, x):
    psi_d, psi_d_dot = 0.1*np.sin(t), 0.1*np.cos(t)
    z, zdot = x[2], x[5]
    U1 = M*G + KP_Z*(Z_TGT - z) - KD_Z*zdot
    phi_d, theta_d = outer_loop(x, U1, X_TGT, Y_TGT)
    phi, theta, psi = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    U2 = (IX/L_ARM)*(KP_ATT*(phi_d-phi) - KD_ATT*p - q*r*(IY-IZ)/IX)
    U3 = (IY/L_ARM)*(KP_ATT*(theta_d-theta) - KD_ATT*q - p*r*(IZ-IX)/IY)
    U4 = IZ*(KP_YAW*(psi_d-psi) + KD_YAW*(psi_d_dot-r) - p*q*(IX-IY)/IZ)
    return np.array([U1, U2, U3, U4])


def generate_singha_dataset(N=50, t_end=18.0, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t_eval = np.arange(0.0, t_end, dt)

    all_traj, all_u = [], []
    for i in range(N):
        x0 = np.zeros(12)
        # scatter around Singha's start (6, 0.2, 0) for variety
        x0[0] = 6.0 + rng.uniform(-1.0, 1.0)
        x0[1] = 0.2 + rng.uniform(-0.5, 0.5)
        x0[2] = 0.0 + rng.uniform(0.0, 1.0)
        x0[6] = rng.normal(0, 0.05)        # small initial tilt
        x0[7] = rng.normal(0, 0.05)

        sol = solve_ivp(
            lambda t, x: quadrotor_dynamics(t, x, full_control_singha(t, x)),
            (0.0, t_end), x0, t_eval=t_eval, max_step=0.02,
        )
        traj = sol.y.T
        # recompute control at each saved step (with its time t)
        u_traj = np.array([full_control_singha(t_eval[k], traj[k]) for k in range(traj.shape[0])])
        all_traj.append(traj)
        all_u.append(u_traj)

    X = np.array(all_traj)
    U = np.array(all_u)
    return t_eval, X, U


if __name__ == "__main__":
    T, X, U = generate_singha_dataset(N=50, t_end=18.0, dt=0.01, seed=0)
    print("States shape: ", X.shape)
    print("Control shape:", U.shape)
    final = X[:, -1, :3]
    print(f"Mean final: x={final[:,0].mean():.3f}, y={final[:,1].mean():.3f}, z={final[:,2].mean():.3f}")
    print(f"  (target: {X_TGT}, {Y_TGT}, {Z_TGT})")
    # confirm yaw is actually moving now
    yaw_range = X[:, :, 8].std()
    print(f"Yaw std across dataset: {yaw_range:.4f}  (should be NONzero now!)")
    np.savez("datasets/singha_dataset.npz", T=T, X=X, U=U)
    print("Saved -> data/singha_dataset.npz")