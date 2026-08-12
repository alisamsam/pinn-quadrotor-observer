"""Generate dataset using the PROFESSOR's controller + his matched dynamics.
Saved separately as spiral_prof_dataset.npz — does NOT touch your validated data."""
import os
import numpy as np
from scipy.integrate import solve_ivp

# ---------- physical parameters (his) ----------
m, g, l = 1.80, 9.81, 0.20
Ix, Iy, Iz = 0.03, 0.03, 0.04

# ---------- gains (his) ----------
Kpx, Kdx = 3.0, 5.0
Kpy, Kdy = 3.0, 5.0
Kpz, Kdz = 6.0, 6.0
Kp_phi, Kd_phi = 25.0, 8.0
Kp_theta, Kd_theta = 25.0, 8.0
Kp_psi, Kd_psi = 8.0, 4.0

def sat(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

# ---------- his helical reference ----------
R_REF, W_REF, VZ_REF = 8.0, 0.35, 1.2
def desired_trajectory(t):
    xd = R_REF*np.cos(W_REF*t); yd = R_REF*np.sin(W_REF*t); zd = VZ_REF*t
    xd_d = -R_REF*W_REF*np.sin(W_REF*t); yd_d = R_REF*W_REF*np.cos(W_REF*t); zd_d = VZ_REF
    xd_dd = -R_REF*W_REF**2*np.cos(W_REF*t); yd_dd = -R_REF*W_REF**2*np.sin(W_REF*t); zd_dd = 0.0
    psi_d = W_REF*t; psi_d_d = W_REF; psi_d_dd = 0.0
    return (xd,yd,zd, xd_d,yd_d,zd_d, xd_dd,yd_dd,zd_dd, psi_d,psi_d_d,psi_d_dd)

# ---------- his controller (returns 4 U's for the dataset) ----------
def controller_u(t, X):
    x,y,z = X[0],X[1],X[2]; vx,vy,vz = X[3],X[4],X[5]
    phi,theta,psi = X[6],X[7],X[8]; p,q,r = X[9],X[10],X[11]
    (xd,yd,zd, xd_d,yd_d,zd_d, xd_dd,yd_dd,zd_dd, psi_d,psi_d_d,psi_d_dd) = desired_trajectory(t)
    ex,ey,ez = x-xd, y-yd, z-zd
    evx,evy,evz = vx-xd_d, vy-yd_d, vz-zd_d
    ax_cmd = sat(xd_dd - Kpx*ex - Kdx*evx, -12.0, 12.0)
    ay_cmd = sat(yd_dd - Kpy*ey - Kdy*evy, -12.0, 12.0)
    az_cmd = sat(zd_dd - Kpz*ez - Kdz*evz, -10.0, 10.0)
    den = np.cos(phi)*np.cos(theta); den = np.sign(den)*max(abs(den),0.25)
    U1 = sat(m*(g+az_cmd)/den, 1.0, 45.0)
    phi_d = sat((1.0/g)*(ax_cmd*np.sin(psi) - ay_cmd*np.cos(psi)), -0.65, 0.65)
    theta_d = sat((1.0/g)*(ax_cmd*np.cos(psi) + ay_cmd*np.sin(psi)), -0.65, 0.65)
    U2 = sat(Ix/l*(-Kp_phi*(phi-phi_d) - Kd_phi*p - q*r*((Iy-Iz)/Ix)), -5.0, 5.0)
    U3 = sat(Iy/l*(-Kp_theta*(theta-theta_d) - Kd_theta*q - p*r*((Iz-Ix)/Iy)), -5.0, 5.0)
    U4 = sat(Iz*(-Kp_psi*(psi-psi_d) - Kd_psi*(r-psi_d_d) - p*q*((Ix-Iy)/Iz)), -5.0, 5.0)
    return np.array([U1, U2, U3, U4])

# ---------- his matched dynamics ----------
def dynamics(t, X):
    x,y,z = X[0],X[1],X[2]; vx,vy,vz = X[3],X[4],X[5]
    phi,theta,psi = X[6],X[7],X[8]; p,q,r = X[9],X[10],X[11]
    U1,U2,U3,U4 = controller_u(t, X)
    Ux = np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi)
    Uy = np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi)
    x_dd = Ux*U1/m; y_dd = Uy*U1/m; z_dd = np.cos(phi)*np.cos(theta)*U1/m - g
    phi_dd = q*r*((Iy-Iz)/Ix) + l*U2/Ix
    theta_dd = p*r*((Iz-Ix)/Iy) + l*U3/Iy
    psi_dd = p*q*((Ix-Iy)/Iz) + U4/Iz
    return np.array([vx,vy,vz, x_dd,y_dd,z_dd, p,q,r, phi_dd,theta_dd,psi_dd])

def generate(N=50, dt=0.01, t_end=30.0, seed=0):
    t_eval = np.arange(0.0, t_end, dt)
    rng = np.random.default_rng(seed)
    all_traj, all_u = [], []
    # his path starts at (R,0,0); scatter initial conditions around it
    for i in range(N):
        x0 = np.zeros(12)
        x0[0] = R_REF + rng.uniform(-1.0, 1.0)
        x0[1] = 0.0   + rng.uniform(-1.0, 1.0)
        x0[2] = 0.0   + rng.uniform(-0.5, 0.5)
        x0[3] = 0.0   + rng.normal(0, 0.05)
        x0[4] = R_REF*W_REF + rng.normal(0, 0.05)   # match his initial vy
        x0[5] = VZ_REF + rng.normal(0, 0.05)
        x0[6] = rng.normal(0, 0.02); x0[7] = rng.normal(0, 0.02); x0[8] = rng.normal(0, 0.02)
        x0[11] = W_REF
        sol = solve_ivp(dynamics, (0.0, t_end), x0, t_eval=t_eval,
                        method="RK45", rtol=1e-6, atol=1e-8, max_step=0.02)
        traj = sol.y.T
        u_traj = np.array([controller_u(t_eval[k], traj[k]) for k in range(traj.shape[0])])
        all_traj.append(traj); all_u.append(u_traj)
        print(f"  flight {i+1}/{N} done", flush=True)
    return t_eval, np.array(all_traj), np.array(all_u)

if __name__ == "__main__":
    T, X, U = generate(N=50)
    # sanity checks
    desired = np.array([desired_trajectory(t)[:3] for t in T])
    rmse = np.sqrt(((X[0,:,:3] - desired)**2).mean())
    print("\nStates shape:", X.shape, " Control shape:", U.shape, flush=True)
    print(f"flight-0 tracking RMSE (his controller): {rmse:.3f} m", flush=True)
    print(f"x range [{X[:,:,0].min():.1f}, {X[:,:,0].max():.1f}]  z range [{X[:,:,2].min():.1f}, {X[:,:,2].max():.1f}]", flush=True)
    os.makedirs("datasets", exist_ok=True)
    np.savez("datasets/spiral_prof_dataset.npz", T=T, X=X, U=U)
    print("Saved datasets/spiral_prof_dataset.npz", flush=True)