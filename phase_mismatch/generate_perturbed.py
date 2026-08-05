"""
Standalone perturbed-flight generator for the model-mismatch study.
Own copy of the Singha dynamics + spiral controller, with physical
parameters passed as arguments (dict P) instead of module constants.
Existing pipeline files are left untouched. Controller is UPDATED to the
perturbed parameters, so only the observer keeps the nominal assumptions.
Variable names match core/generate_quadrotor_data.py exactly.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp
from scenarios.spiral.spiral_reference import spiral_reference, yaw_reference

g = 9.81
NOMINAL = dict(m=1.80, Ix=0.03, Iy=0.03, Iz=0.04, l=0.20)

KP_Z, KD_Z   = 4.0, 3.0
KP_XY, KD_XY = 0.6, 1.0
KP_ATT, KD_ATT = 8.0, 4.0
KP_YAW, KD_YAW = 2.0, 2.0
def quadrotor_dynamics(t, x, u, P):
    m, Ix, Iy, Iz, l = P['m'], P['Ix'], P['Iy'], P['Iz'], P['l']
    x_, y_, z_, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot = x
    U1, U2, U3, U4 = u
    Ux = np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi)
    Uy = np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi)
    xddot = (Ux/m)*U1
    yddot = (Uy/m)*U1
    zddot = (np.cos(phi)*np.cos(theta)/m)*U1 - g
    phiddot   = thetadot*psidot*(Iy - Iz)/Ix + (l/Ix)*U2
    thetaddot = psidot*phidot*(Iz - Ix)/Iy + (l/Iy)*U3
    psiddot   = phidot*thetadot*(Ix - Iy)/Iz + (l/Iz)*U4
    return np.array([xdot, ydot, zdot, xddot, yddot, zddot,
                     phidot, thetadot, psidot, phiddot, thetaddot, psiddot])

def outer_loop_ff(x, U1, pos_d, vel_d, psi, P):
    """Outer loop WITH feedforward: track moving (pos_d, vel_d). psi = desired yaw (as in the real code)."""
    m = P['m']
    px, py = x[0], x[1]
    vx, vy = x[3], x[4]
    Uex = (m/U1) * (KP_XY*(pos_d[0]-px) + KD_XY*(vel_d[0]-vx))
    Uey = (m/U1) * (KP_XY*(pos_d[1]-py) + KD_XY*(vel_d[1]-vy))
    Uex = np.clip(Uex, -0.99, 0.99)
    Uey = np.clip(Uey, -0.99, 0.99)
    phi_d = np.arcsin(Uex*np.sin(psi) - Uey*np.cos(psi))
    phi = x[6]
    theta_d = np.arcsin(np.clip(
        (Uex - np.sin(phi)*np.sin(psi)) / (np.cos(phi)*np.cos(psi)), -0.99, 0.99))
    return phi_d, theta_d


def full_control_spiral(t, x, P):
    m, Ix, Iy, Iz, l = P['m'], P['Ix'], P['Iy'], P['Iz'], P['l']
    pos_d, vel_d, acc_d = spiral_reference(t)
    psi_d, psi_d_dot = yaw_reference(t)

    z, zdot = x[2], x[5]
    U1 = m*g + KP_Z*(pos_d[2] - z) + KD_Z*(vel_d[2] - zdot)

    # NOTE: pass psi_d (desired yaw) into the outer loop, exactly as the real code does
    phi_d, theta_d = outer_loop_ff(x, U1, pos_d, vel_d, psi_d, P)

    phi, theta, psi = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    U2 = (Ix/l)*(KP_ATT*(phi_d - phi) - KD_ATT*p - q*r*(Iy-Iz)/Ix)
    U3 = (Iy/l)*(KP_ATT*(theta_d - theta) - KD_ATT*q - p*r*(Iz-Ix)/Iy)
    U4 = Iz*(KP_YAW*(psi_d - psi) + KD_YAW*(psi_d_dot - r) - p*q*(Ix-Iy)/Iz)
    return np.array([U1, U2, U3, U4])


def generate_set(P, init_conditions, dt=0.01, t_end=30.0):
    t_eval = np.arange(0.0, t_end, dt)
    trajs, us = [], []
    for x0 in init_conditions:
        sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, full_control_spiral(t, x, P), P),
                        (0.0, t_end), x0, t_eval=t_eval, max_step=0.02)
        traj = sol.y.T
        u_traj = np.array([full_control_spiral(t_eval[k], traj[k], P) for k in range(traj.shape[0])])
        trajs.append(traj); us.append(u_traj)
    return t_eval, np.array(trajs), np.array(us)


if __name__ == '__main__':
    base = np.load('datasets/spiral_dataset.npz')
    X = base['X']
    test_x0 = [X[i, 0, :] for i in range(40, 50)]
    os.makedirs('datasets/mismatch', exist_ok=True)
    deviations = [-0.20, -0.10, 0.0, 0.10, 0.20]
    for dev in deviations:
        P = dict(NOMINAL); P['m'] = NOMINAL['m']*(1+dev)
        tag = 'mass_%+d' % int(round(dev*100))
        if dev == 0.0: tag = 'mass_0'
        T, Xp, Up = generate_set(P, test_x0)
        np.savez('datasets/mismatch/%s.npz' % tag, T=T, X=Xp, U=Up, mass=P['m'], dev=dev)
        print('mass %.3f (%+.0f%%) -> %s  shape %s' % (P['m'], dev*100, tag, str(Xp.shape)), flush=True)
    print('done', flush=True)