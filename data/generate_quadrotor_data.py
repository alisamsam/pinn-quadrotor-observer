import numpy as np

# --- Singha et al. (2024), Table 2 parameters ---
m  = 1.80      # mass (kg)
l  = 0.20      # arm length (m)
g  = 9.81      # gravity (m/s^2)
Ix = 0.03      # inertia about x (kg·m^2)
Iy = 0.03      # inertia about y (kg·m^2)
Iz = 0.04      # inertia about z (kg·m^2)


def quadrotor_dynamics(t, x, u):
    """
    Singha 12-state quadrotor dynamics.
    State x: [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot]
    Input u: [U1, U2, U3, U4]  (thrust, roll torque, pitch torque, yaw torque)
    Returns: dx/dt, a length-12 array.
    """
    # --- unpack the 12 states (so the physics reads like the paper) ---
    x_, y_, z_, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot = x
    U1, U2, U3, U4 = u

    # --- the two trig helper terms from eq. (1) ---
    Ux = np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)   # >>> YOU WRITE THIS: Ux = cosψ·sinθ·cosφ + sinψ·sinφ   (use np.cos / np.sin)
    Uy = np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)   # >>> YOU WRITE THIS: Uy = sinψ·sinθ·cosφ − cosψ·sinφ

    # --- translational accelerations, eq. (1) ---
    xddot = (Ux/m)*U1   # >>> YOU WRITE THIS: (Ux/m)·U1
    yddot = (Uy/m)*U1   # >>> YOU WRITE THIS: (Uy/m)·U1
    zddot = (np.cos(phi) * np.cos(theta)/m)*U1 - g  # >>> YOU WRITE THIS: (cosφ·cosθ/m)·U1 − g

    # --- angular accelerations, eq. (2) ---
    phiddot   = thetadot * psidot * (Iy - Iz) / Ix + (l / Ix) * U2   # >>> YOU WRITE THIS: θ̇·ψ̇·(Iy−Iz)/Ix + (l/Ix)·U2
    thetaddot = psidot * phidot * (Iz - Ix) / Iy + (l/Iy) * U3  # >>> YOU WRITE THIS: φ̇·ψ̇·(Iz−Ix)/Iy + (l/Iy)·U3
    psiddot   = phidot * thetadot * (Ix -Iy) / Iz + (l/Iz) * U4   # >>> YOU WRITE THIS: φ̇·θ̇·(Ix−Iy)/Iz + (l/Iz)·U4

    # --- assemble dx/dt: 6 copied-through + 6 computed (already done for you) ---
    dxdt = np.array([
        xdot,       # d/dt(x)   = velocity  (copied)
        ydot,       # d/dt(y)   = velocity  (copied)
        zdot,       # d/dt(z)   = velocity  (copied)
        xddot,      # d/dt(xdot)= acceleration
        yddot,
        zddot,
        phidot,     # d/dt(phi)   = angular rate (copied)
        thetadot,
        psidot,
        phiddot,    # d/dt(phidot)= angular acceleration
        thetaddot,
        psiddot,
    ])
    return dxdt


# --- Step 2: hover sanity-check ---
if __name__ == "__main__":
    x_hover = np.zeros(12)
    U1_hover = m * g          # = 17.658 N
    u_hover = [U1_hover, 0.0, 0.0, 0.0]

    dxdt = quadrotor_dynamics(0.0, x_hover, u_hover)
    print("dx/dt at hover:", np.round(dxdt, 9))
    print("All zero?      ", np.allclose(dxdt, 0.0))

    # --- Step 3: wake up the terms hover couldn't test ---
    dxdt = quadrotor_dynamics(0.0, np.zeros(12), [m*g + 1.0, 0, 0, 0])
    print("zddot with extra thrust:", round(dxdt[5], 4))    # expect small POSITIVE

    dxdt = quadrotor_dynamics(0.0, np.zeros(12), [m*g, 0.01, 0, 0])
    print("phiddot with roll torque:", round(dxdt[9], 4))   # expect POSITIVE
