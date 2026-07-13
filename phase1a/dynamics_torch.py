import torch

# Singha et al. (2024) parameters — same as your NumPy version
m  = 1.80
l  = 0.20
g  = 9.81
Ix = 0.03
Iy = 0.03
Iz = 0.04


def quadrotor_dynamics_torch(x, u):
    """
    PyTorch version of your Singha dynamics (for autograd).
    x : (batch, 12) state estimates
    u : (batch, 4)  controls [U1, U2, U3, U4]
    returns f(x,u) : (batch, 12) the 12 derivatives
    """
    # unpack columns (each is shape (batch,))
    x_, y_, z_       = x[:, 0], x[:, 1], x[:, 2]
    xdot, ydot, zdot = x[:, 3], x[:, 4], x[:, 5]
    phi, theta, psi  = x[:, 6], x[:, 7], x[:, 8]
    phidot, thetadot, psidot = x[:, 9], x[:, 10], x[:, 11]
    U1, U2, U3, U4 = u[:, 0], u[:, 1], u[:, 2], u[:, 3]

    # trig helper terms (eq. 1)
    Ux = torch.cos(psi)*torch.sin(theta)*torch.cos(phi) + torch.sin(psi)*torch.sin(phi)
    Uy = torch.sin(psi)*torch.sin(theta)*torch.cos(phi) - torch.cos(psi)*torch.sin(phi)

    # translational accelerations (eq. 1)
    xddot = (Ux/m)*U1
    yddot = (Uy/m)*U1
    zddot = (torch.cos(phi)*torch.cos(theta)/m)*U1 - g

    # angular accelerations (eq. 2)
    phiddot   = thetadot*psidot*(Iy - Iz)/Ix + (l/Ix)*U2
    thetaddot = psidot*phidot*(Iz - Ix)/Iy + (l/Iy)*U3
    psiddot   = phidot*thetadot*(Ix - Iy)/Iz + (l/Iz)*U4

    # assemble: 6 copied-through velocities/rates + 6 accelerations
    f = torch.stack([
        xdot, ydot, zdot,
        xddot, yddot, zddot,
        phidot, thetadot, psidot,
        phiddot, thetaddot, psiddot,
    ], dim=1)   # (batch, 12)
    return f


# --- sanity test: must match your NumPy hover result ---
if __name__ == "__main__":
    # hover: all states zero, U1 = m*g, rest zero
    x_hover = torch.zeros(1, 12)
    u_hover = torch.tensor([[m*g, 0.0, 0.0, 0.0]])
    f = quadrotor_dynamics_torch(x_hover, u_hover)
    print("f at hover:", torch.round(f, decimals=6))
    print("All zero?", torch.allclose(f, torch.zeros(1, 12), atol=1e-5))