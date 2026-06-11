import torch
import torch.nn as nn


def data_loss(model, t, y_noisy):
    """
    Teacher 1: estimate should match the noisy measurements.
    t       : (batch, 1)  time
    y_noisy : (batch, 12) noisy measured states
    """
    pred = model(t, y_noisy)              # (batch, 12) network's estimate
    mse = nn.functional.mse_loss(pred, y_noisy)
    return mse

# --- quadrotor parameters (same as Singha, Step 1) ---
M, L_ARM, G = 1.80, 0.20, 9.81
IX, IY, IZ = 0.03, 0.03, 0.04


def quadrotor_dynamics_torch(x, u):
    """
    Torch twin of Step 1 dynamics. Works on a whole batch.
    x : (batch, 12) state estimates
    u : (batch, 4)  control inputs
    Returns dx/dt : (batch, 12)
    """
    # unpack columns (each is shape (batch,))
    xdot   = x[:, 3]
    ydot   = x[:, 4]
    zdot   = x[:, 5]
    phi    = x[:, 6]
    theta  = x[:, 7]
    psi    = x[:, 8]
    phidot   = x[:, 9]
    thetadot = x[:, 10]
    psidot   = x[:, 11]
    U1, U2, U3, U4 = u[:, 0], u[:, 1], u[:, 2], u[:, 3]

    # trig helpers (eq. 1) -- torch, not numpy
    Ux = torch.cos(psi)*torch.sin(theta)*torch.cos(phi) + torch.sin(psi)*torch.sin(phi)
    Uy = torch.sin(psi)*torch.sin(theta)*torch.cos(phi) - torch.cos(psi)*torch.sin(phi)

    # translational accelerations (eq. 1)
    xddot = (Ux/M)*U1
    yddot = (Uy/M)*U1
    zddot = (torch.cos(phi)*torch.cos(theta)/M)*U1 - G

    # angular accelerations (eq. 2)
    phiddot   = thetadot*psidot*(IY-IZ)/IX + (L_ARM/IX)*U2
    thetaddot = phidot*psidot*(IZ-IX)/IY + (L_ARM/IY)*U3
    psiddot   = phidot*thetadot*(IX-IY)/IZ + (L_ARM/IZ)*U4

    # stack the 12 derivatives back into (batch, 12)
    dxdt = torch.stack([
        xdot, ydot, zdot,
        xddot, yddot, zddot,
        phidot, thetadot, psidot,
        phiddot, thetaddot, psiddot
    ], dim=1)
    return dxdt


def physics_loss(model, t, y_noisy, u):
    """
    Teacher 2: network's d(estimate)/dt must match the dynamics.
    """
    t = t.clone().detach().requires_grad_(True)   # turn ON time-tracking
    x_hat = model(t, y_noisy)                      # (batch, 12) estimate

    # 🅰️ autograd: d(x_hat)/dt for all 12 states, column by column
    dxhat_dt = torch.zeros_like(x_hat)
    for i in range(12):
        grad_i = torch.autograd.grad(
            outputs=x_hat[:, i:i+1], inputs=t,
            grad_outputs=torch.ones_like(x_hat[:, i:i+1]),
            create_graph=True, retain_graph=True
        )[0]
        dxhat_dt[:, i:i+1] = grad_i

    # 🅱️ what physics says the rate should be
    f_xhat = quadrotor_dynamics_torch(x_hat, u)

    # residual and its MSE
    residual = dxhat_dt - f_xhat
    return torch.mean(residual**2)

def initial_loss(model, t0, y0_noisy, x0_true):
    """
    Teacher 3: estimate at t=0 must match the known initial state.
    """
    x_hat_0 = model(t0, y0_noisy)
    return nn.functional.mse_loss(x_hat_0, x0_true)

def total_loss(model, t, y_noisy, u, t0, y0_noisy, x0_true,
               lambda_phys=0.1, lambda_init=1.0):
    """
    Combine all three teachers into one report card.
    """
    ld = data_loss(model, t, y_noisy)
    lp = physics_loss(model, t, y_noisy, u)
    li = initial_loss(model, t0, y0_noisy, x0_true)

    total = ld + lambda_phys * lp + lambda_init * li
    return total, ld, lp, li      # return all four so we can watch each

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.pinn_observer import PINNObserver

    torch.manual_seed(0)                  # <-- makes runs repeatable now

    model = PINNObserver()
    batch = 5
    t = torch.zeros(batch, 1)
    y = torch.randn(batch, 12)
    u = torch.zeros(batch, 4)
    u[:, 0] = M * G                       # hover thrust
    x0_true = torch.zeros(batch, 12)      # pretend true start = hover

    ld = data_loss(model, t, y)
    lp = physics_loss(model, t, y, u)
    li = initial_loss(model, t, y, x0_true)

    print("Data loss:   ", ld.item())
    print("Physics loss:", lp.item())
    print("Initial loss:", li.item())
    total, ld2, lp2, li2 = total_loss(model, t, y, u, t, y, x0_true)
    print("TOTAL loss:  ", total.item())
    print("All finite?  ",
          torch.isfinite(ld).item() and
          torch.isfinite(lp).item() and
          torch.isfinite(li).item())