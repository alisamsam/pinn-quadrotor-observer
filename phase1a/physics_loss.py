import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # phase1a folder

import torch
import torch.nn as nn
from dynamics_torch import quadrotor_dynamics_torch


def physics_loss(model, t, u):
    """
    Phase 1a physics residual: d/dt(x_hat) - f(x_hat, u) should be zero.
    t : (batch, 1) time  -- MUST require grad for autograd
    u : (batch, 4) controls at those time points
    """
    t = t.clone().requires_grad_(True)   # enable autograd on time
    x_hat = model(t)                     # (batch, 12) estimate

    # d/dt of each of the 12 outputs, via autograd
    dxdt = torch.zeros_like(x_hat)
    for i in range(12):
        grad_i = torch.autograd.grad(
            x_hat[:, i].sum(), t,
            create_graph=True
        )[0]                             # (batch, 1)
        dxdt[:, i] = grad_i[:, 0]

    # physics: what the derivative SHOULD be
    f = quadrotor_dynamics_torch(x_hat, u)   # (batch, 12)

    residual = dxdt - f                  # (batch, 12), want zero
    return nn.functional.mse_loss(residual, torch.zeros_like(residual))


# --- sanity test ---
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pinn_observer_v2 import PINNObserverV2

    model = PINNObserverV2()
    batch = 5
    t = torch.rand(batch, 1)
    u = torch.zeros(batch, 4)
    u[:, 0] = 1.80 * 9.81                # hover thrust

    loss = physics_loss(model, t, u)
    print("Physics loss:", loss.item())
    print("Finite?", torch.isfinite(loss).item())