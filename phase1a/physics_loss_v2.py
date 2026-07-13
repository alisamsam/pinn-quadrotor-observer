import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from dynamics_torch import quadrotor_dynamics_torch

MEAS_IDX = [0, 1, 2, 6, 7, 8]


def physics_loss_v2(model, t, u, y_measured):
    """
    Phase 1b physics residual WITH the adaptive gain correction:
        residual = d/dt(x_hat) - f(x_hat, u) - L*(y - y_hat)
    t          : (batch, 1) time  (needs grad)
    u          : (batch, 4) controls
    y_measured : (batch, 6) live sensor measurements
    """
    t = t.clone().requires_grad_(True)
    x_hat, L = model.get_state_and_gain(t)     # (batch,12), (batch,12,6)

    # d/dt of each of the 12 states via autograd
    dxdt = torch.zeros_like(x_hat)
    for i in range(12):
        grad_i = torch.autograd.grad(
            x_hat[:, i].sum(), t, create_graph=True
        )[0]
        dxdt[:, i] = grad_i[:, 0]

    # physics term
    f = quadrotor_dynamics_torch(x_hat, u)     # (batch, 12)

    # correction term: L * (y - y_hat)
    y_hat = x_hat[:, MEAS_IDX]                  # (batch, 6) estimated measured states
    innov = (y_measured - y_hat).unsqueeze(-1)  # (batch, 6, 1)
    corr = torch.bmm(L, innov).squeeze(-1)      # (batch,12,6)x(batch,6,1)->(batch,12)

    residual = dxdt - f - corr                  # (batch, 12), want zero
    return nn.functional.mse_loss(residual, torch.zeros_like(residual))


# --- sanity test ---
if __name__ == "__main__":
    from pinn_observer_v3 import PINNObserverV3

    model = PINNObserverV3()
    batch = 5
    t = torch.rand(batch, 1)
    u = torch.zeros(batch, 4); u[:, 0] = 1.80 * 9.81
    y = torch.randn(batch, 6)

    loss = physics_loss_v2(model, t, u, y)
    print("Physics loss v2:", loss.item())
    print("Finite?", torch.isfinite(loss).item())