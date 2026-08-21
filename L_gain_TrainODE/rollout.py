# -*- coding: utf-8 -*-
"""
Differentiable observer rollout for TRAINING L "through the integration".

Unlike observer_v2_L_gain.integrate_observer (which is @torch.no_grad, for evaluation),
this version keeps the autograd graph so gradients flow back through the ODE integration
to the network parameters that produce the gain L. Used to train L so that integrating
the observer with it actually drives the estimate onto the true state.

Observer ODE (Farkane), integrated with RK4 over a short window:
    x_hat_dot = f(x_hat, u) + L(t) . ( y - C x_hat )
"""
import torch


def build_C(meas_idx, device="cpu"):
    C = torch.zeros(len(meas_idx), 12, device=device)
    for r, c in enumerate(meas_idx):
        C[r, c] = 1.0
    return C


def rollout_observer(model, dyn_fn, C, t_win, u_win, y_win, x0_cond, x_hat0):
    """
    RK4 rollout over a window, differentiable w.r.t. model parameters.
      t_win  : (W,)      window times (uniform dt)
      u_win  : (W, B, 4) controls per step (zero-order hold within a step)
      y_win  : (W, B, 6) measured outputs per step
      x0_cond: (B, 12)   network conditioning initial state (the flight's x0)
      x_hat0 : (B, 12)   observer initial estimate (deliberately wrong)
    Returns x_traj : (W, B, 12)  the integrated estimate trajectory.
    """
    B = x_hat0.shape[0]
    W = t_win.shape[0]
    dt = (t_win[1] - t_win[0]).item()
    x = x_hat0
    traj = [x]
    for k in range(W - 1):
        tk = t_win[k].view(1, 1).expand(B, 1)
        _, Lk = model.get_state_and_gain(tk, x0_cond)      # (B,12,6), grad flows here
        uk = u_win[k]                                      # (B,4)  zero-order hold
        yk = y_win[k]                                      # (B,6)

        def rhs(xh):
            f = dyn_fn(xh, uk)                             # (B,12)
            innovation = yk - xh @ C.t()                  # y - C x_hat  (B,6)
            correction = torch.bmm(Lk, innovation.unsqueeze(-1)).squeeze(-1)  # L.(y-Cx_hat)
            return f + correction

        k1 = rhs(x)
        k2 = rhs(x + 0.5 * dt * k1)
        k3 = rhs(x + 0.5 * dt * k2)
        k4 = rhs(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(x)
    return torch.stack(traj, dim=0)                        # (W,B,12)


# --- tiny self-test: shapes + gradient actually flows to the model ---
if __name__ == "__main__":
    import sys, os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(ROOT, "phase1a"))
    from pinn_observer_v5 import PINNObserverV5
    from dynamics_torch import quadrotor_dynamics_torch
    MEAS_IDX = [0, 1, 2, 6, 7, 8]
    dev = "cpu"
    m = PINNObserverV5(hidden=32, n_hidden_layers=2).to(dev)
    C = build_C(MEAS_IDX, dev)
    W, B = 20, 4
    t_win = torch.linspace(0, 0.19, W)
    u_win = torch.zeros(W, B, 4); u_win[..., 0] = 1.8 * 9.81   # hover-ish thrust
    y_win = torch.zeros(W, B, 6)
    x0 = torch.zeros(B, 12)
    x_hat0 = x0 + 0.1 * torch.randn(B, 12)                     # wrong start
    traj = rollout_observer(m, quadrotor_dynamics_torch, C, t_win, u_win, y_win, x0, x_hat0)
    loss = (traj ** 2).mean()
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    print("traj shape:", tuple(traj.shape), "| loss", float(loss), "| grad flows:", gnorm > 0)
