# -*- coding: utf-8 -*-
"""
Farkane-style observer evaluation: instead of reading x_hat from the network,
integrate the observer ODE forward in time with the network's learned gain L(t):

    x_hat_dot(t) = f(x_hat(t), u(t)) + L(t) . C . (x(t) - x_hat(t))
                 = f(x_hat(t), u(t)) + L(t) . ( y(t) - C x_hat(t) )      (since y = C x)

Integrated with classical fixed-step RK4 on the dataset grid (dt = 0.01 s).
L(t), u(t), y(t) are taken along one flight; L(t) is produced by the network as a
function of [t, x0] (queried once at the sample times, then linearly interpolated).
"""
import torch


def build_C(meas_idx, device="cpu"):
    """Explicit 6x12 output matrix C: row i selects measured state meas_idx[i]."""
    C = torch.zeros(len(meas_idx), 12, device=device)
    for r, c in enumerate(meas_idx):
        C[r, c] = 1.0
    return C


def _interp(arr, tq, t0, dt, N):
    """Linear interpolation of arr (N, ...) at time tq on a uniform grid."""
    pos = (tq - t0) / dt
    pos = torch.clamp(pos, 0.0, float(N - 1))
    i0 = int(torch.floor(pos).item())
    i1 = min(i0 + 1, N - 1)
    frac = pos - i0
    return arr[i0] * (1.0 - frac) + arr[i1] * frac


@torch.no_grad()
def integrate_observer(model, dyn_fn, T, U, Y, x0_cond, C, device="cpu", x_hat0=None, substeps=1):
    """
    Integrate the observer ODE for ONE flight.
      model    : PINNObserverV5 (provides L(t) via get_state_and_gain)
      dyn_fn   : f(x_hat, u) -> (B,12)  (quadrotor_dynamics_torch)
      T (N,1), U (N,4), Y (N,6 measured) : one flight's time series
      x0_cond  : (12,) the [t,x0]-conditioning initial state for the network
      x_hat0   : observer initial estimate; default = network's x_hat at t0
    Returns (x_hat_integrated (N,12), x_hat_direct (N,12)).
    """
    model.eval()
    N = T.shape[0]
    t = T.view(-1).to(device)
    t0 = float(t[0].item())
    dt = float((t[1] - t[0]).item())
    U = U.to(device); Y = Y.to(device)
    x0b = x0_cond.view(1, 12).to(device)

    # L(t) and the direct network estimate at the sample times
    tt = t.view(-1, 1)
    x0rep = x0b.repeat(N, 1)
    x_hat_direct, L_all = model.get_state_and_gain(tt, x0rep)   # (N,12), (N,12,6)

    if x_hat0 is None:
        x_hat0 = x_hat_direct[0:1].clone()                      # observer starts at network x_hat(t0)
    x = x_hat0.view(1, 12).to(device)

    def rhs(tq, xhat):
        Lq = _interp(L_all, tq, t0, dt, N).unsqueeze(0)         # (1,12,6)
        uq = _interp(U, tq, t0, dt, N).unsqueeze(0)             # (1,4)
        yq = _interp(Y, tq, t0, dt, N).unsqueeze(0)             # (1,6)
        f = dyn_fn(xhat, uq)                                    # (1,12)
        innovation = yq - xhat @ C.t()                          # y - C x_hat  (1,6)
        correction = torch.bmm(Lq, innovation.unsqueeze(-1)).squeeze(-1)   # L.C(x - x_hat)
        return f + correction

    traj = torch.zeros(N, 12, device=device)
    traj[0] = x.view(-1)
    diverged_at = None
    for k in range(N - 1):
        h_big = float((t[k + 1] - t[k]).item())
        h = h_big / substeps
        tk = float(t[k].item())
        for s in range(substeps):
            ts = torch.tensor(tk + s * h, device=device)
            k1 = rhs(ts,         x)
            k2 = rhs(ts + h / 2, x + (h / 2) * k1)
            k3 = rhs(ts + h / 2, x + (h / 2) * k2)
            k4 = rhs(ts + h,     x + h * k3)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[k + 1] = x.view(-1)
        if diverged_at is None and not torch.isfinite(x).all():
            diverged_at = float(t[k + 1].item())
    return traj, x_hat_direct, diverged_at
