import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from data_multi import load_multi_data, MEAS_IDX

# reuse the v4 network and torch dynamics from phase1a
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))
from pinn_observer_v4 import PINNObserverV4
from dynamics_torch import quadrotor_dynamics_torch

EPOCHS, LR, BATCH = 2000, 1e-3, 8192
LAMBDA_PHYS, LAMBDA_INIT, LOG_EVERY = 1.0, 1.0, 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train, test = load_multi_data()
T, X0, Y, U, X = [train[k].to(device) for k in ("T","X0","Y","U","X")]
N = T.shape[0]

t0_mask = (T[:, 0] == T[:, 0].min())
T0, X0_0, Xtrue_0 = T[t0_mask], X0[t0_mask], X[t0_mask]

model = PINNObserverV4().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def physics_residual(t, x0, u):
    t = t.clone().requires_grad_(True)
    x_hat = model(t, x0)
    dxdt = torch.zeros_like(x_hat)
    for i in range(12):
        g = torch.autograd.grad(x_hat[:, i].sum(), t, create_graph=True)[0]
        dxdt[:, i] = g[:, 0]
    f = quadrotor_dynamics_torch(x_hat, u)
    return dxdt - f


print("Training MULTI-trajectory observer (circle+figure8+spiral)...")
for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(N, device=device)
    ep_loss, nb = 0.0, 0
    for s in range(0, N, BATCH):
        idx = perm[s:s+BATCH]
        tb, x0b, yb, ub = T[idx], X0[idx], Y[idx], U[idx]
        optimizer.zero_grad()

        x_hat = model(tb, x0b)
        loss_data = nn.functional.mse_loss(x_hat[:, MEAS_IDX], yb)

        res = physics_residual(tb, x0b, ub)
        loss_phys = nn.functional.mse_loss(res, torch.zeros_like(res))

        x0_pred = model(T0, X0_0)
        loss_init = nn.functional.mse_loss(x0_pred, Xtrue_0)

        loss = loss_data + LAMBDA_PHYS*loss_phys + LAMBDA_INIT*loss_init
        loss.backward()
        optimizer.step()
        ep_loss += loss.item(); nb += 1

    if epoch == 1 or epoch % LOG_EVERY == 0:
        print(f"epoch {epoch:4d} | avg loss {ep_loss/nb:.4e}")

torch.save(model.state_dict(), "phase_multi/pinn_multi.pth")
print("\nSaved to phase_multi/pinn_multi.pth")