import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

# --- settings ---
EPOCHS      = 2000
LR          = 1e-3
BATCH       = 4096
LAMBDA_PHYS = 1.0
LAMBDA_INIT = 1.0
LOG_EVERY   = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --- data ---
train, test = load_phase4_data()
T  = train["T"].to(device)
X0 = train["X0"].to(device)
Y  = train["Y"].to(device)
U  = train["U"].to(device)
X  = train["X"].to(device)
N  = T.shape[0]

# initial-condition points: the t=0 rows (where time == min)
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


print("Training Phase 4 observer ([t, x0], 40 trajectories)...")
for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(N, device=device)
    epoch_loss = 0.0
    n_batches = 0

    for start in range(0, N, BATCH):
        idx = perm[start:start + BATCH]
        tb, x0b, yb, ub = T[idx], X0[idx], Y[idx], U[idx]

        optimizer.zero_grad()

        # data loss (6 measured)
        x_hat = model(tb, x0b)
        loss_data = nn.functional.mse_loss(x_hat[:, MEAS_IDX], yb)

        # physics loss
        res = physics_residual(tb, x0b, ub)
        loss_phys = nn.functional.mse_loss(res, torch.zeros_like(res))

        # initial-condition loss (on the t=0 rows)
        x0_pred = model(T0, X0_0)
        loss_init = nn.functional.mse_loss(x0_pred, Xtrue_0)

        loss = loss_data + LAMBDA_PHYS*loss_phys + LAMBDA_INIT*loss_init
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    if epoch == 1 or epoch % LOG_EVERY == 0:
        print(f"epoch {epoch:4d} | avg loss {epoch_loss/n_batches:.4e}")

torch.save(model.state_dict(), "phase1a/pinn_phase4.pth")
print("\nSaved to phase1a/pinn_phase4.pth")