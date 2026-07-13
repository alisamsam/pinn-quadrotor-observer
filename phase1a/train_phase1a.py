import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from pinn_observer_v2 import PINNObserverV2
from data_phase1a import load_phase1a_data, MEAS_IDX
from losses_v2 import data_loss_v2
from physics_loss import physics_loss

# --- settings ---
EPOCHS      = 8000
LR          = 1e-3
LAMBDA_PHYS = 1.0
LAMBDA_INIT = 1.0
LOG_EVERY   = 250

# --- load data ---
d = load_phase1a_data()
T_train = d["T_train"]      # (1800, 1)
Y_train = d["Y_train"]      # (1800, 6)  measured
U_train = d["U_train"]      # (1800, 4)  controls
X_train = d["X_train"]      # (1800, 12) full (for initial condition + eval)

# --- known initial condition (from the true trajectory at t=0) ---
x0_true = X_train[0:1, :]   # (1, 12) the real starting state
t0      = T_train[0:1, :]   # (1, 1)

# --- model + optimizer ---
model = PINNObserverV2()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Training Phase 1a observer (partial measurement)...")
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()

    # Teacher 1: data loss (6 measured states)
    loss_data = data_loss_v2(model, T_train, Y_train)

    # Teacher 2: physics loss (residual, uses controls U)
    loss_phys = physics_loss(model, T_train, U_train)

    # Teacher 3: initial-condition loss (pin estimate at t=0 to true start)
    x0_pred = model(t0)                       # (1, 12)
    loss_init = nn.functional.mse_loss(x0_pred, x0_true)

    # total
    loss = loss_data + LAMBDA_PHYS * loss_phys + LAMBDA_INIT * loss_init
    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % LOG_EVERY == 0:
        print(f"epoch {epoch:5d} | total {loss.item():.4e} | "
              f"data {loss_data.item():.4e} | "
              f"phys {loss_phys.item():.4e} | "
              f"init {loss_init.item():.4e}")

# --- save the trained model ---
torch.save(model.state_dict(), "phase1a/pinn_phase1a_8000.pth")
print("\nSaved trained model to phase1a/pinn_phase1a_8000.pth")