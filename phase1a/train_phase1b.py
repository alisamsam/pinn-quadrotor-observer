import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from pinn_observer_v3 import PINNObserverV3
from data_phase1a import load_phase1a_data, MEAS_IDX
from physics_loss_v2 import physics_loss_v2

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
X_train = d["X_train"]      # (1800, 12) full (init + eval)

# known initial condition
x0_true = X_train[0:1, :]   # (1, 12)
t0      = T_train[0:1, :]   # (1, 1)

# --- model + optimizer ---
model = PINNObserverV3()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Training Phase 1b observer (adaptive gain L)...")
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()

    # Teacher 1: data loss (6 measured states)
    x_hat, _ = model.get_state_and_gain(T_train)
    y_hat = x_hat[:, MEAS_IDX]
    loss_data = nn.functional.mse_loss(y_hat, Y_train)

    # Teacher 2: physics loss WITH correction (uses U and live measurements)
    loss_phys = physics_loss_v2(model, T_train, U_train, Y_train)

    # Teacher 3: initial-condition loss
    x0_pred, _ = model.get_state_and_gain(t0)
    loss_init = nn.functional.mse_loss(x0_pred, x0_true)

    loss = loss_data + LAMBDA_PHYS * loss_phys + LAMBDA_INIT * loss_init
    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % LOG_EVERY == 0:
        print(f"epoch {epoch:5d} | total {loss.item():.4e} | "
              f"data {loss_data.item():.4e} | "
              f"phys {loss_phys.item():.4e} | "
              f"init {loss_init.item():.4e}")

torch.save(model.state_dict(), "phase1a/pinn_phase1b_8000.pth")
print("\nSaved trained model to phase1a/pinn_phase1b_8000.pth")