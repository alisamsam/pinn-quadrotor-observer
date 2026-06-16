"""Phase 2 Step 2.6: train observer on controlled setpoint data."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from models.pinn_observer import PINNObserver
from training.losses import physics_loss

torch.manual_seed(0)
N_EPOCHS = 200; LR = 1e-3; NOISE_STD = 0.02
LAMBDA_PHYS = 1.0; LAMBDA_INIT = 1.0

# ===== load CONTROLLED dataset =====
data = np.load("data/setpoint_dataset.npz")
T = data["T"]; X = data["X"]; U = data["U"]      # U is new: (50, 800, 4)
N_traj, N_steps, n_states = X.shape

# per-state sigma (normalization, same as the fix)
X_all = X.reshape(-1, n_states)
sigma = X_all.std(axis=0)
sigma = np.where(sigma < 1e-6, 1.0, sigma)
sigma_t = torch.tensor(sigma, dtype=torch.float32)
print("Per-state sigma:", np.round(sigma, 3))

# flatten everything to samples
t_flat = np.tile(T, N_traj).reshape(-1, 1)
X_flat = X_all
U_flat = U.reshape(-1, 4)                          # (40000, 4) the real control
rng = np.random.default_rng(0)
Y_flat = X_flat + rng.normal(0, NOISE_STD, X_flat.shape)

X0_true = X[:, 0, :]; t0 = np.zeros((N_traj, 1))
Y0 = X0_true + rng.normal(0, NOISE_STD, X0_true.shape)

t_t  = torch.tensor(t_flat, dtype=torch.float32)
Y_t  = torch.tensor(Y_flat, dtype=torch.float32)
U_t  = torch.tensor(U_flat, dtype=torch.float32)   # time-varying control
t0_t = torch.tensor(t0, dtype=torch.float32)
Y0_t = torch.tensor(Y0, dtype=torch.float32)
X0_t = torch.tensor(X0_true, dtype=torch.float32)

def data_loss_norm(model, t, y, sig):
    return torch.mean(((model(t, y) - y) / sig) ** 2)

def initial_loss_norm(model, t0, y0, x0, sig):
    return torch.mean(((model(t0, y0) - x0) / sig) ** 2)

model = PINNObserver()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("\nEpoch |   Total  |   Data   | Physics  | Initial")
print("-" * 52)
for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    ld = data_loss_norm(model, t_t, Y_t, sigma_t)
    lp = physics_loss(model, t_t, Y_t, U_t)        # <-- real time-varying U
    li = initial_loss_norm(model, t0_t, Y0_t, X0_t, sigma_t)
    total = ld + LAMBDA_PHYS*lp + LAMBDA_INIT*li
    total.backward(); optimizer.step()
    if epoch % 20 == 0 or epoch == N_EPOCHS-1:
        print(f"{epoch:5d} | {total.item():8.4f} | {ld.item():8.4f} | {lp.item():8.4f} | {li.item():8.4f}")

torch.save(model.state_dict(), "models/pinn_setpoint.pth")
print("\nSaved -> models/pinn_setpoint.pth")