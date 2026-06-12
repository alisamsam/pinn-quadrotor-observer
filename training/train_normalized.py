"""Step 8: train with per-state normalized losses (fixes angle attenuation)."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from models.pinn_observer import PINNObserver
from training.losses import physics_loss, M, G   # reuse physics loss as-is

torch.manual_seed(0)

N_EPOCHS = 200
LR = 1e-3
NOISE_STD = 0.02
LAMBDA_PHYS = 1.0      # the balanced weight that won before
LAMBDA_INIT = 1.0

# ===== load dataset =====
data = np.load("data/hover_dataset.npz")
T = data["T"]
X = data["X"]
N_traj, N_steps, n_states = X.shape

# ===== compute per-state sigma (the 12 spreads) =====
X_all = X.reshape(-1, n_states)                       # (25000, 12)
sigma = X_all.std(axis=0)                              # (12,) one std per state
sigma = np.where(sigma < 1e-6, 1.0, sigma)            # guard against divide-by-zero
sigma_t = torch.tensor(sigma, dtype=torch.float32)
print("Per-state sigma:", np.round(sigma, 4))

# ===== build training tensors =====
X_flat = X_all
t_flat = np.tile(T, N_traj).reshape(-1, 1)
rng = np.random.default_rng(0)
Y_flat = X_flat + rng.normal(0, NOISE_STD, X_flat.shape)

X0_true = X[:, 0, :]
t0 = np.zeros((N_traj, 1))
Y0 = X0_true + rng.normal(0, NOISE_STD, X0_true.shape)

t_t  = torch.tensor(t_flat,  dtype=torch.float32)
Y_t  = torch.tensor(Y_flat,  dtype=torch.float32)
t0_t = torch.tensor(t0,      dtype=torch.float32)
Y0_t = torch.tensor(Y0,      dtype=torch.float32)
X0_t = torch.tensor(X0_true, dtype=torch.float32)

u_t = torch.zeros(t_flat.shape[0], 4)
u_t[:, 0] = M * G

# ===== NORMALIZED loss helpers =====
def data_loss_norm(model, t, y, sig):
    pred = model(t, y)
    return torch.mean(((pred - y) / sig) ** 2)        # divide error by sigma

def initial_loss_norm(model, t0, y0, x0, sig):
    pred0 = model(t0, y0)
    return torch.mean(((pred0 - x0) / sig) ** 2)

# ===== model + optimizer =====
model = PINNObserver()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("\nEpoch |   Total  |   Data   | Physics  | Initial")
print("-" * 52)
for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    ld = data_loss_norm(model, t_t, Y_t, sigma_t)
    lp = physics_loss(model, t_t, Y_t, u_t)           # physics unchanged for now
    li = initial_loss_norm(model, t0_t, Y0_t, X0_t, sigma_t)
    total = ld + LAMBDA_PHYS * lp + LAMBDA_INIT * li
    total.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
        print(f"{epoch:5d} | {total.item():8.4f} | {ld.item():8.4f} | "
              f"{lp.item():8.4f} | {li.item():8.4f}")

torch.save(model.state_dict(), "models/pinn_normalized.pth")
print("\nSaved -> models/pinn_normalized.pth")