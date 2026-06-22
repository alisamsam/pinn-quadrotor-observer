"""Step 7: train the PINN observer on the hover dataset."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from models.pinn_observer import PINNObserver
from training.losses import total_loss, M, G

torch.manual_seed(0)

# ----- Hyperparameters -----
N_EPOCHS  = 200
LR        = 1e-3
NOISE_STD = 0.02          # how noisy the fake sensor is
LAMBDA_PHYS = 0.1
LAMBDA_INIT = 1.0

# ===== 7.1 load the real dataset =====
data = np.load("datasets/hover_dataset.npz")
T = data["T"]             # (500,)   times
X = data["X"]             # (50, 500, 12)  true states

N_traj, N_steps, n_states = X.shape
print(f"Loaded {N_traj} trajectories, {N_steps} steps each, {n_states} states")

# ----- flatten (50, 500, 12) -> (25000, 12): one big pile of samples -----
X_flat = X.reshape(-1, n_states)                       # (25000, 12) true states
# time for each sample: tile the 500 times across all 50 trajectories
t_flat = np.tile(T, N_traj).reshape(-1, 1)             # (25000, 1)

# ===== 7.2 make noisy measurements (fake sensor) =====
rng = np.random.default_rng(0)
Y_flat = X_flat + rng.normal(0, NOISE_STD, X_flat.shape)   # true + noise

# ----- initial-condition samples: the t=0 frame of each trajectory -----
X0_true = X[:, 0, :]                                   # (50, 12) true starts
t0      = np.zeros((N_traj, 1))                        # (50, 1)
Y0      = X0_true + rng.normal(0, NOISE_STD, X0_true.shape)

# ----- convert everything to torch tensors -----
t_t   = torch.tensor(t_flat,  dtype=torch.float32)
Y_t   = torch.tensor(Y_flat,  dtype=torch.float32)
t0_t  = torch.tensor(t0,      dtype=torch.float32)
Y0_t  = torch.tensor(Y0,      dtype=torch.float32)
X0_t  = torch.tensor(X0_true, dtype=torch.float32)

# control input: constant hover thrust for every sample
u_t = torch.zeros(t_flat.shape[0], 4)
u_t[:, 0] = M * G

# ===== 7.3 model + optimizer =====
model = PINNObserver()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ===== 7.4 the epoch loop =====
print("\nEpoch |   Total  |   Data   | Physics  | Initial")
print("-" * 52)
for epoch in range(N_EPOCHS):
    optimizer.zero_grad()                              # clear old knob-nudges
    total, ld, lp, li = total_loss(
        model, t_t, Y_t, u_t, t0_t, Y0_t, X0_t,
        lambda_phys=LAMBDA_PHYS, lambda_init=LAMBDA_INIT
    )
    total.backward()                                   # backprop: which way to turn knobs
    optimizer.step()                                   # nudge every knob

    if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
        print(f"{epoch:5d} | {total.item():8.4f} | {ld.item():8.4f} | "
              f"{lp.item():8.4f} | {li.item():8.4f}")

# ===== 7.6 save the trained model =====
fname = f"models/pinn_lam{LAMBDA_PHYS}.pth"
torch.save(model.state_dict(), fname)
print(f"\nSaved trained model -> {fname}")
