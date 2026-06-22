"""Singha training with mini-batching (fits in laptop RAM)."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from models.pinn_observer import PINNObserver
from training.losses import physics_loss

torch.manual_seed(0)
N_EPOCHS = 300
LR = 1e-3; NOISE_STD = 0.02
LAMBDA_PHYS = 1.0; LAMBDA_INIT = 1.0
BATCH = 2000          # <-- process 2000 random samples per step, not all 90k

data = np.load("datasets/singha_dataset.npz")
T = data["T"]; X = data["X"]; U = data["U"]
N_traj, N_steps, n_states = X.shape

X_all = X.reshape(-1, n_states)
sigma = X_all.std(axis=0)
sigma = np.where(sigma < 1e-6, 1.0, sigma)
sigma_t = torch.tensor(sigma, dtype=torch.float32)

t_flat = np.tile(T, N_traj).reshape(-1, 1)
U_flat = U.reshape(-1, 4)
rng = np.random.default_rng(0)
Y_flat = X_all + rng.normal(0, NOISE_STD, X_all.shape)
N_samples = X_all.shape[0]

X0 = X[:, 0, :]; t0 = np.zeros((N_traj, 1))
Y0 = X0 + rng.normal(0, NOISE_STD, X0.shape)

# full tensors (stay on CPU; we index small batches out of them)
t_all = torch.tensor(t_flat, dtype=torch.float32)
Y_all = torch.tensor(Y_flat, dtype=torch.float32)
U_all = torch.tensor(U_flat, dtype=torch.float32)
t0_t = torch.tensor(t0, dtype=torch.float32)
Y0_t = torch.tensor(Y0, dtype=torch.float32)
X0_t = torch.tensor(X0, dtype=torch.float32)

def dl(m,t,y,s): return torch.mean(((m(t,y)-y)/s)**2)
def il(m,t,y,x,s): return torch.mean(((m(t,y)-x)/s)**2)

model = PINNObserver(hidden=256, n_hidden_layers=3)
opt = torch.optim.Adam(model.parameters(), lr=LR)

print(f"Network: 3 x 256 | Batch: {BATCH} | Epochs: {N_EPOCHS}")
print("\nEpoch |   Total  |   Data   | Physics  | Initial")
print("-"*52)
gen = np.random.default_rng(1)
for e in range(N_EPOCHS):
    # pick a random mini-batch of indices
    idx = gen.integers(0, N_samples, size=BATCH)
    tb = t_all[idx]; yb = Y_all[idx]; ub = U_all[idx]

    opt.zero_grad()
    ld = dl(model, tb, yb, sigma_t)
    lp = physics_loss(model, tb, yb, ub)       # physics on the small batch
    li = il(model, t0_t, Y0_t, X0_t, sigma_t)
    tot = ld + LAMBDA_PHYS*lp + LAMBDA_INIT*li
    tot.backward(); opt.step()
    if e%30==0 or e==N_EPOCHS-1:
        print(f"{e:5d} | {tot.item():8.4f} | {ld.item():8.4f} | {lp.item():8.4f} | {li.item():8.4f}")

torch.save(model.state_dict(), "models/pinn_singha_v3.pth")
print("\nSaved -> models/pinn_singha_v3.pth")