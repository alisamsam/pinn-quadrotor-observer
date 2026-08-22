import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "phase1a"))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pinn_observer_v5 import PINNObserverV5
from dynamics_torch import quadrotor_dynamics_torch

# Stability test of the learned gain L (supervisor request).
# For each sampled instant t: A(t) = Jacobian of the physics f at the true
# state, C = selector of the 6 measured states, L(t) = learned gain.
# Eigenvalues of (A - L C): all real parts < 0  -> error decays (stable);
# any real part > 0 -> error grows (L is not a stabilizing gain).

MEAS_IDX = [0, 1, 2, 6, 7, 8]
FLIGHT = 45          # unseen test flight (flights 40-49 were never trained on)
STEP = 25            # evaluate every 25th instant (3000/25 = 120 points)

d = np.load("datasets/spiral_dataset.npz")
T, X, U = d["T"], d["X"], d["U"]
x_true = X[FLIGHT]; u_true = U[FLIGHT]; x0 = x_true[0]

model = PINNObserverV5()
model.load_state_dict(torch.load("phase1a/pinn_phase5.pth", map_location="cpu"))
model.eval()

Tt  = torch.tensor(T, dtype=torch.float32).unsqueeze(1)
X0t = torch.tensor(np.tile(x0, (len(T), 1)), dtype=torch.float32)
with torch.no_grad():
    _, L_all = model.get_state_and_gain(Tt, X0t)   # (n_steps, 12, 6)

C = np.zeros((6, 12)); C[np.arange(6), MEAS_IDX] = 1.0

times, max_re = [], []
for k in range(0, len(T), STEP):
    xk = torch.tensor(x_true[k], dtype=torch.float32)
    uk = torch.tensor(u_true[k], dtype=torch.float32)
    f_single = lambda x: quadrotor_dynamics_torch(x.unsqueeze(0), uk.unsqueeze(0)).squeeze(0)
    A = torch.autograd.functional.jacobian(f_single, xk).numpy()   # (12,12)
    L = L_all[k].numpy()                                           # (12,6)
    eigvals = np.linalg.eigvals(A - L @ C)
    times.append(T[k]); max_re.append(eigvals.real.max())

times = np.array(times); max_re = np.array(max_re)
frac_unstable = float((max_re > 0).mean())

os.makedirs("docs", exist_ok=True)
plt.figure(figsize=(11, 5))
plt.plot(times, max_re, "r-", linewidth=1.8, label="max Re eig(A - LC)")
plt.axhline(0, color="k", linewidth=1)
plt.fill_between(times, 0, max_re, where=(max_re > 0), color="red", alpha=0.15)
plt.xlabel("time (s)"); plt.ylabel("largest real part of eigenvalues")
plt.title(f"Stability test of learned L - unseen spiral flight {FLIGHT} "
          f"(above 0 = unstable; {frac_unstable*100:.0f}% of flight unstable)")
plt.grid(alpha=0.3); plt.legend()
plt.savefig("docs/eig_analysis_phase5.png", dpi=140, bbox_inches="tight")
plt.close()

print(f"Largest real part over the flight: {max_re.max():.3f}")
print(f"Smallest (best) value:             {max_re.min():.3f}")
print(f"Fraction of flight with max Re > 0 (unstable): {frac_unstable*100:.1f}%")
print()
if frac_unstable > 0:
    print("VERDICT: the learned L does NOT define a stable observer -")
    print("the error-dynamics matrix (A - LC) has growing directions during the flight.")
else:
    print("VERDICT: (A - LC) is stable along the whole flight - surprising, report it.")
print()
print("Saved docs/eig_analysis_phase5.png")