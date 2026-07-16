import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pinn_observer_v4 import PINNObserverV4

STATE_NAMES = ["x","y","z","vx","vy","vz","phi","theta","psi","p","q","r"]
UNITS = ["m","m","m","m/s","m/s","m/s","rad","rad","rad","rad/s","rad/s","rad/s"]
TEST_FLIGHT = 45   # an unseen flight (40-49)

device = torch.device("cpu")
model = PINNObserverV4()
model.load_state_dict(torch.load("phase_multi/pinn_multi.pth", map_location=device))
model.eval()

os.makedirs("docs", exist_ok=True)

for shape in ["circle", "figure8", "spiral"]:
    d = np.load(f"datasets/{shape}_dataset.npz")
    T, X = d["T"], d["X"]
    x_true = X[TEST_FLIGHT]                    # (n_steps, 12)
    x0 = x_true[0]

    Tt = torch.tensor(T, dtype=torch.float32).unsqueeze(1)
    X0t = torch.tensor(np.tile(x0, (len(T), 1)), dtype=torch.float32)
    with torch.no_grad():
        x_est = model(Tt, X0t).numpy()         # (n_steps, 12)

    # --- Plot 1: 3D trajectory, true vs estimated ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_true[:,0], x_true[:,1], x_true[:,2], "b-",  label="True", linewidth=2)
    ax.plot(x_est[:,0],  x_est[:,1],  x_est[:,2],  "r--", label="Estimated", linewidth=2)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"{shape.capitalize()} — unseen flight {TEST_FLIGHT}: true vs estimated")
    ax.legend()
    fig.savefig(f"docs/multi_{shape}_3d.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: 12-state overlay grid ---
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for i, ax in enumerate(axes.flat):
        ax.plot(T, x_true[:, i], "b-",  label="True", linewidth=1.5)
        ax.plot(T, x_est[:, i],  "r--", label="Estimated", linewidth=1.5)
        ax.set_title(f"{STATE_NAMES[i]} ({UNITS[i]})")
        ax.grid(alpha=0.3)
        if i == 0: ax.legend()
    fig.suptitle(f"{shape.capitalize()} — all 12 states, unseen flight {TEST_FLIGHT}", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"docs/multi_{shape}_states.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"{shape}: saved docs/multi_{shape}_3d.png and docs/multi_{shape}_states.png")

print("\nDone - 6 figures in docs/")