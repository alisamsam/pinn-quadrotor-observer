import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from spiral_reference import spiral_reference, OMEGA

FLIGHT = 0
d = np.load("datasets/spiral_dataset.npz")
T, X = d["T"], d["X"]
actual = X[FLIGHT]                                        # (steps, 12)
desired = np.array([spiral_reference(t)[0] for t in T])   # (steps, 3)

err = actual[:, :3] - desired                             # (steps, 3)
rmse = np.sqrt((err**2).mean())
rmse_xyz = np.sqrt((err**2).mean(axis=0))

os.makedirs("docs", exist_ok=True)

# ---- Figure 1: 3D desired vs actual, with RMSE in the title ----
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot(desired[:, 0], desired[:, 1], desired[:, 2],
        "b--", lw=2.2, label="desired (reference)")
ax.plot(actual[:, 0], actual[:, 1], actual[:, 2],
        "r-", lw=1.5, label="actual (Singha flight)")
ax.scatter(actual[0, 0], actual[0, 1], actual[0, 2],
           color="green", s=90, marker="*", label="start")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
ax.set_title(f"Spiral tracking: desired vs actual (ω={OMEGA}, position RMSE={rmse:.2f} m)")
ax.legend(loc="upper right")
ax.view_init(elev=12, azim=-72)
fig.savefig("docs/actual_vs_desired_3d_clear.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: position over time (desired vs actual), 3 panels ----
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
labels = ["x (m)", "y (m)", "z (m)"]
for i, ax in enumerate(axes):
    ax.plot(T, desired[:, i], "b--", lw=2, label="desired")
    ax.plot(T, actual[:, i], "r-", lw=1.3, label="actual")
    ax.set_ylabel(labels[i]); ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(loc="upper right")
axes[-1].set_xlabel("time (s)")
axes[0].set_title(f"Desired vs actual position over time (flight {FLIGHT})")
fig.tight_layout()
fig.savefig("docs/actual_vs_desired_time_clear.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ---- Figure 3: tracking error over time (this is what makes the gap clear) ----
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(T, err[:, 0], "r-", lw=1.3, label=f"error x (RMSE {rmse_xyz[0]:.2f} m)")
ax.plot(T, err[:, 1], "g-", lw=1.3, label=f"error y (RMSE {rmse_xyz[1]:.2f} m)")
ax.plot(T, err[:, 2], "b-", lw=1.3, label=f"error z (RMSE {rmse_xyz[2]:.2f} m)")
ax.axhline(0, color="grey", lw=0.8)
ax.set_xlabel("time (s)"); ax.set_ylabel("actual − desired (m)")
ax.set_title(f"Controller tracking error over time (flight {FLIGHT})")
ax.legend(loc="upper right"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("docs/actual_vs_desired_error_clear.png", dpi=140, bbox_inches="tight")
plt.close(fig)

print("Saved 3 figures: actual_vs_desired_{3d,time,error}_clear.png")
print(f"Position RMSE (x,y,z): {rmse_xyz[0]:.3f}, {rmse_xyz[1]:.3f}, {rmse_xyz[2]:.3f} m  |  overall {rmse:.3f} m")