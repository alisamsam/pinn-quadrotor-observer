import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from spiral_reference import spiral_reference

FLIGHT = 0                      # which flight to show (0-49)
d = np.load("datasets/spiral_dataset.npz")
T, X = d["T"], d["X"]
actual = X[FLIGHT]              # (steps, 12): the real Singha flight
# desired path at the same time instants
desired = np.array([spiral_reference(t)[0] for t in T])   # (steps, 3)

os.makedirs("docs", exist_ok=True)

# ---- Figure 1: 3D desired vs actual ----
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot(desired[:,0], desired[:,1], desired[:,2], "b--", lw=2, label="desired (reference)")
ax.plot(actual[:,0],  actual[:,1],  actual[:,2],  "r-",  lw=1.5, label="actual (Singha flight)")
ax.scatter(actual[0,0], actual[0,1], actual[0,2], color="green", s=90, marker="*", label="start")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
ax.set_title(f"Spiral tracking: desired vs actual (flight {FLIGHT})")
ax.legend()
ax.view_init(elev=12, azim=-72)
fig.savefig("docs/actual_vs_desired_3d_4x180.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: x, y, z vs time ----
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
labels = ["x (m)", "y (m)", "z (m)"]
for i, ax in enumerate(axes):
    ax.plot(T, desired[:,i], "b--", lw=2, label="desired")
    ax.plot(T, actual[:,i],  "r-",  lw=1.3, label="actual")
    ax.set_ylabel(labels[i]); ax.grid(alpha=0.3)
    if i == 0: ax.legend(loc="upper right")
axes[-1].set_xlabel("time (s)")
axes[0].set_title(f"Desired vs actual position over time(flight {FLIGHT})")
fig.tight_layout()
fig.savefig("docs/actual_vs_desired_time_4x180.png", dpi=140, bbox_inches="tight")
plt.close(fig)

print("Saved docs/actual_vs_desired_3d_4x180.png and docs/actual_vs_desired_time_4x180.png")