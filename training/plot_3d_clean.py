"""3D path for the CLEAN controlled model."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from models.pinn_observer import PINNObserver

NOISE_STD = 0.02; TRAJ_IDX = 0

data = np.load("datasets/setpoint_clean.npz")
T = data["T"]; X = data["X"]
x_true = X[TRAJ_IDX]
rng = np.random.default_rng(0)
y_noisy = x_true + rng.normal(0, NOISE_STD, x_true.shape)
t_t = torch.tensor(T.reshape(-1,1), dtype=torch.float32)
y_t = torch.tensor(y_noisy, dtype=torch.float32)

model = PINNObserver()
model.load_state_dict(torch.load("models/pinn_clean.pth"))
model.eval()
with torch.no_grad():
    x_est = model(t_t, y_t).numpy()

xt,yt,zt = x_true[:,0], x_true[:,1], x_true[:,2]
xe,ye,ze = x_est[:,0], x_est[:,1], x_est[:,2]

fig = plt.figure(figsize=(11,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xt,yt,zt,'g-',linewidth=2.5,label="True path")
ax.plot(xe,ye,ze,'b--',linewidth=1.8,label="Estimated path")
ax.scatter(xt[0],yt[0],zt[0],color='black',s=80,marker='o',label="Start")
ax.scatter(xt[-1],yt[-1],zt[-1],color='red',s=80,marker='X',label="True end")
ax.scatter(0,0,2.0,color='green',s=120,marker='*',label="Setpoint (z=2)")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
ax.set_title("Phase 2 (clean): 3D flight path — true vs estimate")
ax.legend()
os.makedirs("docs", exist_ok=True)
plt.savefig("docs/trajectory_3d_clean.png", dpi=120, bbox_inches="tight")
print("Saved -> docs/trajectory_3d_clean.png")