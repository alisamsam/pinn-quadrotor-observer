"""All 12 states: v1 (2x64) vs v2 (3x128) vs v3 (3x256)."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib.pyplot as plt
from models.pinn_observer import PINNObserver

NOISE_STD = 0.02; TRAJ_IDX = 0
STATE_INFO = [("x","m"),("y","m"),("z","m"),
    ("x_dot","m/s"),("y_dot","m/s"),("z_dot","m/s"),
    ("phi","rad"),("theta","rad"),("psi (yaw)","rad"),
    ("phi_dot","rad/s"),("theta_dot","rad/s"),("psi_dot","rad/s")]

data = np.load("datasets/singha_dataset.npz")
T = data["T"]; X = data["X"]; x_true = X[TRAJ_IDX]
rng = np.random.default_rng(0)
y_noisy = x_true + rng.normal(0, NOISE_STD, x_true.shape)
t_t = torch.tensor(T.reshape(-1,1), dtype=torch.float32)
y_t = torch.tensor(y_noisy, dtype=torch.float32)

def run(path,h,nl):
    m=PINNObserver(hidden=h,n_hidden_layers=nl); m.load_state_dict(torch.load(path)); m.eval()
    with torch.no_grad(): return m(t_t,y_t).numpy()

e1=run("models/pinn_singha.pth",64,2)
e2=run("models/pinn_singha_v2.pth",128,3)
e3=run("models/pinn_singha_v3.pth",256,3)

fig,axes=plt.subplots(4,3,figsize=(15,12)); axes=axes.flatten()
for i,(name,unit) in enumerate(STATE_INFO):
    ax=axes[i]
    ax.plot(T,x_true[:,i],'g-',lw=2.2,label="True")
    ax.plot(T,e1[:,i],'r--',lw=0.9,alpha=0.55,label="v1 (2x64)")
    ax.plot(T,e2[:,i],color='orange',ls='--',lw=0.9,alpha=0.7,label="v2 (3x128)")
    ax.plot(T,e3[:,i],'b-',lw=1.2,label="v3 (3x256)")
    ax.set_title(f"{name} [{unit}]",fontsize=11); ax.grid(alpha=0.3)
    if i==0: ax.legend(fontsize=7)
fig.suptitle("All 12 states: capacity comparison (v1 / v2 / v3) on Singha scenario",fontsize=15)
fig.tight_layout(rect=[0,0,1,0.98])
os.makedirs("docs",exist_ok=True)
plt.savefig("docs/compare_all_three.png",dpi=110,bbox_inches="tight")
print("Saved -> docs/compare_all_three.png")