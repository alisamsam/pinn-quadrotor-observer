"""Consolidated observer comparison: PINN vs Luenberger vs EKF vs UKF."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

obs = ["PINN", "Luenberger", "EKF", "UKF"]
# (measured, hidden) per observer, per condition
CLEAN   = {"PINN":(0.0442,0.0559),"Luenberger":(0.0100,0.0094),"EKF":(0.0000,0.0080),"UKF":(0.0000,0.0129)}
STRONG  = {"PINN":(0.0850,0.0720),"Luenberger":(0.0183,0.0368),"EKF":(0.0110,0.0174),"UKF":(0.0113,0.0198)}
MISMATCH= {"PINN":(0.3384,0.6963),"Luenberger":(0.0189,0.2457),"EKF":(0.0006,0.6769),"UKF":(0.0006,0.6767)}
panels = [("Clean", CLEAN), ("Strong noise (0.10 m / 0.02 rad)", STRONG),
          ("Combined-worst mismatch", MISMATCH)]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
x = np.arange(len(obs)); w = 0.38
for ax,(title,data) in zip(axes, panels):
    meas = [data[o][0] for o in obs]; hid = [data[o][1] for o in obs]
    b1 = ax.bar(x-w/2, meas, w, label="measured", color="#1C7293")
    b2 = ax.bar(x+w/2, hid,  w, label="hidden",   color="#C21E7A")
    for b in list(b1)+list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(obs, fontsize=9)
    ax.set_title(title, fontsize=11); ax.set_ylabel("test RMSE"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(meas),max(hid))*1.35)
    ax.legend(fontsize=8)
fig.suptitle("Observer comparison on the spiral (10 unseen flights): PINN vs classical observers\n"
             "Classical observers use the live measurement stream; the PINN uses only time + initial condition.",
             fontsize=13)
plt.tight_layout(rect=[0,0,1,0.93])
plt.savefig("docs/benchmark_comparison.png", dpi=140, bbox_inches="tight")
print("saved -> docs/benchmark_comparison.png")
