# -*- coding: utf-8 -*-
# Plot the FIGURE-8 figures in HD from the exported CSV.
# Reads:  paper_figures/fig_figure8_observer_states.csv
# Writes (vector PDF + 300-dpi PNG) into paper_figures/:
#   figure8_observer_3d.pdf/.png       (true vs estimated 3-D trajectory)
#   figure8_observer_12states.pdf/.png (true vs estimated, all 12 states)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})

NAMES = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r']
UNITS = ['m', 'm', 'm', 'm/s', 'm/s', 'm/s', 'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s']
MEAS = ['x', 'y', 'z', 'phi', 'theta', 'psi']


def save(fig, name):
    fig.savefig(name + ".pdf")
    fig.savefig(name + ".png", dpi=300)
    plt.close(fig)
    print("saved ->", name + ".pdf / .png")


d = np.genfromtxt("paper_figures/fig_figure8_observer_states.csv", delimiter=",", names=True)
t = d["t"]

# 3-D true vs estimated
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(d["x_true"], d["y_true"], d["z_true"], "r-", lw=2.0, label="true")
ax.plot(d["x_est"], d["y_est"], d["z_est"], "g--", lw=1.6, label="estimated")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
ax.set_title("Figure-eight: PINN observer, true vs estimated (3-D)")
ax.legend(); ax.view_init(elev=18, azim=-60)
save(fig, "paper_figures/figure8_observer_3d")

# 12-state grid
fig, axes = plt.subplots(4, 3, figsize=(15, 11))
for i, ax in enumerate(axes.flat):
    nm = NAMES[i]
    tr, es = d[nm + "_true"], d[nm + "_est"]
    rmse = float(np.sqrt(np.mean((tr - es) ** 2)))
    if nm in MEAS:
        ax.plot(t, d[nm + "_noisy"], ".", ms=1.5, color="0.7", label="noisy meas")
    ax.plot(t, tr, "r-", lw=1.2, label="true")
    ax.plot(t, es, "g--", lw=1.2, label="estimated")
    kind = "meas" if nm in MEAS else "hidden"
    ax.set_title("%s (%s)  RMSE=%.3f %s" % (nm, kind, rmse, UNITS[i]), fontsize=10)
    ax.set_xlabel("t (s)", fontsize=8); ax.set_ylabel("%s (%s)" % (nm, UNITS[i]), fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
fig.suptitle("Figure-eight: PINN observer on an unseen flight -- true vs estimated (all 12 states)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.98])
save(fig, "paper_figures/figure8_observer_12states")
print("DONE.")
