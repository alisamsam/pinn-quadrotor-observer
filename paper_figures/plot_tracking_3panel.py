# -*- coding: utf-8 -*-
# Combined 3-panel controller-tracking figure: circle | figure-8 | spiral.
# Reads the three fig_*_controller_tracking.csv files in paper_figures/.
# Writes paper_figures/controller_tracking_3panel.pdf/.png
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

plt.rcParams.update({"font.size": 10, "savefig.bbox": "tight", "pdf.fonttype": 42})

panels = [
    ("fig_circle_controller_tracking.csv",  "(a) Circle",     (12, -72)),
    ("fig_figure8_controller_tracking.csv", "(b) Figure-eight", (20, -60)),
    ("fig_spiral_controller_tracking.csv",  "(c) Spiral",     (18, -60)),
]

fig = plt.figure(figsize=(15, 5))
for i, (fname, title, (elev, azim)) in enumerate(panels):
    d = np.genfromtxt("paper_figures/" + fname, delimiter=",", names=True)
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.plot(d["x_des"], d["y_des"], d["z_des"], "b-", lw=2.0, label="desired")
    ax.plot(d["x_act"], d["y_act"], d["z_act"], "r--", lw=1.4, label="actual")
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.set_zlabel("z (m)", fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.view_init(elev=elev, azim=azim)
    if i == 0:
        ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig("paper_figures/controller_tracking_3panel.pdf")
fig.savefig("paper_figures/controller_tracking_3panel.png", dpi=300)
print("saved -> paper_figures/controller_tracking_3panel.pdf / .png")
