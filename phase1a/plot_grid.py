import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
with open("docs/grid_study.csv") as fp:
    for line in csv.DictReader(fp):
        rows.append(line)

layers_all  = sorted({int(x["layers"])  for x in rows})
neurons_all = sorted({int(x["neurons"]) for x in rows})
hid = {(int(x["layers"]), int(x["neurons"])): float(x["rmse_hidden"]) for x in rows}

os.makedirs("docs", exist_ok=True)

# grouped bar chart, log y-axis
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.8 / len(layers_all)
x = np.arange(len(neurons_all))
colors = ["#065A82", "#1C7293", "#C0392B"]
for i, L in enumerate(layers_all):
    heights = [hid.get((L, N), np.nan) for N in neurons_all]
    bars = ax.bar(x + i*width, heights, width, label=f"{L} layers", color=colors[i % 3])
    for b, h in zip(bars, heights):
        if not np.isnan(h):
            ax.text(b.get_x()+b.get_width()/2, h*1.05, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8, rotation=90)
ax.set_yscale("log")
ax.set_xlabel("Neurons per layer")
ax.set_ylabel("Hidden-state RMSE  (log scale)")
ax.set_title("Architecture grid: hidden RMSE vs width and depth (neutral weights)")
ax.set_xticks(x + width*(len(layers_all)-1)/2)
ax.set_xticklabels(neurons_all)
ax.legend(title="Depth")
ax.grid(axis="y", alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("docs/grid_bar.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# heatmap, log colour
M = np.full((len(layers_all), len(neurons_all)), np.nan)
for i, L in enumerate(layers_all):
    for j, N in enumerate(neurons_all):
        M[i, j] = hid.get((L, N), np.nan)
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(M, cmap="viridis_r", aspect="auto", norm=matplotlib.colors.LogNorm())
ax.set_xticks(range(len(neurons_all))); ax.set_xticklabels(neurons_all)
ax.set_yticks(range(len(layers_all)));  ax.set_yticklabels(layers_all)
ax.set_xlabel("Neurons per layer"); ax.set_ylabel("Layers")
ax.set_title("Hidden-state RMSE (log colour scale)")
for i in range(len(layers_all)):
    for j in range(len(neurons_all)):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", color="white", fontsize=9)
fig.colorbar(im, ax=ax, label="hidden RMSE")
fig.tight_layout()
fig.savefig("docs/grid_heatmap.png", dpi=140, bbox_inches="tight")
plt.close(fig)

print("Saved docs/grid_bar.png and docs/grid_heatmap.png")