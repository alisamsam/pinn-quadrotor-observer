"""Reproduce the architecture-grid histogram (grouped bars by depth, log-scale y).
Usage: python docs/make_arch_histogram.py [grid_csv] [out_png] [title]
Default CSV = docs/grid_study_v2.csv  (the NEW controller grid).
"""
import sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV   = sys.argv[1] if len(sys.argv) > 1 else "docs/grid_study_v2.csv"
OUT   = sys.argv[2] if len(sys.argv) > 2 else "docs/arch_grid_hist_v2.png"
TITLE = sys.argv[3] if len(sys.argv) > 3 else \
        "Architecture grid: hidden RMSE vs width and depth (neutral weights)"

# read grid csv -> {layers: {neurons: rmse_hidden}}
data = {}
with open(CSV) as fp:
    for r in csv.DictReader(fp):
        L, H = int(r["layers"]), int(r["neurons"])
        data.setdefault(L, {})[H] = float(r["rmse_hidden"])

LAYERS  = sorted(data)                                   # e.g. [4, 9, 12]
NEURONS = sorted({h for d in data.values() for h in d})  # [20, 60, 100, 128]
COLORS  = {4: "#0F5A73", 9: "#2E86AB", 12: "#C0392B"}    # dark teal / teal / red
FALLBACK = ["#0F5A73", "#2E86AB", "#C0392B", "#8E44AD"]

x = np.arange(len(NEURONS))
n = len(LAYERS)
w = 0.8 / n

fig, ax = plt.subplots(figsize=(11, 6.2))
for i, L in enumerate(LAYERS):
    vals = [data[L].get(H, np.nan) for H in NEURONS]
    off  = (i - (n - 1) / 2) * w
    color = COLORS.get(L, FALLBACK[i % len(FALLBACK)])
    bars = ax.bar(x + off, vals, w, label=f"{L} layers", color=color,
                  edgecolor="white", linewidth=0.6)
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width()/2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, rotation=90)

ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(NEURONS)
ax.set_xlabel("Neurons per layer", fontsize=11)
ax.set_ylabel("Hidden-state RMSE  (log scale)", fontsize=11)
ax.set_title(TITLE, fontsize=12)
ax.grid(axis="y", which="both", ls="-", lw=0.4, alpha=0.35)
ax.set_axisbelow(True)
ax.legend(title="Depth", loc="upper left", framealpha=0.95)
ax.margins(y=0.18)
fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print("saved", OUT)
