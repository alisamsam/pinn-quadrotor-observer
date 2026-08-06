import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVS = [-20, -10, 0, 10, 20]
FILES = {
    "mass":    "docs/mismatch_mass_unaware.csv",
    "inertia": "docs/mismatch_inertia_unaware.csv",
    "arm":     "docs/mismatch_arm_unaware.csv",
}
COMBINED = {"meas": 3.0034, "hidden": 3.2302}   # measured worst case, all three stacked
COLORS = {"mass": "#065A82", "inertia": "#E08E0B", "arm": "#C0392B"}


def load(path):
    out = {}
    with open(path) as fp:
        for r in csv.DictReader(fp):
            keys = list(r.keys())
            dev = int(round(float(r[keys[1]].strip('%'))))   # 2nd column is the deviation
            out[dev] = (float(r["test_meas_RMSE"]), float(r["test_hidden_RMSE"]))
    return out


data = {p: load(f) for p, f in FILES.items()}
os.makedirs("docs", exist_ok=True)
params = ["mass", "inertia", "arm"]

# ===== CHART 1: grouped by parameter, 5 deviation bars each (hidden RMSE, log) =====
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(params))
width = 0.16
all_h = []
for i, dev in enumerate(DEVS):
    heights = [data[p][dev][1] for p in params]
    all_h += heights
    bars = ax.bar(x + (i-2)*width, heights, width,
                  label=f"{dev:+d}%" if dev != 0 else "nominal",
                  color=plt.cm.viridis(i/len(DEVS)))
    for b, h in zip(bars, heights):
        ax.text(b.get_x()+b.get_width()/2, h*1.05, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_yscale("log")
ax.set_ylim(top=max(all_h) * 2.2)          # headroom so rotated labels are not clipped
ax.set_xticks(x); ax.set_xticklabels([p.capitalize() for p in params])
ax.set_ylabel("Hidden-state RMSE (log scale)")
ax.set_xlabel("Perturbed parameter")
ax.set_title("Observer sensitivity to parameter mismatch (controller unaware)")
ax.legend(title="Deviation", ncol=5, fontsize=9, loc="upper center")
ax.grid(axis="y", alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("docs/mismatch_by_parameter.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ===== CHART 2: worst-per-parameter + combined (hidden RMSE, log) =====
worst = {}
for p in params:
    vals = {d: data[p][d][1] for d in DEVS if d != 0}
    wd = max(vals, key=vals.get)
    worst[p] = (wd, vals[wd])
nominal = data["mass"][0][1]

labels = [f"Mass\n({worst['mass'][0]:+d}%)",
          f"Inertia\n({worst['inertia'][0]:+d}%)",
          f"Arm\n({worst['arm'][0]:+d}%)",
          "Combined\nworst"]
heights = [worst["mass"][1], worst["inertia"][1], worst["arm"][1], COMBINED["hidden"]]
colors  = [COLORS["mass"], COLORS["inertia"], COLORS["arm"], "#21295C"]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(labels, heights, color=colors, width=0.6)
for b, h in zip(bars, heights):
    ax.text(b.get_x()+b.get_width()/2, h*1.04, f"{h:.2f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.axhline(nominal, color="grey", ls="--", lw=1.5, label=f"nominal ({nominal:.3f})")
ax.set_yscale("log")
ax.set_ylim(top=max(heights) * 1.8)        # headroom for the bold labels
ax.set_ylabel("Hidden-state RMSE (log scale)")
ax.set_title("Worst-case observer degradation per parameter, and combined")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("docs/mismatch_worst_case.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ===== TABLE: full CSV summary =====
with open("docs/mismatch_summary_table.csv", "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["parameter", "metric", "-20%", "-10%", "nominal", "+10%", "+20%"])
    for p in params:
        w.writerow([p, "measured RMSE"] + [f"{data[p][d][0]:.4f}" for d in DEVS])
        w.writerow([p, "hidden RMSE"]   + [f"{data[p][d][1]:.4f}" for d in DEVS])
    w.writerow(["combined worst", "measured RMSE", "", "", f"{data['mass'][0][0]:.4f}", "", f"{COMBINED['meas']:.4f}"])
    w.writerow(["combined worst", "hidden RMSE",   "", "", f"{data['mass'][0][1]:.4f}", "", f"{COMBINED['hidden']:.4f}"])

print("Saved docs/mismatch_by_parameter.png, docs/mismatch_worst_case.png, docs/mismatch_summary_table.csv")
print("Worst directions:", {p: worst[p] for p in params})