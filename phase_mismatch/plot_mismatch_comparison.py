import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOISE = 0.009

def load(path):
    devs, hid = [], []
    with open(path) as fp:
        for r in csv.DictReader(fp):
            # deviation stored like "+10%" -> 10
            key = [k for k in r if "dev" in k][0]
            devs.append(int(r[key].replace("%","").replace("+","")))
            hid.append(float(r["test_hidden_RMSE"]))
    order = np.argsort(devs)
    return np.array(devs)[order], np.array(hid)[order]

mass_d,   mass_h   = load("docs/mismatch_mass_unaware.csv")
inert_d,  inert_h  = load("docs/mismatch_inertia_unaware.csv")
arm_d,    arm_h    = load("docs/mismatch_arm_unaware.csv")

os.makedirs("docs", exist_ok=True)
fig, ax = plt.subplots(figsize=(11, 6.5))

ax.plot(mass_d,  mass_h,  "o-", color="#0B3C5D", lw=2, ms=7, label="mass")
ax.plot(inert_d, inert_h, "s-", color="#E08E0B", lw=2, ms=7, label="inertia (Ix,Iy,Iz)")
ax.plot(arm_d,   arm_h,   "^-", color="#1C7293", lw=2, ms=7, label="arm length")

# combined worst-case as a reference line + annotation
COMBINED = 3.2302
ax.axhline(COMBINED, color="#C0392B", ls="--", lw=1.6,
           label=f"combined worst-case = {COMBINED:.2f}")
ax.annotate("all three at worst, simultaneously",
            xy=(0, COMBINED), xytext=(-18, COMBINED*1.15),
            fontsize=9, color="#C0392B")

# noise band near nominal
ax.axhspan(0, NOISE, color="grey", alpha=0.25, label=f"±{NOISE:.3f} noise band")

ax.set_yscale("log")
ax.set_xlabel("Parameter deviation from nominal (%)")
ax.set_ylabel("Hidden-state RMSE  (log scale)")
ax.set_title("Observer sensitivity to model mismatch (controller-unaware, common scale)")
ax.set_xticks([-20, -10, 0, 10, 20])
ax.grid(alpha=0.3, which="both")
ax.legend(loc="upper center", fontsize=9, ncol=2)
fig.tight_layout()
fig.savefig("docs/mismatch_comparison.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("Saved docs/mismatch_comparison.png")