import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOISE = 0.009   # run-to-run std from the seed study

def load(path):
    rows = []
    with open(path) as fp:
        for r in csv.DictReader(fp):
            rows.append(r)
    return rows

def make_chart(path, title, outfile):
    rows = load(path)
    cases   = [f"C{r['case']}\n({r['w0']},{r['w_ode']},{r['wy']})" for r in rows]
    hidden  = [float(r["test_hidden_RMSE"]) for r in rows]
    measured= [float(r["test_meas_RMSE"])   for r in rows]

    best = min(hidden)
    x = np.arange(len(rows))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - width/2, measured, width, label="measured RMSE", color="#E08E0B")  # amber
    b2 = ax.bar(x + width/2, hidden,   width, label="hidden RMSE",   color="#0B3C5D")  # deep navy

    # noise band around the best hidden value
    ax.axhspan(best, best + NOISE, color="grey", alpha=0.22,
               label=f"±{NOISE:.3f} noise band above best")

    # numeric labels on BOTH bar groups
    for b, v in zip(b1, measured):
        ax.text(b.get_x()+b.get_width()/2, v+0.001, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8, rotation=90, color="#8A5300")
    for b, v in zip(b2, hidden):
        ax.text(b.get_x()+b.get_width()/2, v+0.001, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8, rotation=90, color="#0B3C5D")

    ax.set_xticks(x); ax.set_xticklabels(cases, fontsize=8)
    ax.set_xlabel("Weight case  (w0, w_ode, wy)")
    ax.set_ylabel("RMSE on unseen flights")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    # a little headroom so the rotated labels are not clipped
    ax.set_ylim(0, max(max(hidden), max(measured)) * 1.25)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")

os.makedirs("docs", exist_ok=True)
make_chart("docs/weight_study.csv",
           "Loss-weight sensitivity at 4x128 (retained architecture)",
           "docs/weight_bar_4x128.png")
make_chart("docs/weight_study_4x100.csv",
           "Loss-weight sensitivity at 4x100",
           "docs/weight_bar_4x100.png")