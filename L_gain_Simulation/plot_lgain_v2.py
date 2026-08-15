"""Effect of the learned correction gain L on observer accuracy.
Averaged measured / hidden RMSE on the unseen test flights, L-OFF vs L-ON.
L-OFF = plain PINN observer; L-ON = PINN + learned Farkane-style gain L*(y - yhat).
House style; outputs into this folder.
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

HERE = os.path.dirname(os.path.abspath(__file__))
OFF, ON = "#1C6FB0", "#C0392B"   # without L (blue) / with L (red)

# aggregate test-set RMSE from the v2 gain-L run
meas_off, hid_off = 0.0442, 0.0559     # plain PINN observer (L-OFF)
meas_on,  hid_on  = 0.0958, 0.3430     # PINN + learned gain L (L-ON)

groups = ["measured states", "hidden states"]
off = [meas_off, hid_off]; on = [meas_on, hid_on]
x = np.arange(2); wbar = 0.36
fig, ax = plt.subplots(figsize=(8.5, 5.6))
b1 = ax.bar(x-wbar/2, off, wbar, label="without L", color=OFF, edgecolor="white", linewidth=0.6)
b2 = ax.bar(x+wbar/2, on,  wbar, label="with L",    color=ON,  edgecolor="white", linewidth=0.6)
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.004, f"{b.get_height():.3f}",
            ha="center", va="bottom", fontsize=10)
# annotate the blow-up on hidden states
ax.annotate(f"x{hid_on/hid_off:.1f} worse", (x[1]+wbar/2, hid_on), xytext=(x[1]+wbar/2, hid_on+0.05),
            ha="center", fontsize=10, fontweight="bold", color=ON)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=11)
ax.set_xlabel("State group", fontsize=11)
ax.set_ylabel("Average RMSE  (per-state units: m, m/s, rad, rad/s)", fontsize=11)
ax.set_title("Effect of the learned correction gain L on observer accuracy", fontsize=13)
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
ax.legend(loc="upper left"); ax.set_ylim(0, 0.40)
plt.tight_layout()
PNG = os.path.join(HERE, "lgain_effect.png")
plt.savefig(PNG, dpi=140, bbox_inches="tight"); plt.close(fig)
print("saved", PNG)

# ---- CSV ----
with open(os.path.join(HERE, "lgain_v2.csv"), "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["config","avg_measured_RMSE","avg_hidden_RMSE"])
    w.writerow(["without L (plain PINN)", meas_off, hid_off])
    w.writerow(["with L (learned gain)",  meas_on,  hid_on])

# ---- Excel ----
wb = openpyxl.Workbook(); ws = wb.active; ws.title = "L-gain effect"
WHITE = Font(bold=True, color="FFFFFF"); HEAD = PatternFill("solid", fgColor="1F2A5A")
good = PatternFill("solid", fgColor="4E7D3A"); bad = PatternFill("solid", fgColor="C0392B")
thin = Side(style="thin", color="C9D2D9"); border = Border(thin,thin,thin,thin)
ws["A1"] = "Effect of the learned correction gain L (unseen test flights)"; ws["A1"].font = Font(bold=True, size=14)
ws["A2"] = ("4x100 observer, loss weights (0.5, 1.5, 1.0). L-OFF = plain PINN; "
            "L-ON = PINN + learned gain L applied as L*(y - yhat).")
ws["A2"].font = Font(italic=True, color="555555")
hdr = ["Configuration","Avg measured RMSE","Avg hidden RMSE"]
for c,h in enumerate(hdr, start=1):
    cell = ws.cell(row=4, column=c, value=h); cell.font = WHITE; cell.fill = HEAD
    cell.alignment = Alignment(horizontal="center"); cell.border = border
rows = [("without L (plain PINN)", meas_off, hid_off, good),
        ("with L (learned gain)",  meas_on,  hid_on,  bad)]
for i,(name,mr,hr,fill) in enumerate(rows):
    r = 5+i
    for cc,val in enumerate([name,mr,hr], start=1):
        cell = ws.cell(row=r, column=cc, value=val); cell.border = border
        cell.alignment = Alignment(horizontal="center")
    ws.cell(row=r, column=2).number_format = "0.0000"
    ws.cell(row=r, column=3).number_format = "0.0000"
    ws.cell(row=r, column=3).fill = fill; ws.cell(row=r, column=3).font = Font(bold=True, color="FFFFFF")
ws.cell(row=8, column=1, value=("Finding: adding the learned gain L leaves measured accuracy similar but makes the "
                                "hidden states ~6x worse (0.056 -> 0.343). The plain PINN observer is kept."))
ws.cell(row=8, column=1).font = Font(italic=True, color="666666")
for col,wd in {"A":26,"B":18,"C":16}.items(): ws.column_dimensions[col].width = wd
ws.freeze_panes = "A5"
XLSX = os.path.join(HERE, "lgain_v2_table.xlsx")
wb.save(XLSX); print("saved", XLSX)
