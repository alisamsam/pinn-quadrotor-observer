"""Observer robustness to model-parameter mismatch (nominal-trained observer, evaluated on
perturbed plants). Mass / inertia / arm-length swept from -20% to +20%, plus a combined-worst
scenario. House style: amber = measured, navy = hidden. Outputs into this folder.
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

HERE = os.path.dirname(os.path.abspath(__file__))
AMBER, NAVY = "#E08E0B", "#0B3C5D"
devs = [-20, -10, 0, 10, 20]

data = {
 "Mass":        {"meas":[0.1509,0.0892,0.0436,0.0679,0.1068], "hid":[0.3461,0.1758,0.0552,0.0840,0.1119]},
 "Inertia":     {"meas":[0.0546,0.0462,0.0436,0.0615,0.1005], "hid":[0.1125,0.0823,0.0552,0.1424,0.3069]},
 "Arm length":  {"meas":[0.1239,0.0650,0.0436,0.0454,0.0519], "hid":[0.3835,0.1598,0.0552,0.0783,0.1034]},
}
nominal = {"meas":0.0436, "hid":0.0552}
worst   = {"meas":0.3384, "hid":0.6963}

# ---- figure: 2x2 (three sweeps + combined-worst bar) ----
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
panels = [("Mass", axes[0,0]), ("Inertia", axes[0,1]), ("Arm length", axes[1,0])]
for name, a in panels:
    a.plot(devs, data[name]["meas"], "--s", color=AMBER, lw=1.8, ms=5, label="measured states")
    a.plot(devs, data[name]["hid"],  "-o",  color=NAVY,  lw=2.0, ms=5, label="hidden states")
    a.axvline(0, color="#bbb", lw=1, ls=":")
    a.set_title(f"{name} mismatch", fontsize=12)
    a.set_xlabel("Parameter deviation (%)", fontsize=10)
    a.set_ylabel("Test RMSE  (per-state units)", fontsize=10)
    a.grid(alpha=0.3); a.set_axisbelow(True); a.legend(fontsize=9)

# combined-worst bar panel
a = axes[1,1]
x = np.arange(2); wbar = 0.36
b1 = a.bar(x-wbar/2, [nominal["meas"], nominal["hid"]], wbar, label="nominal (no mismatch)", color="#7F8C8D")
b2 = a.bar(x+wbar/2, [worst["meas"],   worst["hid"]],   wbar, label="combined-worst mismatch", color="#C0392B")
for b in list(b1)+list(b2):
    a.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{b.get_height():.3f}",
           ha="center", va="bottom", fontsize=9)
a.set_xticks(x); a.set_xticklabels(["measured", "hidden"])
a.set_title("Combined-worst mismatch", fontsize=12)
a.set_ylabel("Test RMSE  (per-state units)", fontsize=10)
a.grid(axis="y", alpha=0.3); a.set_axisbelow(True); a.legend(fontsize=9); a.set_ylim(0, 0.8)

fig.suptitle("Observer robustness to model-parameter mismatch (nominal-trained observer on perturbed plants)",
             fontsize=14)
fig.tight_layout(rect=[0,0,1,0.97])
PNG = os.path.join(HERE, "mismatch_robustness.png")
fig.savefig(PNG, dpi=140, bbox_inches="tight"); plt.close(fig)
print("saved", PNG)

# ---- CSV ----
cols = ["-20%","-10%","nominal","+10%","+20%"]
with open(os.path.join(HERE, "mismatch_v2_summary.csv"), "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["parameter","metric"]+cols)
    for p in data:
        w.writerow([p,"measured RMSE"]+[f"{x:.4f}" for x in data[p]["meas"]])
        w.writerow([p,"hidden RMSE"]+[f"{x:.4f}" for x in data[p]["hid"]])
    w.writerow(["combined worst","measured RMSE","","",f"{nominal['meas']:.4f}","",f"{worst['meas']:.4f}"])
    w.writerow(["combined worst","hidden RMSE","","",f"{nominal['hid']:.4f}","",f"{worst['hid']:.4f}"])
print("saved mismatch_v2_summary.csv")

# ---- Excel ----
wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Mismatch robustness"
WHITE = Font(bold=True, color="FFFFFF"); HEAD = PatternFill("solid", fgColor="1F2A5A")
nomfill = PatternFill("solid", fgColor="EAF1E6"); worstfill = PatternFill("solid", fgColor="F5D6D2")
thin = Side(style="thin", color="C9D2D9"); bd = Border(thin,thin,thin,thin)
ws["A1"] = "Observer robustness to model-parameter mismatch (unseen test flights)"; ws["A1"].font = Font(bold=True, size=14)
ws["A2"] = ("Nominal-trained 4x100 observer evaluated on plants with perturbed mass / inertia / arm length. "
            "Combined-worst = each parameter set to its individually worst direction at 20%.")
ws["A2"].font = Font(italic=True, color="555555")
hdr = ["Parameter","Metric"]+cols
for c,h in enumerate(hdr, start=1):
    cell = ws.cell(row=4, column=c, value=h); cell.font = WHITE; cell.fill = HEAD
    cell.alignment = Alignment(horizontal="center"); cell.border = bd
r = 5
for p in data:
    for key,lab in [("meas","measured RMSE"),("hid","hidden RMSE")]:
        vals = data[p][key]
        ws.cell(row=r, column=1, value=p); ws.cell(row=r, column=2, value=lab)
        for j,v in enumerate(vals):
            cell = ws.cell(row=r, column=3+j, value=v); cell.number_format = "0.0000"
            if devs[j] == 0: cell.fill = nomfill
            if v == max(vals): cell.font = Font(bold=True, color="C0392B")
        for c in range(1, 8):
            ws.cell(row=r, column=c).alignment = Alignment(horizontal="center"); ws.cell(row=r, column=c).border = bd
        r += 1
# combined-worst rows
for key,lab,wv,nv in [("meas","measured RMSE",worst["meas"],nominal["meas"]),
                      ("hid","hidden RMSE",worst["hid"],nominal["hid"])]:
    ws.cell(row=r, column=1, value="combined worst"); ws.cell(row=r, column=2, value=lab)
    ws.cell(row=r, column=5, value=nv).number_format = "0.0000"; ws.cell(row=r, column=5).fill = nomfill
    ws.cell(row=r, column=7, value=wv).number_format = "0.0000"; ws.cell(row=r, column=7).fill = worstfill
    ws.cell(row=r, column=7).font = Font(bold=True, color="C0392B")
    for c in range(1, 8):
        ws.cell(row=r, column=c).alignment = Alignment(horizontal="center"); ws.cell(row=r, column=c).border = bd
    r += 1
ws.cell(row=r+1, column=1, value=("Reading: error stays lowest at nominal (0%) and grows with deviation; the observer "
                                  "degrades gracefully (no divergence). Hidden states are the most sensitive, and the "
                                  "combined-worst case is the hardest but remains bounded."))
ws.cell(row=r+1, column=1).font = Font(italic=True, color="666666")
widths = {"A":16,"B":15,"C":10,"D":10,"E":10,"F":10,"G":10}
for col,wd in widths.items(): ws.column_dimensions[col].width = wd
ws.freeze_panes = "C5"
XLSX = os.path.join(HERE, "mismatch_v2_table.xlsx")
wb.save(XLSX); print("saved", XLSX)
