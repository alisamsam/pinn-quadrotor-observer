import csv, os
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

rows=[]
with open("docs/eig_analysis_phase5.csv") as fp:
    r=csv.reader(fp); header=next(r)
    for line in r: rows.append(line)

wb=openpyxl.Workbook(); ws=wb.active; ws.title="Eigenvalue stability"
title="Stability test of learned gain L  -  eig(A - L C)  -  Phase 5 model, unseen spiral flight 45"
ws["A1"]=title; ws["A1"].font=Font(name="Arial",bold=True,size=12)
ws.merge_cells("A1:E1")

hdr=["time (s)","max real part","min real part","max |imag|","verdict"]
for j,h in enumerate(hdr,1):
    c=ws.cell(row=3,column=j,value=h)
    c.font=Font(name="Arial",bold=True,color="FFFFFF")
    c.fill=PatternFill("solid",fgColor="21295C")
    c.alignment=Alignment(horizontal="center")

red=PatternFill("solid",fgColor="F8D7DA"); green=PatternFill("solid",fgColor="D4EDDA")
thin=Side(style="thin",color="CCCCCC"); border=Border(left=thin,right=thin,top=thin,bottom=thin)
for i,row in enumerate(rows,start=4):
    t,mx,mn,im,verd=row
    vals=[round(float(t),3),round(float(mx),4),round(float(mn),4),round(float(im),4),verd]
    for j,v in enumerate(vals,1):
        c=ws.cell(row=i,column=j,value=v)
        c.font=Font(name="Arial"); c.border=border; c.alignment=Alignment(horizontal="center")
        if j==5: c.fill = red if verd=="UNSTABLE" else green

# summary block
n=len(rows); mxs=[float(r[1]) for r in rows]; unstable=sum(1 for r in rows if r[4]=="UNSTABLE")
srow=4+n+1
summ=[("Points sampled",n),("Largest max-real-part over flight",round(max(mxs),4)),
      ("Smallest max-real-part over flight",round(min(mxs),4)),
      ("Fraction unstable (max Re > 0)",f"{unstable/n*100:.1f}%"),
      ("Interpretation","max Re > 0 everywhere => learned L is NOT a stabilizing observer gain")]
for k,(lab,val) in enumerate(summ):
    ws.cell(row=srow+k,column=1,value=lab).font=Font(name="Arial",bold=True)
    ws.cell(row=srow+k,column=2,value=val).font=Font(name="Arial")

widths=[10,16,16,14,44]
for j,w in enumerate(widths,1): ws.column_dimensions[chr(64+j)].width=w
ws.freeze_panes="A4"

wb.save("docs/eig_analysis_phase5.xlsx")
print("wrote docs/eig_analysis_phase5.xlsx  with",n,"data rows")