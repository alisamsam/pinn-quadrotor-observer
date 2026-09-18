"""Pipeline mind map with the Adaptive PINN-Obs architecture nested inside Stage 4."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

# contrasting palette
TEAL="#1C7293"; GREY="#6b7a86"; BLUE="#0B4F8A"; MID="#21295C"
GREEN_F="#E3F2D6"; GREEN_E="#4E9A2F"
BLUE_F="#D7E6F7"; BLUE_E="#2A6FB0"
YEL_F="#FCEFC7"; YEL_E="#C99A18"
RED_F="#F7D2D2"; RED_E="#B23A48"
GRY_F="#ECEFF1"; GRY_E="#8A97A0"

def rbox(x,y,w,h,fc,ec,lw=1.6):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.15,rounding_size=1.0",
                                fc=fc,ec=ec,lw=lw))
def arrow(x1,y1,x2,y2,color="#444",lw=1.7,ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,
                                 color=color,lw=lw,linestyle=ls,shrinkA=0,shrinkB=0))

ax.text(50,98,"PINN Quadrotor — Simulation Pipeline",ha="center",va="center",
        fontsize=20,fontweight="bold")

# ---------- overview strip ----------
ov=[("1  Desired path",TEAL),("2  Controller",TEAL),("3  Dataset (.npz)",GREY),
    ("4  PINN Observer",BLUE),("5  Results",MID)]
xs=[4,23,42,61,80]; wov=15
for (label,c),x in zip(ov,xs):
    hl = 2.6 if "PINN" in label else 1.6
    rbox(x,90,wov,5.5,c,"#111" if "PINN" in label else c,lw=hl)
    ax.text(x+wov/2,92.7,label,ha="center",va="center",color="white",
            fontsize=11,fontweight="bold")
for i in range(4):
    arrow(xs[i]+wov,92.7,xs[i+1],92.7,color="#555")
# zoom connector from Observer stage into the big panel
arrow(61+wov/2,90,50,84.5,color=BLUE,lw=1.8,ls="--")
ax.text(50,86,"open up Stage 4",ha="center",va="center",fontsize=10,
        color=BLUE,fontstyle="italic")


# ---------- big panel: Adaptive PINN-Obs ----------
rbox(2.5,10,95,73,"#FBFCFD",BLUE_E,lw=2.0)
ax.text(50,80.5,"Inside Stage 4 — Adaptive PINN-Obs (nonlinear state observer)",
        ha="center",va="center",fontsize=14,fontweight="bold",color=BLUE)

# ===== MLP column =====
rbox(5,30,33,45,GRY_F,GRY_E,lw=1.4)
ax.text(21,77,r"MLP : $\theta_\theta,\ \theta_z$",ha="center",fontsize=13,fontweight="bold")
xin,xh1,xh2,xout=9,17,25,33
yh=np.linspace(36,68,6); yo=np.linspace(34,70,7)
# edges
for y in yh: arrow(xin+0.9,52,xh1-0.9,y,color="#b8c2c9",lw=0.6) if False else ax.plot([xin,xh1],[52,y],color="#c2ccd2",lw=0.5,zorder=1)
for a in yh:
    for b in yh: ax.plot([xh1,xh2],[a,b],color="#c2ccd2",lw=0.5,zorder=1)
for a in yh:
    for b in yo: ax.plot([xh2,xout],[a,b],color="#c2ccd2",lw=0.5,zorder=1)
# nodes
ax.add_patch(Circle((xin,52),1.1,fc="white",ec="#333",zorder=3)); ax.text(xin,52,r"$t$",ha="center",va="center",fontsize=10)
for y in yh:
    ax.add_patch(Circle((xh1,y),1.1,fc="white",ec="#333",zorder=3))
    ax.add_patch(Circle((xh2,y),1.1,fc="white",ec="#333",zorder=3))
olabels=[r"$\ell_d(t)$",r"$\vdots$",r"$\ell_1(t)$",r"$\hat{x}_n(t)$",r"$\vdots$",r"$\hat{x}_2(t)$",r"$\hat{x}_1(t)$"]
for y,lab in zip(yo,olabels):
    ax.add_patch(Circle((xout,y),1.0,fc="white",ec="#333",zorder=3))
    ax.text(xout+2.2,y,lab,ha="left",va="center",fontsize=9)


# ===== AD-Com column (green) =====
rbox(40,30,13,45,GREEN_F,GREEN_E,lw=1.6)
ax.text(46.5,77,"AD-Com",ha="center",fontsize=13,fontweight="bold",color=GREEN_E)
adcom=[r"$\dot{\hat{x}}_1(t)$",r"$\dot{\hat{x}}_2(t)$",r"$\vdots$",r"$\dot{\hat{x}}_n(t)$",
       r"$L(t)$",r"$f_1(t)$",r"$\vdots$",r"$f_n(t)$"]
for y,lab in zip(np.linspace(70,34,len(adcom)),adcom):
    ax.text(46.5,y,lab,ha="center",va="center",fontsize=10)

# ===== Governing Equation column (blue) =====
rbox(55,30,22,45,BLUE_F,BLUE_E,lw=1.6)
ax.text(66,77,"Governing Equation",ha="center",fontsize=13,fontweight="bold",color=BLUE_E)
gov=[r"$\hat{x}_1(t_0)$",r"$\hat{x}_2(t_0)$",r"$\vdots$",r"$\hat{x}_n(t_0)$"]
for y,lab in zip(np.linspace(70,58,len(gov)),gov):
    ax.text(66,y,lab,ha="center",va="center",fontsize=10)
ax.text(66,48,r"$g(t,\hat{x}) = \dot{\hat{x}}(t) - f(\hat{x},t) - Bu$",
        ha="center",va="center",fontsize=10.5)
ax.text(66,43.5,r"$-\, L(t)\,C\,(x(t)-\hat{x}(t))$",
        ha="center",va="center",fontsize=10.5)

# ===== Data column (yellow) =====
rbox(80,30,15,45,YEL_F,YEL_E,lw=1.6)
ax.text(87.5,77,"Data",ha="center",fontsize=13,fontweight="bold",color="#9c7a12")
data=[r"$\hat{x}_a^1(t_0)$",r"$\hat{x}_a^2(t_0)$",r"$\vdots$",r"$\hat{x}_a^n(t_0)$",
      r"$y_1(t)$",r"$y_2(t)$",r"$\vdots$",r"$y_m(t)$"]
for y,lab in zip(np.linspace(70,34,len(data)),data):
    ax.text(87.5,y,lab,ha="center",va="center",fontsize=10)


# ===== flow arrows between blocks =====
arrow(38,52,40,52,color="#333")          # MLP -> AD-Com
arrow(53,52,55,52,color="#333")          # AD-Com -> Governing
arrow(80,52,77,52,color="#333")          # Data -> Governing

# ===== Physics-Informed Loss (red) =====
rbox(40,15,34,7,RED_F,RED_E,lw=1.8)
ax.text(57,19.4,r"$MSE = MSE_0 + MSE_g + MSE_y$",ha="center",va="center",fontsize=12)
ax.text(57,16.3,"Physics-Informed Loss",ha="center",va="center",fontsize=10,
        fontstyle="italic",color=RED_E)
# feeders into loss
arrow(46.5,30,50,22,color=GREEN_E)       # AD-Com -> loss (MSE_g)
arrow(66,30,60,22,color=BLUE_E)          # Governing -> loss
arrow(87.5,30,70,22,color=YEL_E)         # Data -> loss (MSE_y)

# ===== stopping / update loop =====
ax.add_patch(Polygon([(19,18.5),(27,22),(35,18.5),(27,15)],closed=True,
             fc="#EDEFF2",ec="#555",lw=1.4))
ax.text(27,18.6,"Stopping\ncriteria",ha="center",va="center",fontsize=9)
arrow(40,18.5,35,18.5,color=RED_E,lw=1.8)          # loss -> stopping
# update parameters feedback up into MLP
ax.add_patch(FancyArrowPatch((19,18.5),(21,30),connectionstyle="arc3,rad=-0.35",
             arrowstyle="-|>",mutation_scale=14,color="#333",lw=1.7))
ax.text(9.5,24,r"Update parameters $\theta$",ha="center",va="center",fontsize=10)
arrow(27,15,27,11,color="#333")
ax.text(27,9.3,"End",ha="center",va="center",fontsize=10,fontweight="bold")

ax.text(50,3.2,"Controller track fills the Dataset; the PINN-Obs learns hidden states from partial measurements + physics.",
        ha="center",va="center",fontsize=10,color="#444")

plt.tight_layout()
plt.savefig("docs/pinn_pipeline_mindmap.png",dpi=150,bbox_inches="tight")
print("saved -> docs/pinn_pipeline_mindmap.png")
