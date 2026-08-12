"""Standalone Adaptive PINN-Obs architecture figure (for the email / thesis)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 9.5))
ax.set_xlim(0, 100); ax.set_ylim(2, 92); ax.axis("off")

GRY_F="#ECEFF1"; GRY_E="#8A97A0"; GREEN_E="#4E9A2F"; BLUE="#0B4F8A"
GREEN_F="#E3F2D6"; BLUE_F="#D7E6F7"; BLUE_E="#2A6FB0"
YEL_F="#FCEFC7"; YEL_E="#C99A18"; RED_F="#F7D2D2"; RED_E="#B23A48"; FB="#D7263D"

def rbox(x,y,w,h,fc,ec,lw=1.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.15,rounding_size=1.0",fc=fc,ec=ec,lw=lw))
def arrow(x1,y1,x2,y2,color="#333",lw=1.7):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,color=color,lw=lw,shrinkA=0,shrinkB=0))

ax.text(50,89,"Adaptive PINN-Obs — nonlinear state observer",ha="center",fontsize=18,fontweight="bold")
ax.text(50,85,"state estimate and adaptive gain from time and initial condition, constrained by physics and data",
        ha="center",fontsize=11,color="#555",fontstyle="italic")

# ===== MLP =====
rbox(4,28,34,50,GRY_F,GRY_E,lw=1.4)
ax.text(21,80,r"MLP : $\theta_\theta,\ \theta_z$",ha="center",fontsize=13,fontweight="bold")
xin,xh1,xh2,xout=9,17,26,34
yh=np.linspace(34,72,6); yo=np.linspace(32,74,7)
for y in yh: ax.plot([xin,xh1],[53,y],color="#c2ccd2",lw=0.5,zorder=1)
for a in yh:
    for b in yh: ax.plot([xh1,xh2],[a,b],color="#c2ccd2",lw=0.5,zorder=1)
for a in yh:
    for b in yo: ax.plot([xh2,xout],[a,b],color="#c2ccd2",lw=0.5,zorder=1)
ax.add_patch(Circle((xin,53),1.2,fc="white",ec="#333",zorder=3)); ax.text(xin,53,r"$t$",ha="center",va="center",fontsize=10)
for y in yh:
    ax.add_patch(Circle((xh1,y),1.2,fc="white",ec="#333",zorder=3))
    ax.add_patch(Circle((xh2,y),1.2,fc="white",ec="#333",zorder=3))
olabels=[r"$\ell_d(t)$",r"$\vdots$",r"$\ell_1(t)$",r"$\hat{x}_n(t)$",r"$\vdots$",r"$\hat{x}_2(t)$",r"$\hat{x}_1(t)$"]
for y,lab in zip(yo,olabels):
    ax.add_patch(Circle((xout,y),1.1,fc="white",ec="#333",zorder=3))
    ax.text(xout+2.4,y,lab,ha="left",va="center",fontsize=9)


# ===== AD-Com =====
rbox(41,28,13,50,GREEN_F,GREEN_E,lw=1.6)
ax.text(47.5,80,"AD-Com",ha="center",fontsize=13,fontweight="bold",color=GREEN_E)
adcom=[r"$\dot{\hat{x}}_1(t)$",r"$\dot{\hat{x}}_2(t)$",r"$\vdots$",r"$\dot{\hat{x}}_n(t)$",
       r"$L(t)$",r"$f_1(t)$",r"$\vdots$",r"$f_n(t)$"]
for y,lab in zip(np.linspace(73,34,len(adcom)),adcom):
    ax.text(47.5,y,lab,ha="center",va="center",fontsize=10)

# ===== Governing Equation =====
rbox(57,28,22,50,BLUE_F,BLUE_E,lw=1.6)
ax.text(68,80,"Governing Equation",ha="center",fontsize=13,fontweight="bold",color=BLUE_E)
for y,lab in zip(np.linspace(73,61,4),
                 [r"$\hat{x}_1(t_0)$",r"$\hat{x}_2(t_0)$",r"$\vdots$",r"$\hat{x}_n(t_0)$"]):
    ax.text(68,y,lab,ha="center",va="center",fontsize=10)
ax.text(68,50,r"$g(t,\hat{x}) = \dot{\hat{x}}(t) - f(\hat{x},t) - Bu$",ha="center",va="center",fontsize=10.5)
ax.text(68,45,r"$-\, L(t)\,C\,(x(t)-\hat{x}(t))$",ha="center",va="center",fontsize=10.5)

# ===== Data =====
rbox(82,28,15,50,YEL_F,YEL_E,lw=1.6)
ax.text(89.5,80,"Data",ha="center",fontsize=13,fontweight="bold",color="#9c7a12")
data=[r"$\hat{x}_a^1(t_0)$",r"$\hat{x}_a^2(t_0)$",r"$\vdots$",r"$\hat{x}_a^n(t_0)$",
      r"$y_1(t)$",r"$y_2(t)$",r"$\vdots$",r"$y_m(t)$"]
for y,lab in zip(np.linspace(73,34,len(data)),data):
    ax.text(89.5,y,lab,ha="center",va="center",fontsize=10)

arrow(38,53,41,53); arrow(54,53,57,53); arrow(82,53,79,53)


# ===== Physics-Informed Loss =====
rbox(41,14,36,8,RED_F,RED_E,lw=1.8)
ax.text(59,18.8,r"$MSE = MSE_0 + MSE_g + MSE_y$",ha="center",va="center",fontsize=12.5)
ax.text(59,15.5,"Physics-Informed Loss",ha="center",va="center",fontsize=10,fontstyle="italic",color=RED_E)
arrow(47.5,28,52,22,color=GREEN_E)
arrow(68,28,62,22,color=BLUE_E)
arrow(89.5,28,72,22,color=YEL_E)

# ===== stopping / update loop =====
ax.add_patch(Polygon([(18,17.5),(27,21.5),(36,17.5),(27,13.5)],closed=True,fc="#EDEFF2",ec="#555",lw=1.4))
ax.text(27,17.6,"Stopping\ncriteria",ha="center",va="center",fontsize=9)
arrow(41,18,36,18,color=RED_E,lw=1.8)
ax.add_patch(FancyArrowPatch((18,17.5),(20,28),connectionstyle="arc3,rad=-0.35",
             arrowstyle="-|>",mutation_scale=14,color="#333",lw=1.7))
ax.text(8.5,23,r"Update parameters $\theta$",ha="center",va="center",fontsize=10)
arrow(27,13.5,27,9.5,color="#333")
ax.text(27,8,"End",ha="center",va="center",fontsize=10,fontweight="bold")

ax.text(50,4.5,"Inputs: time t and initial condition.  Outputs: state estimate x_hat and adaptive gain L.  "
        "Trained by three losses: initial condition (MSE_0), physics residual (MSE_g), and data (MSE_y).",
        ha="center",va="center",fontsize=9.5,color="#444")

plt.tight_layout()
plt.savefig("docs/pinn_obs_architecture.png",dpi=150,bbox_inches="tight")
print("saved -> docs/pinn_obs_architecture.png")
