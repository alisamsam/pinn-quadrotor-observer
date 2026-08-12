"""Standalone Adaptive PINN-Obs architecture figure (sharp colors, for email/thesis)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 9.5))
ax.set_xlim(0, 100); ax.set_ylim(2, 92); ax.axis("off")

# sharp, saturated palette (same family as the pipeline figure)
MLP_C="#0F8B8D"; AD_C="#2E9E5B"; GOV_C="#1D5FB0"; DATA_C="#E8701A"; LOSS_C="#C0392B"
FB="#D7263D"

def rbox(x,y,w,h,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.15,rounding_size=1.2",fc=fc,ec="white",lw=2.2))
def arrow(x1,y1,x2,y2,color="#333",lw=1.8):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=15,color=color,lw=lw,shrinkA=0,shrinkB=0))

ax.text(50,89,"Adaptive PINN-Obs — nonlinear state observer",ha="center",fontsize=18,fontweight="bold")
ax.text(50,85,"state estimate and adaptive gain from time and initial condition, constrained by physics and data",
        ha="center",fontsize=11,color="#555",fontstyle="italic")

# ===== MLP (teal) =====
rbox(4,28,34,50,MLP_C)
ax.text(21,74.5,r"MLP : $\theta_\theta,\ \theta_z$",ha="center",fontsize=13,fontweight="bold",color="white")
xin,xh1,xh2,xout=9,17,26,34
yh=np.linspace(33,69,6); yo=np.linspace(31,71,7)
for y in yh: ax.plot([xin,xh1],[51,y],color="#CDEAEA",lw=0.5,zorder=1)
for a in yh:
    for b in yh: ax.plot([xh1,xh2],[a,b],color="#CDEAEA",lw=0.5,zorder=1)
for a in yh:
    for b in yo: ax.plot([xh2,xout],[a,b],color="#CDEAEA",lw=0.5,zorder=1)
ax.add_patch(Circle((xin,51),1.2,fc="white",ec="#0a3f40",zorder=3)); ax.text(xin,51,r"$t$",ha="center",va="center",fontsize=10)
for y in yh:
    ax.add_patch(Circle((xh1,y),1.2,fc="white",ec="#0a3f40",zorder=3))
    ax.add_patch(Circle((xh2,y),1.2,fc="white",ec="#0a3f40",zorder=3))
olabels=[r"$\ell_d(t)$",r"$\vdots$",r"$\ell_1(t)$",r"$\hat{x}_n(t)$",r"$\vdots$",r"$\hat{x}_2(t)$",r"$\hat{x}_1(t)$"]
for y,lab in zip(yo,olabels):
    ax.add_patch(Circle((xout,y),1.1,fc="white",ec="#0a3f40",zorder=3))
    ax.text(xout+2.4,y,lab,ha="left",va="center",fontsize=9,color="white")


# ===== AD-Com (green) =====
rbox(41,28,13,50,AD_C)
ax.text(47.5,74.5,"AD-Com",ha="center",fontsize=13,fontweight="bold",color="white")
adcom=[r"$\dot{\hat{x}}_1(t)$",r"$\dot{\hat{x}}_2(t)$",r"$\vdots$",r"$\dot{\hat{x}}_n(t)$",
       r"$L(t)$",r"$f_1(t)$",r"$\vdots$",r"$f_n(t)$"]
for y,lab in zip(np.linspace(69,32,len(adcom)),adcom):
    ax.text(47.5,y,lab,ha="center",va="center",fontsize=10,color="white")

# ===== Governing Equation (blue) =====
rbox(57,28,22,50,GOV_C)
ax.text(68,74.5,"Governing Equation",ha="center",fontsize=13,fontweight="bold",color="white")
for y,lab in zip(np.linspace(69,59,4),
                 [r"$\hat{x}_1(t_0)$",r"$\hat{x}_2(t_0)$",r"$\vdots$",r"$\hat{x}_n(t_0)$"]):
    ax.text(68,y,lab,ha="center",va="center",fontsize=10,color="white")
ax.text(68,49,r"$g(t,\hat{x}) = \dot{\hat{x}}(t) - f(\hat{x},t) - Bu$",ha="center",va="center",fontsize=10.5,color="white")
ax.text(68,44,r"$-\, L(t)\,C\,(x(t)-\hat{x}(t))$",ha="center",va="center",fontsize=10.5,color="white")

# ===== Data (orange) =====
rbox(82,28,15,50,DATA_C)
ax.text(89.5,74.5,"Data",ha="center",fontsize=13,fontweight="bold",color="white")
data=[r"$\hat{x}_a^1(t_0)$",r"$\hat{x}_a^2(t_0)$",r"$\vdots$",r"$\hat{x}_a^n(t_0)$",
      r"$y_1(t)$",r"$y_2(t)$",r"$\vdots$",r"$y_m(t)$"]
for y,lab in zip(np.linspace(69,32,len(data)),data):
    ax.text(89.5,y,lab,ha="center",va="center",fontsize=10,color="white")

arrow(38,51,41,51); arrow(54,51,57,51); arrow(82,51,79,51)


# ===== Physics-Informed Loss (red) =====
rbox(41,13,36,8,LOSS_C)
ax.text(59,17.8,r"$MSE = MSE_0 + MSE_g + MSE_y$",ha="center",va="center",fontsize=12.5,color="white")
ax.text(59,14.5,"Physics-Informed Loss",ha="center",va="center",fontsize=10,fontstyle="italic",color="white")
arrow(47.5,28,52,21,color=AD_C); arrow(68,28,62,21,color=GOV_C); arrow(89.5,28,72,21,color=DATA_C)

# ===== stopping / update loop =====
ax.add_patch(Polygon([(18,17.0),(27,21.0),(36,17.0),(27,13.0)],closed=True,fc="#E7E9ED",ec="#555",lw=1.4))
ax.text(27,17.1,"Stopping\ncriteria",ha="center",va="center",fontsize=9)
arrow(41,17,36,17,color=LOSS_C,lw=1.9)
ax.add_patch(FancyArrowPatch((18,17.0),(20,28),connectionstyle="arc3,rad=-0.35",
             arrowstyle="-|>",mutation_scale=15,color="#333",lw=1.8))
ax.text(8.5,23,r"Update parameters $\theta$",ha="center",va="center",fontsize=10)
arrow(27,13.0,27,9.2,color="#333")
ax.text(27,7.7,"End",ha="center",va="center",fontsize=10,fontweight="bold")

ax.text(50,4.3,"Inputs: time t and initial condition.  Outputs: state estimate x_hat and adaptive gain L.  "
        "Trained by three losses: initial condition (MSE_0), physics residual (MSE_g), and data (MSE_y).",
        ha="center",va="center",fontsize=9.5,color="#444")

plt.tight_layout()
plt.savefig("docs/pinn_obs_architecture.png",dpi=150,bbox_inches="tight")
print("saved -> docs/pinn_obs_architecture.png")
