"""Detailed 'Desired Path' model figure (Stage 1), nested under the pipeline strip."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

TEAL="#1C7293"; GREY="#6b7a86"; BLUE="#0B4F8A"; MID="#21295C"
P_F="#D6EAF0"; P_E="#1C7293"
POS_F="#E3F2D6"; POS_E="#4E9A2F"
VEL_F="#D7E6F7"; VEL_E="#2A6FB0"
ACC_F="#FCEFC7"; ACC_E="#C99A18"
YAW_F="#E7DDF3"; YAW_E="#6C3FA0"
OUT_F="#F7D2D2"; OUT_E="#B23A48"

def rbox(x,y,w,h,fc,ec,lw=1.6):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.15,rounding_size=1.0",fc=fc,ec=ec,lw=lw))
def arrow(x1,y1,x2,y2,color="#444",lw=1.7,ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=13,color=color,lw=lw,linestyle=ls,shrinkA=0,shrinkB=0))

ax.text(50,98,"PINN Quadrotor — Simulation Pipeline",ha="center",fontsize=20,fontweight="bold")

# overview strip (Stage 1 highlighted)
ov=[("1  Desired path",TEAL),("2  Controller",GREY),("3  Dataset (.npz)",GREY),
    ("4  PINN Observer",GREY),("5  Results",MID)]
xs=[4,23,42,61,80]; wov=15
for (label,c),x in zip(ov,xs):
    hl=2.6 if "Desired" in label else 1.6
    rbox(x,90,wov,5.5,c,"#111" if "Desired" in label else c,lw=hl)
    ax.text(x+wov/2,92.7,label,ha="center",va="center",color="white",fontsize=11,fontweight="bold")
for i in range(4): arrow(xs[i]+wov,92.7,xs[i+1],92.7,color="#555")
arrow(4+wov/2,90,50,84.5,color=TEAL,lw=1.8,ls="--")
ax.text(50,86,"open up Stage 1",ha="center",va="center",fontsize=10,color=TEAL,fontstyle="italic")


# ---------- panel ----------
rbox(2.5,10,95,74,"#FBFCFD",TEAL,lw=2.0)
ax.text(50,80.5,"Inside Stage 1 — Desired Path (trajectory generator)",
        ha="center",fontsize=14,fontweight="bold",color=TEAL)

# Parameters block
rbox(5,52,19,24,P_F,P_E,lw=1.6)
ax.text(14.5,73.5,"Parameters + time",ha="center",fontsize=12,fontweight="bold",color=P_E)
for y,txt in zip(np.linspace(69,55,5),
                 [r"$R = 10\ \mathrm{m}$",r"$\omega = 0.6\ \mathrm{rad/s}$",
                  r"$v_z = 2\ \mathrm{m/s}$",r"$\psi_d = \pi/4$",r"input: time $t$"]):
    ax.text(14.5,y,txt,ha="center",va="center",fontsize=11)

# Position block
rbox(27,52,20,24,POS_F,POS_E,lw=1.6)
ax.text(37,73.5,r"Position  $\mathbf{pos_d}$",ha="center",fontsize=12,fontweight="bold",color=POS_E)
for y,txt in zip(np.linspace(68,57,3),
                 [r"$x_d = R\,\sin(\omega t)$",r"$y_d = R\,\cos(\omega t)$",r"$z_d = v_z\, t$"]):
    ax.text(37,y,txt,ha="center",va="center",fontsize=11)

# Velocity block
rbox(50,52,20,24,VEL_F,VEL_E,lw=1.6)
ax.text(60,73.5,r"Velocity  $\mathbf{vel_d}$",ha="center",fontsize=12,fontweight="bold",color=VEL_E)
for y,txt in zip(np.linspace(68,57,3),
                 [r"$\dot{x}_d = R\omega\,\cos(\omega t)$",r"$\dot{y}_d = -R\omega\,\sin(\omega t)$",r"$\dot{z}_d = v_z$"]):
    ax.text(60,y,txt,ha="center",va="center",fontsize=11)

# Acceleration block
rbox(73,52,20,24,ACC_F,ACC_E,lw=1.6)
ax.text(83,73.5,r"Acceleration  $\mathbf{acc_d}$",ha="center",fontsize=12,fontweight="bold",color="#9c7a12")
for y,txt in zip(np.linspace(68,57,3),
                 [r"$\ddot{x}_d = -R\omega^2\sin(\omega t)$",r"$\ddot{y}_d = -R\omega^2\cos(\omega t)$",r"$\ddot{z}_d = 0$"]):
    ax.text(83,y,txt,ha="center",va="center",fontsize=10.5)

arrow(24,64,27,64,color="#333")
arrow(47,64,50,64,color="#333"); ax.text(48.5,66,"d/dt",ha="center",fontsize=9,color="#333")
arrow(70,64,73,64,color="#333"); ax.text(71.5,66,"d/dt",ha="center",fontsize=9,color="#333")


# Yaw block
rbox(27,34,20,13,YAW_F,YAW_E,lw=1.6)
ax.text(37,44.5,"Yaw",ha="center",fontsize=12,fontweight="bold",color=YAW_E)
ax.text(37,40.5,r"$\psi_d = \pi/4$  (constant)",ha="center",va="center",fontsize=11)
ax.text(37,37,r"$\dot{\psi}_d = 0$",ha="center",va="center",fontsize=11)

# spiral schematic (top view)
ax.add_patch(Circle((14.5,39),7,fill=False,ec=TEAL,lw=2.0))
ax.add_patch(FancyArrowPatch((21.5,39),(20.6,42.6),connectionstyle="arc3,rad=0.4",
             arrowstyle="-|>",mutation_scale=12,color=TEAL,lw=1.6))
ax.text(14.5,39,"spiral",ha="center",va="center",fontsize=10,color=TEAL)
ax.text(14.5,30,"top view (x,y);  z climbs with t",ha="center",va="center",fontsize=9,color="#555")

# Reference output block
rbox(38,15,34,9,OUT_F,OUT_E,lw=1.8)
ax.text(55,21,"Reference output  →  Controller",ha="center",va="center",fontsize=12,fontweight="bold",color=OUT_E)
ax.text(55,17.5,r"$pos_d,\ vel_d,\ acc_d,\ \psi_d,\ \dot{\psi}_d$",ha="center",va="center",fontsize=11)

arrow(37,52,46,24,color=POS_E)
arrow(60,52,58,24,color=VEL_E)
arrow(83,52,66,24,color="#C99A18")
arrow(41,34,49,24,color=YAW_E)

ax.text(55,11.5,r"Used by controller_v2: feedforward $acc_d$ (Step B) + PD on $pos_d,\ vel_d$",
        ha="center",va="center",fontsize=10,color="#444")
ax.text(50,5.5,"The desired path is a pure math generator: parameters + time give position and its 1st and 2nd derivatives. No learning here.",
        ha="center",va="center",fontsize=10,color="#444")

plt.tight_layout()
plt.savefig("docs/desired_path_model.png",dpi=150,bbox_inches="tight")
print("saved -> docs/desired_path_model.png")
