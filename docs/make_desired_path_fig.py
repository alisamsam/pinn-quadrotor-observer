"""Detailed 'Desired Path' model figure (Stage 1). omega = 0.6."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 10.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

TEAL="#1C7293"; GREY="#6b7a86"; MID="#21295C"
P_F="#D6EAF0"; P_E="#1C7293"
POS_F="#E3F2D6"; POS_E="#4E9A2F"
VEL_F="#D7E6F7"; VEL_E="#2A6FB0"
ACC_F="#FCEFC7"; ACC_E="#C99A18"
YAW_F="#E7DDF3"; YAW_E="#6C3FA0"
SET_F="#EEF1F3"; SET_E="#6b7a86"
OUT_F="#F7D2D2"; OUT_E="#B23A48"

def rbox(x,y,w,h,fc,ec,lw=1.6):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.15,rounding_size=1.0",fc=fc,ec=ec,lw=lw))
def arrow(x1,y1,x2,y2,color="#444",lw=1.7,ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=13,color=color,lw=lw,linestyle=ls,shrinkA=0,shrinkB=0))

ax.text(50,98,"PINN Quadrotor — Simulation Pipeline",ha="center",fontsize=20,fontweight="bold")
ov=[("1  Desired path",TEAL),("2  Controller",GREY),("3  Dataset (.npz)",GREY),
    ("4  PINN Observer",GREY),("5  Results",MID)]
xs=[4,23,42,61,80]; wov=15
for (label,c),x in zip(ov,xs):
    hl=2.6 if "Desired" in label else 1.6
    rbox(x,91,wov,5.2,c,"#111" if "Desired" in label else c,lw=hl)
    ax.text(x+wov/2,93.6,label,ha="center",va="center",color="white",fontsize=11,fontweight="bold")
for i in range(4): arrow(xs[i]+wov,93.6,xs[i+1],93.6,color="#555")
arrow(4+wov/2,91,50,85.5,color=TEAL,lw=1.8,ls="--")
ax.text(50,87,"open up Stage 1",ha="center",va="center",fontsize=10,color=TEAL,fontstyle="italic")


# ---------- panel ----------
rbox(2.5,9,95,76,"#FBFCFD",TEAL,lw=2.0)
ax.text(50,82,"Inside Stage 1 — Desired Path (trajectory generator, omega = 0.6)",
        ha="center",fontsize=14,fontweight="bold",color=TEAL)

# top row: Parameters -> Position -> Velocity -> Acceleration
rbox(5,55,19,22,P_F,P_E,lw=1.6)
ax.text(14.5,74.5,"Parameters + time",ha="center",fontsize=11.5,fontweight="bold",color=P_E)
for y,txt in zip(np.linspace(70.5,57.5,5),
                 [r"$R = 10\ \mathrm{m}$",r"$\omega = 0.6\ \mathrm{rad/s}$",
                  r"$v_z = 2\ \mathrm{m/s}$",r"$\psi_d = \pi/4$",r"input: time $t$"]):
    ax.text(14.5,y,txt,ha="center",va="center",fontsize=10.5)

rbox(27,55,20,22,POS_F,POS_E,lw=1.6)
ax.text(37,74.5,r"Position  $\mathbf{pos_d}$",ha="center",fontsize=11.5,fontweight="bold",color=POS_E)
for y,txt in zip(np.linspace(69,59,3),
                 [r"$x_d = R\,\sin(\omega t)$",r"$y_d = R\,\cos(\omega t)$",r"$z_d = v_z\, t$"]):
    ax.text(37,y,txt,ha="center",va="center",fontsize=10.5)

rbox(50,55,20,22,VEL_F,VEL_E,lw=1.6)
ax.text(60,74.5,r"Velocity  $\mathbf{vel_d}$",ha="center",fontsize=11.5,fontweight="bold",color=VEL_E)
for y,txt in zip(np.linspace(69,59,3),
                 [r"$\dot{x}_d = R\omega\,\cos(\omega t)$",r"$\dot{y}_d = -R\omega\,\sin(\omega t)$",r"$\dot{z}_d = v_z$"]):
    ax.text(60,y,txt,ha="center",va="center",fontsize=10.5)

rbox(73,55,20,22,ACC_F,ACC_E,lw=1.6)
ax.text(83,74.5,r"Acceleration  $\mathbf{acc_d}$",ha="center",fontsize=11.5,fontweight="bold",color="#9c7a12")
for y,txt in zip(np.linspace(69,59,3),
                 [r"$\ddot{x}_d = -R\omega^2\sin(\omega t)$",r"$\ddot{y}_d = -R\omega^2\cos(\omega t)$",r"$\ddot{z}_d = 0$"]):
    ax.text(83,y,txt,ha="center",va="center",fontsize=10)

arrow(24,66,27,66,color="#333")
arrow(47,66,50,66,color="#333"); ax.text(48.5,68,"d/dt",ha="center",fontsize=9)
arrow(70,66,73,66,color="#333"); ax.text(71.5,68,"d/dt",ha="center",fontsize=9)


# Setup & path demands (context box, lower-left)
rbox(5,29,34,22,SET_F,SET_E,lw=1.5)
ax.text(22,48.5,"Setup & path demands",ha="center",fontsize=11.5,fontweight="bold",color="#4a555e")
for y,txt in zip(np.linspace(44.5,31.5,6),
                 [r"start $x_0 = (4,\,5,\,0)$",
                  r"path$(t{=}0) = (0,\,10,\,0)$  $\Rightarrow$ initial transient",
                  r"window: $t \in [0,30]\,$s, $dt=0.01$  (3000 pts)",
                  r"$\approx 2.9$ loops  ($T = 2\pi/\omega \approx 10.5\,$s)",
                  r"peak speed $R\omega = 6\,$m/s ; accel $R\omega^2 = 3.6\,$m/s$^2$",
                  r"required tilt $\approx 20^\circ$"]):
    ax.text(6.5,y,txt,ha="left",va="center",fontsize=9.5)

# Yaw block
rbox(43,32,19,15,YAW_F,YAW_E,lw=1.6)
ax.text(52.5,44.5,"Yaw",ha="center",fontsize=11.5,fontweight="bold",color=YAW_E)
ax.text(52.5,40,r"$\psi_d = \pi/4$ (const)",ha="center",va="center",fontsize=10.5)
ax.text(52.5,36,r"$\dot{\psi}_d = 0$",ha="center",va="center",fontsize=10.5)

# Reference output
rbox(66,30,29,16,OUT_F,OUT_E,lw=1.8)
ax.text(80.5,41.5,"Reference output",ha="center",va="center",fontsize=12,fontweight="bold",color=OUT_E)
ax.text(80.5,37.5,r"$pos_d,\ vel_d,\ acc_d,$",ha="center",va="center",fontsize=10.5)
ax.text(80.5,33.5,r"$\psi_d,\ \dot{\psi}_d\ \rightarrow$ Controller",ha="center",va="center",fontsize=10.5)

arrow(37,55,74,46,color=POS_E)
arrow(60,55,78,46,color=VEL_E)
arrow(83,55,88,46,color="#C99A18")
arrow(62,39,66,38,color=YAW_E)

ax.text(50,20,r"Used by controller_v2: feedforward $acc_d$ (Step B) + PD on $pos_d,\ vel_d$; $\psi_d$ to the yaw loop.",
        ha="center",va="center",fontsize=10,color="#444")
ax.text(50,15.5,"Same generator also produces the circle and figure-8 references (used for the gain-transfer check).",
        ha="center",va="center",fontsize=10,color="#444")
ax.text(50,11.5,"Pure math generator: parameters + time give position and its 1st/2nd derivatives. No physical params (m, l, I, g) and no learning here.",
        ha="center",va="center",fontsize=9.5,color="#666",fontstyle="italic")

plt.tight_layout()
plt.savefig("docs/desired_path_model.png",dpi=150,bbox_inches="tight")
print("saved -> docs/desired_path_model.png")
