"""Desired Path (Stage 1) figure with the EXPANDED pipeline path on top. omega=0.6."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(17, 10.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

TEAL="#1C7293"; GREY="#6b7a86"; DYN="#8a5a2b"; MID="#21295C"
P_F="#D6EAF0"; P_E="#1C7293"
POS_F="#E3F2D6"; POS_E="#4E9A2F"
VEL_F="#D7E6F7"; VEL_E="#2A6FB0"
ACC_F="#FCEFC7"; ACC_E="#C99A18"
YAW_F="#E7DDF3"; YAW_E="#6C3FA0"
SET_F="#EEF1F3"; SET_E="#6b7a86"
OUT_F="#F7D2D2"; OUT_E="#B23A48"

def rbox(x,y,w,h,fc,ec,lw=1.6):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.15,rounding_size=0.8",fc=fc,ec=ec,lw=lw))
def arrow(x1,y1,x2,y2,color="#444",lw=1.6,ls="-",rad=None):
    cs = f"arc3,rad={rad}" if rad is not None else "arc3,rad=0"
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,color=color,lw=lw,linestyle=ls,shrinkA=0,shrinkB=0,connectionstyle=cs))

ax.text(50,98.5,"PINN Quadrotor — Simulation Pipeline",ha="center",fontsize=19,fontweight="bold")

# ---- expanded pipeline strip (7 nodes) ----
labels=["1\nDesired\npath","2\nController","3\nDrone\ndynamics","4\nMeasure\n$y=Cx$",
        "5\nDataset\n(.npz)","6\nPINN\nObserver","7\nResults"]
fills =[TEAL,GREY,DYN,GREY,GREY,GREY,MID]
lefts=np.linspace(3.5,82.7,7); w=11; cy=92.8
cx=lefts+w/2
for x,lab,fc in zip(lefts,labels,fills):
    hl=2.4 if lab.startswith("1") else 1.4
    rbox(x,89.8,w,6.0,fc,"#111" if lab.startswith("1") else fc,lw=hl)
    ax.text(x+w/2,cy,lab,ha="center",va="center",color="white",fontsize=8.6,fontweight="bold",linespacing=1.05)
for i in range(6):
    arrow(lefts[i]+w,cy,lefts[i+1],cy,color="#555")
# closed loop: dynamics -> controller (state feedback)
arrow(cx[2],95.8,cx[1],95.8,color="#333",rad=0.5,lw=1.4)
ax.text((cx[1]+cx[2])/2,98.3,"state feedback",ha="center",fontsize=7.5,color="#333",fontstyle="italic")
# evaluation taps
arrow(cx[2],89.8,cx[2],87.9,color=POS_E,lw=1.3)
ax.text(cx[2],87.0,"desired vs actual\n(controller RMSE)",ha="center",va="top",fontsize=7.3,color=POS_E)
arrow(cx[6],89.8,cx[6],87.9,color=VEL_E,lw=1.3)
ax.text(cx[6],87.0,"true vs estimated\n(observer RMSE)",ha="center",va="top",fontsize=7.3,color=VEL_E)
# open up stage 1
arrow(cx[0],89.8,16,84.5,color=TEAL,lw=1.7,ls="--")
ax.text(24,86.2,"open up Stage 1",ha="center",fontsize=9.5,color=TEAL,fontstyle="italic")


# ---------- panel (Desired Path detail) ----------
rbox(2.5,8,95,75,"#FBFCFD",TEAL,lw=2.0)
ax.text(50,80,"Inside Stage 1 — Desired Path (trajectory generator, omega = 0.6)",
        ha="center",fontsize=14,fontweight="bold",color=TEAL)

rbox(5,54,19,21,P_F,P_E,lw=1.6)
ax.text(14.5,72.5,"Parameters + time",ha="center",fontsize=11.5,fontweight="bold",color=P_E)
for y,txt in zip(np.linspace(68.5,56,5),
                 [r"$R = 10\ \mathrm{m}$",r"$\omega = 0.6\ \mathrm{rad/s}$",
                  r"$v_z = 2\ \mathrm{m/s}$",r"$\psi_d = \pi/4$",r"input: time $t$"]):
    ax.text(14.5,y,txt,ha="center",va="center",fontsize=10.5)

rbox(27,54,20,21,POS_F,POS_E,lw=1.6)
ax.text(37,72.5,r"Position  $\mathbf{pos_d}$",ha="center",fontsize=11.5,fontweight="bold",color=POS_E)
for y,txt in zip(np.linspace(67,58,3),
                 [r"$x_d = R\,\sin(\omega t)$",r"$y_d = R\,\cos(\omega t)$",r"$z_d = v_z\, t$"]):
    ax.text(37,y,txt,ha="center",va="center",fontsize=10.5)

rbox(50,54,20,21,VEL_F,VEL_E,lw=1.6)
ax.text(60,72.5,r"Velocity  $\mathbf{vel_d}$",ha="center",fontsize=11.5,fontweight="bold",color=VEL_E)
for y,txt in zip(np.linspace(67,58,3),
                 [r"$\dot{x}_d = R\omega\,\cos(\omega t)$",r"$\dot{y}_d = -R\omega\,\sin(\omega t)$",r"$\dot{z}_d = v_z$"]):
    ax.text(60,y,txt,ha="center",va="center",fontsize=10.5)

rbox(73,54,20,21,ACC_F,ACC_E,lw=1.6)
ax.text(83,72.5,r"Acceleration  $\mathbf{acc_d}$",ha="center",fontsize=11.5,fontweight="bold",color="#9c7a12")
for y,txt in zip(np.linspace(67,58,3),
                 [r"$\ddot{x}_d = -R\omega^2\sin(\omega t)$",r"$\ddot{y}_d = -R\omega^2\cos(\omega t)$",r"$\ddot{z}_d = 0$"]):
    ax.text(83,y,txt,ha="center",va="center",fontsize=10)

arrow(24,64.5,27,64.5,color="#333"); arrow(47,64.5,50,64.5,color="#333"); arrow(70,64.5,73,64.5,color="#333")
ax.text(48.5,66.3,"d/dt",ha="center",fontsize=9); ax.text(71.5,66.3,"d/dt",ha="center",fontsize=9)


# Setup & path demands
rbox(5,28,34,21,SET_F,SET_E,lw=1.5)
ax.text(22,46.5,"Setup & path demands",ha="center",fontsize=11.5,fontweight="bold",color="#4a555e")
for y,txt in zip(np.linspace(42.5,30,6),
                 [r"start $x_0 = (4,\,5,\,0)$",
                  r"path$(t{=}0) = (0,\,10,\,0)$  $\Rightarrow$ initial transient",
                  r"window: $t \in [0,30]\,$s, $dt=0.01$  (3000 pts)",
                  r"$\approx 2.9$ loops  ($T = 2\pi/\omega \approx 10.5\,$s)",
                  r"peak speed $R\omega = 6\,$m/s ; accel $R\omega^2 = 3.6\,$m/s$^2$",
                  r"required tilt $\approx 20^\circ$"]):
    ax.text(6.5,y,txt,ha="left",va="center",fontsize=9.5)

rbox(43,31,19,15,YAW_F,YAW_E,lw=1.6)
ax.text(52.5,43.5,"Yaw",ha="center",fontsize=11.5,fontweight="bold",color=YAW_E)
ax.text(52.5,39,r"$\psi_d = \pi/4$ (const)",ha="center",va="center",fontsize=10.5)
ax.text(52.5,35,r"$\dot{\psi}_d = 0$",ha="center",va="center",fontsize=10.5)

rbox(66,29,29,16,OUT_F,OUT_E,lw=1.8)
ax.text(80.5,40.5,"Reference output",ha="center",va="center",fontsize=12,fontweight="bold",color=OUT_E)
ax.text(80.5,36.5,r"$pos_d,\ vel_d,\ acc_d,$",ha="center",va="center",fontsize=10.5)
ax.text(80.5,32.5,r"$\psi_d,\ \dot{\psi}_d\ \rightarrow$ Controller",ha="center",va="center",fontsize=10.5)

arrow(37,54,74,45,color=POS_E); arrow(60,54,78,45,color=VEL_E)
arrow(83,54,88,45,color="#C99A18"); arrow(62,38,66,37,color=YAW_E)

ax.text(50,19,r"Used by controller_v2: feedforward $acc_d$ (Step B) + PD on $pos_d,\ vel_d$; $\psi_d$ to the yaw loop.",
        ha="center",va="center",fontsize=10,color="#444")
ax.text(50,15,"Same generator also produces the circle and figure-8 references (used for the gain-transfer check).",
        ha="center",va="center",fontsize=10,color="#444")
ax.text(50,11,"Pure math generator: parameters + time give position and its 1st/2nd derivatives. No physical params (m, l, I, g) and no learning here.",
        ha="center",va="center",fontsize=9.5,color="#666",fontstyle="italic")

plt.tight_layout()
plt.savefig("docs/desired_path_model.png",dpi=150,bbox_inches="tight")
print("saved -> docs/desired_path_model.png")
