"""Standalone simulation pipeline process diagram — bold colors + feedback loops."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(18, 9))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

# sharp, saturated colors
C=["#0F8B8D","#1D5FB0","#E8701A","#7A2E9D","#2E9E5B","#C21E7A","#16305E"]
FB="#D7263D"   # feedback loops
DK="#242424"   # main flow arrows

def stage(x,w,fc,title,sub):
    ax.add_patch(FancyBboxPatch((x,44),w,17,boxstyle="round,pad=0.2,rounding_size=1.4",
                                fc=fc,ec="white",lw=2.2))
    ax.text(x+w/2,55.5,title,ha="center",va="center",color="white",fontsize=11.5,fontweight="bold")
    ax.text(x+w/2,49.0,sub,ha="center",va="center",color="white",fontsize=8.4,linespacing=1.1)

def flow(x1,x2,label):
    ax.add_patch(FancyArrowPatch((x1,52.5),(x2,52.5),arrowstyle="-|>",mutation_scale=20,color=DK,lw=2.6,shrinkA=0,shrinkB=0))
    ax.text((x1+x2)/2,54.6,label,ha="center",va="bottom",fontsize=8,color=DK,fontweight="bold")

ax.text(50,95,"PINN Quadrotor — Simulation Pipeline",ha="center",fontsize=22,fontweight="bold")
ax.text(50,90,"process flow and feedback loops",ha="center",fontsize=12,color="#555",fontstyle="italic")

titles=["1 · Desired path","2 · Controller","3 · Drone dynamics","4 · Measurement",
        "5 · Dataset (.npz)","6 · PINN Observer","7 · Results"]
subs=["spiral reference\n$R,\\omega,v_z,\\psi_d$","feedforward + PD\ntilt-comp","Singha model\n$m,l,I,g$ (solve_ivp)",
      "$y=Cx$ (+noise)\n6 of 12 states","40/10 split\nnormalize","learn $\\hat{x}$\ndata + physics","12 states\nRMSE (min)"]
lefts=np.linspace(3.5,84.5,7); w=11.5; cx=lefts+w/2
for x,t,s,c in zip(lefts,titles,subs,C): stage(x,w,c,t,s)
flabels=["ref","U","x(t)","y","data","$\\hat{x}$"]
for i,lb in enumerate(flabels): flow(lefts[i]+w,lefts[i+1],lb)


# ---- feedback loop 1: closed-loop control (Dynamics -> Controller) ----
ax.add_patch(FancyArrowPatch((cx[2],61),(cx[1],61),arrowstyle="-|>",mutation_scale=20,
             color=FB,lw=2.8,connectionstyle="arc3,rad=0.55",shrinkA=0,shrinkB=0))
ax.text((cx[1]+cx[2])/2,75.5,"state feedback  (closed-loop control)",ha="center",
        fontsize=9.5,color=FB,fontweight="bold")

# ---- feedback loop 2: PINN training loop (self-loop on observer) ----
ax.add_patch(FancyArrowPatch((cx[5]+5,61.3),(cx[5]-5,61.3),arrowstyle="-|>",mutation_scale=18,
             color=FB,lw=2.8,connectionstyle="arc3,rad=-0.9",shrinkA=0,shrinkB=0))
ax.text(cx[5],75.5,"training loop:  update $\\theta$",ha="center",fontsize=9.5,color=FB,fontweight="bold")
ax.text(cx[5],72.5,"until stopping criteria",ha="center",fontsize=9.5,color=FB,fontweight="bold")

# ---- evaluation taps ----
def tap(cxi,num,txt,col):
    bx=min(max(cxi-13,1),73)
    ax.add_patch(FancyArrowPatch((cxi,44),(cxi,37.5),arrowstyle="-|>",mutation_scale=16,color=col,lw=2.2,shrinkA=0,shrinkB=0))
    ax.add_patch(FancyBboxPatch((bx,29),26,7.5,boxstyle="round,pad=0.2,rounding_size=1.0",fc="white",ec=col,lw=2.0))
    ax.text(bx+13,34.4,num,ha="center",va="center",fontsize=9.5,color=col,fontweight="bold")
    ax.text(bx+13,31.4,txt,ha="center",va="center",fontsize=8.4,color="#333")
tap(cx[2],"TAP ①  desired vs actual","controller quality (Step-2 tracking RMSE)",C[2])
tap(cx[6],"TAP ②  true vs estimated","observer quality (Stage-6 state RMSE)",C[5])

# legend
ax.add_patch(FancyArrowPatch((6,20),(14,20),arrowstyle="-|>",mutation_scale=16,color=FB,lw=2.6))
ax.text(15.5,20,"= feedback loop",ha="left",va="center",fontsize=9,color=FB,fontweight="bold")
ax.add_patch(FancyArrowPatch((6,16),(14,16),arrowstyle="-|>",mutation_scale=16,color=DK,lw=2.6))
ax.text(15.5,16,"= forward data flow",ha="left",va="center",fontsize=9,color=DK)

ax.text(50,7,"Two quality measures stay separate:  TAP ① controller (desired vs actual)   |   TAP ② observer (true vs estimated).",
        ha="center",fontsize=9.5,color="#444")

plt.tight_layout()
plt.savefig("docs/pipeline_process.png",dpi=150,bbox_inches="tight")
print("saved -> docs/pipeline_process.png")
