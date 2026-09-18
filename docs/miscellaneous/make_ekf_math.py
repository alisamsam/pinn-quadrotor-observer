"""Nicely typeset EKF math with kid-friendly explanations."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
plt.rcParams["mathtext.fontset"] = "cm"   # Computer Modern (LaTeX-like) font

BLUE="#1D5FB0"; GREEN="#2E9E5B"
def box(x,y,w,h,fc,ec):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.2",fc=fc,ec=ec,lw=2))

ax.text(50,96,"Extended Kalman Filter (EKF) — the math",ha="center",fontsize=20,fontweight="bold")
ax.text(50,91.5,"keep a guess  $\\hat{x}$  and a confidence  $P$ ;  repeat Predict then Update every step",
        ha="center",fontsize=12,color="#555",fontstyle="italic")

def row(y, eq, cap):
    ax.text(8, y, eq, fontsize=18, va="center")
    ax.text(42, y, cap, fontsize=11.5, va="center", color="#222")

# ---- PREDICT ----
box(3, 60, 94, 26, "#EAF1FB", BLUE)
ax.text(6, 83.5, "PREDICT  (roll the model forward)", fontsize=14, fontweight="bold", color=BLUE)
row(77, r"$\hat{x}^{-} = f(\hat{x},\, u)$",            "guess the next state using the physics model")
row(70, r"$F = \partial f / \partial x$",              "F = the model's slope near the guess (local linear map)")
row(63.5, r"$P^{-} = F\,P\,F^{T} + Q$",                "grow the uncertainty;  Q = how much we distrust the model")

# ---- UPDATE ----
box(3, 12, 94, 44, "#E7F5EA", GREEN)
ax.text(6, 53.5, "UPDATE  (correct with the sensor)", fontsize=14, fontweight="bold", color=GREEN)
row(47, r"$\tilde{y} = y - C\,\hat{x}^{-}$",           "innovation = real sensor reading  -  predicted reading")
row(40, r"$S = C\,P^{-}C^{T} + R$",                    "how uncertain that gap is;  R = sensor noise")
row(33, r"$K = P^{-}C^{T}\,S^{-1}$",                   "K = the pull-strength (auto-tuned gain)")
row(26, r"$\hat{x} = \hat{x}^{-} + K\,\tilde{y}$",     "pull the guess toward the sensor")
row(18.5, r"$P = (I - K\,C)\,P^{-}$",                  "shrink the confidence after using the sensor")

ax.text(50, 6.5,
        "Kid version: predict with physics (get a little less sure), then look at the sensor and pull the guess toward it -",
        ha="center", fontsize=10.5, color="#444")
ax.text(50, 3.2,
        "harder when sensors are trustworthy, gentler when noisy.  The pull-strength K is recomputed every step.",
        ha="center", fontsize=10.5, color="#444")

plt.tight_layout()
plt.savefig("docs/ekf_math.png", dpi=150, bbox_inches="tight")
print("saved -> docs/ekf_math.png")
