"""Combined worst-case mismatch: all three parameters at their individually
worst directions simultaneously (mass -20%, inertia +20%, arm -20%),
controller and observer both nominal. Deliberate extreme corner case.
Reuses the validated dynamics + controller via generate_set_unaware.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from phase_mismatch.generate_inertia_unaware import generate_set_unaware
from phase_mismatch.generate_perturbed import NOMINAL

base = np.load("datasets/spiral_dataset.npz")
X = base["X"]
test_x0 = [X[i, 0, :] for i in range(40, 50)]
os.makedirs("datasets/mismatch", exist_ok=True)

# individually-worst directions found in the separate studies
P_true = dict(NOMINAL)
P_true["m"]  = NOMINAL["m"]  * (1 - 0.20)     # mass  -20%
P_true["Ix"] = NOMINAL["Ix"] * (1 + 0.20)     # inertia +20%
P_true["Iy"] = NOMINAL["Iy"] * (1 + 0.20)
P_true["Iz"] = NOMINAL["Iz"] * (1 + 0.20)
P_true["l"]  = NOMINAL["l"]  * (1 - 0.20)     # arm   -20%
P_ctrl = dict(NOMINAL)                         # controller stays nominal

# nominal baseline set too, for the validation check
for tag, Pt in [("combined_worst_0", dict(NOMINAL)), ("combined_worst", P_true)]:
    T, Xp, Up = generate_set_unaware(Pt, P_ctrl, test_x0)
    np.savez(f"datasets/mismatch/{tag}.npz", T=T, X=Xp, U=Up,
             m=Pt["m"], Ix=Pt["Ix"], l=Pt["l"])
    print("%s: m=%.3f Ix=%.4f l=%.4f -> %s" %
          (tag, Pt["m"], Pt["Ix"], Pt["l"], str(Xp.shape)), flush=True)
print("done", flush=True)