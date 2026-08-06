"""Arm-length mismatch, CONTROLLER-UNAWARE variant.
True dynamics use perturbed arm length l; controller keeps nominal l.
(Arm length, like inertia, cancels under the updated design, so unaware
is the meaningful test.) Reuses the validated dynamics + controller.
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
deviations = [-0.20, -0.10, 0.0, 0.10, 0.20]
for dev in deviations:
    P_true = dict(NOMINAL); P_true["l"] = NOMINAL["l"] * (1 + dev)   # true drone
    P_ctrl = dict(NOMINAL)                                            # controller nominal
    tag = "arm_unaware_%+d" % int(round(dev*100))
    if dev == 0.0:
        tag = "arm_unaware_0"
    T, Xp, Up = generate_set_unaware(P_true, P_ctrl, test_x0)
    np.savez(f"datasets/mismatch/{tag}.npz", T=T, X=Xp, U=Up,
             l_true=P_true["l"], dev=dev)
    print("arm(unaware) %+.0f%% (true l=%.4f, ctrl l=%.4f) -> %s  %s" %
          (dev*100, P_true["l"], P_ctrl["l"], tag, str(Xp.shape)), flush=True)
print("done", flush=True)