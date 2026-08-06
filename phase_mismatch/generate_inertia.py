"""Inertia perturbation sets for the mismatch study.
Reuses the validated dynamics + controller from generate_perturbed.py.
Perturbs Ix, Iy, Iz together by the same percentage.
Controller updated to the perturbed inertias (clean observer model-mismatch).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from phase_mismatch.generate_perturbed import generate_set, NOMINAL

base = np.load("datasets/spiral_dataset.npz")
X = base["X"]
test_x0 = [X[i, 0, :] for i in range(40, 50)]

os.makedirs("datasets/mismatch", exist_ok=True)
deviations = [-0.20, -0.10, 0.0, 0.10, 0.20]
for dev in deviations:
    P = dict(NOMINAL)
    P["Ix"] = NOMINAL["Ix"] * (1 + dev)
    P["Iy"] = NOMINAL["Iy"] * (1 + dev)
    P["Iz"] = NOMINAL["Iz"] * (1 + dev)
    tag = "inertia_%+d" % int(round(dev*100))
    if dev == 0.0:
        tag = "inertia_0"
    T, Xp, Up = generate_set(P, test_x0)
    np.savez(f"datasets/mismatch/{tag}.npz", T=T, X=Xp, U=Up,
             Ix=P["Ix"], Iy=P["Iy"], Iz=P["Iz"], dev=dev)
    print("inertia %+.0f%% (Ix=%.4f) -> %s  shape %s" %
          (dev*100, P["Ix"], tag, str(Xp.shape)), flush=True)
print("done", flush=True)