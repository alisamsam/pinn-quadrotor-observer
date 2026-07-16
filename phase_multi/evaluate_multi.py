import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))

import numpy as np
import torch
from pinn_observer_v4 import PINNObserverV4
from data_multi import MEAS_IDX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINNObserverV4().to(device)
model.load_state_dict(torch.load("phase_multi/pinn_multi.pth", map_location=device))
model.eval()

hid = [i for i in range(12) if i not in MEAS_IDX]

def eval_dataset(name, path, idx_range):
    d = np.load(path)
    T, X = d["T"], d["X"]
    Ts, X0s, Xs = [], [], []
    for i in idx_range:
        x0_i = X[i, 0, :]
        for k in range(T.shape[0]):
            Ts.append(T[k]); X0s.append(x0_i); Xs.append(X[i, k, :])
    to_t = lambda a: torch.tensor(np.array(a), dtype=torch.float32).to(device)
    Tt, X0t, Xt = to_t(Ts).unsqueeze(1), to_t(X0s), to_t(Xs)
    with torch.no_grad():
        Xp = model(Tt, X0t)
    rmse = torch.sqrt(((Xp - Xt)**2).mean(dim=0))
    print(f"{name:>22} | meas {rmse[MEAS_IDX].mean().item():.4f} | hidden {rmse[hid].mean().item():.4f}")

print("=== Avg RMSE per trajectory type (TEST = unseen flights 40-49) ===")
for nm, p in [("circle", "datasets/circle_dataset.npz"),
              ("figure8", "datasets/figure8_dataset.npz"),
              ("spiral", "datasets/spiral_dataset.npz")]:
    eval_dataset(f"{nm} TRAIN(0-39)", p, range(0, 40))
    eval_dataset(f"{nm} TEST(40-49)", p, range(40, 50))