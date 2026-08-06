import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))
import numpy as np, torch
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import MEAS_IDX

HID = [i for i in range(12) if i not in MEAS_IDX]
model = PINNObserverV4(hidden=100, n_hidden_layers=4)
model.load_state_dict(torch.load("phase1a/pinn_4x100.pth", map_location="cpu"))
model.eval()

for tag in ["combined_worst_0", "combined_worst"]:
    d = np.load(f"datasets/mismatch/{tag}.npz")
    T, X = d["T"], d["X"]
    me, he = [], []
    for f in range(X.shape[0]):
        x_true = X[f]; x0 = x_true[0]
        Tt = torch.tensor(T, dtype=torch.float32).unsqueeze(1)
        X0t = torch.tensor(np.tile(x0, (len(T),1)), dtype=torch.float32)
        with torch.no_grad():
            x_hat = model(Tt, X0t).numpy()
        rmse = np.sqrt(((x_hat - x_true)**2).mean(axis=0))
        me.append(rmse[MEAS_IDX].mean()); he.append(rmse[HID].mean())
    print(f"{tag:>18}: meas RMSE {np.mean(me):.4f} | hidden RMSE {np.mean(he):.4f}", flush=True)