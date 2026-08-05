import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))

import numpy as np
import torch
import csv
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import MEAS_IDX

MODEL_PATH = "phase1a/pinn_4x100.pth"
LAYERS, HIDDEN = 4, 100
HID = [i for i in range(12) if i not in MEAS_IDX]

# load the frozen observer (assumes nominal mass 1.80, baked in during training)
model = PINNObserverV4(hidden=HIDDEN, n_hidden_layers=LAYERS)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

sets = ["mass_-20", "mass_-10", "mass_0", "mass_+10", "mass_+20"]
rows = []
for tag in sets:
    d = np.load(f"datasets/mismatch/{tag}.npz")
    T, X, dev = d["T"], d["X"], float(d["dev"])
    n_flights = X.shape[0]
    meas_errs, hid_errs = [], []
    for f in range(n_flights):
        x_true = X[f]
        x0 = x_true[0]
        Tt  = torch.tensor(T, dtype=torch.float32).unsqueeze(1)
        X0t = torch.tensor(np.tile(x0, (len(T), 1)), dtype=torch.float32)
        with torch.no_grad():
            x_hat = model(Tt, X0t).numpy()
        rmse = np.sqrt(((x_hat - x_true)**2).mean(axis=0))
        meas_errs.append(rmse[MEAS_IDX].mean())
        hid_errs.append(rmse[HID].mean())
    mr, hr = float(np.mean(meas_errs)), float(np.mean(hid_errs))
    rows.append([tag, f"{dev*100:+.0f}%", f"{mr:.4f}", f"{hr:.4f}"])
    print(f"{tag:>9} (mass {dev*100:+.0f}%): meas RMSE {mr:.4f} | hidden RMSE {hr:.4f}", flush=True)

os.makedirs("docs", exist_ok=True)
with open("docs/mismatch_mass.csv", "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["set", "mass_deviation", "test_meas_RMSE", "test_hidden_RMSE"])
    w.writerows(rows)
print("\nSaved docs/mismatch_mass.csv", flush=True)