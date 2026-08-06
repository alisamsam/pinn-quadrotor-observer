import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1a"))
import numpy as np, torch, csv
from pinn_observer_v4 import PINNObserverV4
from data_phase4 import MEAS_IDX

MODEL_PATH = "phase1a/pinn_4x100.pth"
HID = [i for i in range(12) if i not in MEAS_IDX]
model = PINNObserverV4(hidden=100, n_hidden_layers=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

sets = ["inertia_-20","inertia_-10","inertia_0","inertia_+10","inertia_+20"]
rows = []
for tag in sets:
    d = np.load(f"datasets/mismatch/{tag}.npz")
    T, X, dev = d["T"], d["X"], float(d["dev"])
    me, he = [], []
    for f in range(X.shape[0]):
        x_true = X[f]; x0 = x_true[0]
        Tt = torch.tensor(T, dtype=torch.float32).unsqueeze(1)
        X0t = torch.tensor(np.tile(x0, (len(T),1)), dtype=torch.float32)
        with torch.no_grad():
            x_hat = model(Tt, X0t).numpy()
        rmse = np.sqrt(((x_hat - x_true)**2).mean(axis=0))
        me.append(rmse[MEAS_IDX].mean()); he.append(rmse[HID].mean())
    mr, hr = float(np.mean(me)), float(np.mean(he))
    rows.append([tag, f"{dev*100:+.0f}%", f"{mr:.4f}", f"{hr:.4f}"])
    print(f"{tag:>12} (inertia {dev*100:+.0f}%): meas {mr:.4f} | hidden {hr:.4f}", flush=True)

os.makedirs("docs", exist_ok=True)
with open("docs/mismatch_inertia.csv","w",newline="") as fp:
    w = csv.writer(fp); w.writerow(["set","inertia_deviation","test_meas_RMSE","test_hidden_RMSE"]); w.writerows(rows)
print("\nSaved docs/mismatch_inertia.csv", flush=True)