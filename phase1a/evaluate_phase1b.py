import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from pinn_observer_v3 import PINNObserverV3
from data_phase1a import load_phase1a_data, MEAS_IDX

STATE_NAMES = ["x", "y", "z", "vx", "vy", "vz",
               "phi", "theta", "psi", "p", "q", "r"]

d = load_phase1a_data()
model = PINNObserverV3()
model.load_state_dict(torch.load("phase1a/pinn_phase1b_8000.pth"))
model.eval()


def evaluate(split_name, T, X_true):
    with torch.no_grad():
        X_pred, _ = model.get_state_and_gain(T)   # ignore L at eval
    err = X_pred - X_true
    rmse = torch.sqrt((err**2).mean(dim=0))

    print(f"\n=== {split_name} ===")
    print(f"{'state':>6} | {'RMSE':>10} | {'measured?'}")
    print("-" * 34)
    for i, name in enumerate(STATE_NAMES):
        tag = "measured" if i in MEAS_IDX else "HIDDEN"
        print(f"{name:>6} | {rmse[i].item():>10.4f} | {tag}")
    return rmse


rmse_train = evaluate("TRAIN (60%, seen)",  d["T_train"], d["X_train"])
rmse_test  = evaluate("TEST (40%, unseen)", d["T_test"],  d["X_test"])

hidden_idx = [i for i in range(12) if i not in MEAS_IDX]
print("\n=== SUMMARY (test half) ===")
print(f"Avg RMSE measured states: {rmse_test[MEAS_IDX].mean().item():.4f}")
print(f"Avg RMSE hidden  states: {rmse_test[hidden_idx].mean().item():.4f}")