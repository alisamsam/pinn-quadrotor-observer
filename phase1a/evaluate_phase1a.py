import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from pinn_observer_v2 import PINNObserverV2
from data_phase1a import load_phase1a_data, MEAS_IDX

STATE_NAMES = ["x", "y", "z", "vx", "vy", "vz",
               "phi", "theta", "psi", "p", "q", "r"]

# --- load data + trained model ---
d = load_phase1a_data()
model = PINNObserverV2()
model.load_state_dict(torch.load("phase1a/pinn_phase1a.pth"))
model.eval()


def evaluate(split_name, T, X_true):
    """Compare estimate vs truth for all 12 states, print per-state RMSE."""
    with torch.no_grad():
        X_pred = model(T)                      # (N, 12)
    err = (X_pred - X_true)                     # (N, 12)
    rmse = torch.sqrt((err**2).mean(dim=0))     # (12,) per-state RMSE

    print(f"\n=== {split_name} ===")
    print(f"{'state':>6} | {'RMSE':>10} | {'measured?'}")
    print("-" * 34)
    for i, name in enumerate(STATE_NAMES):
        tag = "measured" if i in MEAS_IDX else "HIDDEN"
        print(f"{name:>6} | {rmse[i].item():>10.4f} | {tag}")
    return rmse


# Evaluate on both halves
rmse_train = evaluate("TRAIN (60%, seen)",  d["T_train"], d["X_train"])
rmse_test  = evaluate("TEST (40%, unseen)", d["T_test"],  d["X_test"])

# Summary: average RMSE over hidden vs measured
hidden_idx = [i for i in range(12) if i not in MEAS_IDX]
print("\n=== SUMMARY (test half) ===")
print(f"Avg RMSE measured states: {rmse_test[MEAS_IDX].mean().item():.4f}")
print(f"Avg RMSE hidden  states: {rmse_test[hidden_idx].mean().item():.4f}")