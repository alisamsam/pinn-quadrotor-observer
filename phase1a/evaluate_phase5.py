import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from pinn_observer_v5 import PINNObserverV5
from data_phase4 import load_phase4_data, MEAS_IDX

STATE_NAMES = ["x","y","z","vx","vy","vz","phi","theta","psi","p","q","r"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train, test = load_phase4_data()
model = PINNObserverV5().to(device)
model.load_state_dict(torch.load("phase1a/pinn_phase5.pth", map_location=device))
model.eval()


def evaluate(name, data):
    T, X0, X = data["T"].to(device), data["X0"].to(device), data["X"].to(device)
    with torch.no_grad():
        X_pred, _ = model.get_state_and_gain(T, X0)
    rmse = torch.sqrt(((X_pred - X)**2).mean(dim=0))
    print(f"\n=== {name} ===")
    for i, nm in enumerate(STATE_NAMES):
        tag = "measured" if i in MEAS_IDX else "HIDDEN"
        print(f"{nm:>6} | {rmse[i].item():>10.4f} | {tag}")
    hid = [i for i in range(12) if i not in MEAS_IDX]
    print(f"  avg measured: {rmse[MEAS_IDX].mean().item():.4f}")
    print(f"  avg hidden:   {rmse[hid].mean().item():.4f}")

evaluate("TRAIN (40 seen)", train)
evaluate("TEST (10 UNSEEN)", test)