import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from pinn_observer_v3 import PINNObserverV3
from data_phase1a import load_phase1a_data

d = load_phase1a_data()
model = PINNObserverV3()
model.load_state_dict(torch.load("phase1a/pinn_phase1b_8000.pth"))
model.eval()

with torch.no_grad():
    _, L = model.get_state_and_gain(d["T_train"])   # (1800, 12, 6)

# Look at L's magnitude
print("L shape:", tuple(L.shape))
print(f"L min:  {L.min().item():.4f}")
print(f"L max:  {L.max().item():.4f}")
print(f"L mean: {L.mean().item():.4f}")
print(f"L abs mean (typical size): {L.abs().mean().item():.4f}")

# Does L change over time, or is it basically constant?
L_std_over_time = L.std(dim=0).mean().item()   # how much L varies across time
print(f"L variation over time (std): {L_std_over_time:.4f}")

# Peek at L at the first time point (the 12x6 matrix)
print("\nL at t=0 (12x6 matrix):")
print(torch.round(L[0], decimals=2))