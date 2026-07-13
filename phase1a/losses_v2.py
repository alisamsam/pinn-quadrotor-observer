import torch
import torch.nn as nn

# Which states the sensors actually measure:
# [x, y, z, phi, theta, psi] -> indices in the 12-state vector
MEAS_IDX = [0, 1, 2, 6, 7, 8]


def data_loss_v2(model, t, y_measured):
    """
    Phase 1a data loss: compare ONLY the 6 measured states.
    t          : (batch, 1)  time
    y_measured : (batch, 6)  noisy measurements of [x, y, z, phi, theta, psi]
    """
    pred = model(t)                    # (batch, 12) full estimate, time-only input
    pred_measured = pred[:, MEAS_IDX]  # (batch, 6) pick out the measured states
    mse = nn.functional.mse_loss(pred_measured, y_measured)
    return mse


# --- sanity test ---
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.pinn_observer_v2 import PINNObserverV2

    model = PINNObserverV2()
    batch = 5
    t = torch.zeros(batch, 1)
    y_meas = torch.randn(batch, 6)     # fake 6-state measurements

    loss = data_loss_v2(model, t, y_meas)
    print("Data loss (6 measured):", loss.item())
    print("Is it finite?", torch.isfinite(loss).item())