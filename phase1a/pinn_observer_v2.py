import torch
import torch.nn as nn


class PINNObserverV2(nn.Module):
    """
    Phase 1a PINN observer for the 12-state quadrotor.
    Farkane-style: time-only input.
    Input : [t]              -> 1 number (time only)
    Output: [12 state estimates]
    """
    def __init__(self, n_states=12, hidden=64, n_hidden_layers=2):
        super().__init__()
        input_dim = 1            # TIME ONLY (was 1 + n_states)
        output_dim = n_states    # 12 estimates

        layers = []
        layers.append(nn.Linear(input_dim, hidden))
        layers.append(nn.Tanh())

        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        """
        t : time, shape (batch, 1)
        returns: state estimate, shape (batch, 12)
        """
        return self.net(t)      # no measurements as input


# --- sanity test ---
if __name__ == "__main__":
    model = PINNObserverV2()
    print(model)

    batch = 5
    t_test = torch.zeros(batch, 1)
    out = model(t_test)

    print("\nInput t shape:", t_test.shape)
    print("Output shape: ", out.shape)    # expect (5, 12)