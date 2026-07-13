import torch
import torch.nn as nn


class PINNObserverV3(nn.Module):
    """
    Phase 1b PINN observer: outputs BOTH state estimate AND gain L.
    Farkane-style, time-only input.
    Input : [t]                          -> 1 number
    Output: [12 states] + [12x6 gain L]  -> 84 numbers
    """
    def __init__(self, n_states=12, n_meas=6, hidden=64, n_hidden_layers=2):
        super().__init__()
        self.n_states = n_states
        self.n_meas   = n_meas

        input_dim  = 1
        output_dim = n_states + n_states * n_meas   # 12 + 72 = 84

        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        """Returns the raw 84-dim output."""
        return self.net(t)

    def get_state_and_gain(self, t):
        """
        Split the output into state estimate and gain matrix.
        Returns:
          x_hat : (batch, 12)
          L     : (batch, 12, 6)
        """
        out = self.net(t)                          # (batch, 84)
        x_hat = out[:, :self.n_states]             # (batch, 12)
        L_flat = out[:, self.n_states:]            # (batch, 72)
        L = L_flat.view(-1, self.n_states, self.n_meas)  # (batch, 12, 6)
        return x_hat, L


# --- sanity test ---
if __name__ == "__main__":
    model = PINNObserverV3()
    print(model)
    batch = 5
    t = torch.zeros(batch, 1)

    out = model(t)
    x_hat, L = model.get_state_and_gain(t)
    print("\nRaw output shape:", out.shape)        # (5, 84)
    print("x_hat shape:     ", x_hat.shape)        # (5, 12)
    print("L shape:         ", L.shape)            # (5, 12, 6)