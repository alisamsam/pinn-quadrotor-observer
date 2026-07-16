import torch
import torch.nn as nn


class PINNObserverV5(nn.Module):
    """
    [t, x0] input + gain L output. Combines v4 (t,x0 input) and v3 (L output).
    Input : [t, x0]                      -> 13
    Output: [12 states] + [12x6 gain L]  -> 84
    """
    def __init__(self, n_states=12, n_meas=6, hidden=128, n_hidden_layers=4):
        super().__init__()
        self.n_states = n_states
        self.n_meas   = n_meas
        input_dim  = 1 + n_states
        output_dim = n_states + n_states * n_meas   # 84

        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, t, x0):
        inp = torch.cat([t, x0], dim=1)
        return self.net(inp)

    def get_state_and_gain(self, t, x0):
        out = self.forward(t, x0)
        x_hat = out[:, :self.n_states]                       # (batch, 12)
        L = out[:, self.n_states:].view(-1, self.n_states, self.n_meas)  # (batch,12,6)
        return x_hat, L


# --- sanity test ---
if __name__ == "__main__":
    model = PINNObserverV5()
    batch = 5
    t  = torch.zeros(batch, 1)
    x0 = torch.zeros(batch, 12)
    x_hat, L = model.get_state_and_gain(t, x0)
    print("x_hat:", x_hat.shape)   # (5, 12)
    print("L:    ", L.shape)       # (5, 12, 6)