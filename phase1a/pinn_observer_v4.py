import torch
import torch.nn as nn


class PINNObserverV4(nn.Module):
    """
    Phase 4 enhanced observer: input is [t, x0] (time + initial condition).
    Input : [t, x0]  -> 1 + 12 = 13 numbers
    Output: [12 state estimates]   (gain L can be added later)
    """
    def __init__(self, n_states=12, hidden=128, n_hidden_layers=4):
        super().__init__()
        self.n_states = n_states
        input_dim  = 1 + n_states      # time + initial condition = 13
        output_dim = n_states          # 12 estimates

        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, t, x0):
        """
        t  : (batch, 1)  time
        x0 : (batch, 12) initial condition for each sample
        """
        inp = torch.cat([t, x0], dim=1)   # (batch, 13)
        return self.net(inp)              # (batch, 12)


# --- sanity test ---
if __name__ == "__main__":
    model = PINNObserverV4()
    print(model)
    batch = 5
    t  = torch.zeros(batch, 1)
    x0 = torch.zeros(batch, 12)
    out = model(t, x0)
    print("\nInput t shape: ", t.shape)
    print("Input x0 shape:", x0.shape)
    print("Output shape:  ", out.shape)   # expect (5, 12)