import torch
import torch.nn as nn


class PINNObserver(nn.Module):
    """
    PINN observer for the 12-state quadrotor.
    Input : [t, 12 noisy measured states]  -> 13 numbers
    Output: [12 clean state estimates]      -> 12 numbers
    """
    def __init__(self, n_states=12, hidden=64, n_hidden_layers=2):
        super().__init__()
        input_dim = 1 + n_states      # time + measured states = 13
        output_dim = n_states         # 12 estimates

        layers = []
        layers.append(nn.Linear(input_dim, hidden))   # input -> first hidden
        layers.append(nn.Tanh())

        for _ in range(n_hidden_layers - 1):          # the remaining hidden layers
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden, output_dim))  # last hidden -> output
        self.net = nn.Sequential(*layers)

    def forward(self, t, y):
        """
        t : time,  shape (batch, 1)
        y : noisy measured states, shape (batch, 12)
        """
        inp = torch.cat([t, y], dim=1)   # glue time + measurements -> (batch, 13)
        return self.net(inp)             # -> (batch, 12)


# --- Step 5.6: sanity test ---
if __name__ == "__main__":
    model = PINNObserver()
    print(model)

    # make a fake batch of 5 samples to check shapes
    batch = 5
    t_test = torch.zeros(batch, 1)        # 5 time values
    y_test = torch.zeros(batch, 12)       # 5 noisy state vectors
    out = model(t_test, y_test)

    print("\nInput t shape:", t_test.shape)
    print("Input y shape:", y_test.shape)
    print("Output shape: ", out.shape)    # expect (5, 12)