import numpy as np
import torch

# Which states the sensors measure: [x, y, z, phi, theta, psi]
MEAS_IDX = [0, 1, 2, 6, 7, 8]

def load_phase1a_data(npz_path="datasets/spiral_dataset.npz",
                      traj_index=0, train_frac=0.60):
    """
    Load ONE trajectory, split its time axis 60/40 (contiguous),
    and extract the 6 measured states.
    """
    data = np.load(npz_path)
    T = data["T"]              # (3000,)
    X = data["X"][traj_index]  # (3000, 12) -> one trajectory's full states
    U = data["U"][traj_index]  # (3000, 4)  -> one trajectory's control inputs

    n_steps = T.shape[0]                    # 3000
    n_train = int(train_frac * n_steps)     # 1800 (60%)

    # Contiguous split: first 60% train, last 40% test
    T_train, T_test = T[:n_train], T[n_train:]
    X_train, X_test = X[:n_train], X[n_train:]
    U_train, U_test = U[:n_train], U[n_train:]

    # Extract the 6 measured states (what the sensor sees)
    Y_train = X_train[:, MEAS_IDX]   # (1800, 6)
    Y_test  = X_test[:,  MEAS_IDX]   # (1200, 6)

    # Convert to tensors
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    T_train = to_t(T_train).unsqueeze(1)   # (1800, 1)
    T_test  = to_t(T_test).unsqueeze(1)    # (1200, 1)
    X_train, X_test = to_t(X_train), to_t(X_test)   # full 12 (for eval)
    Y_train, Y_test = to_t(Y_train), to_t(Y_test)   # measured 6 (for loss)
    U_train, U_test = to_t(U_train), to_t(U_test)   # controls (for physics loss)

    return {
        "T_train": T_train, "T_test": T_test,
        "Y_train": Y_train, "Y_test": Y_test,   # 6 measured -> data loss
        "U_train": U_train, "U_test": U_test,   # controls -> physics loss
        "X_train": X_train, "X_test": X_test,   # full 12 -> evaluation only
    }


# --- sanity test ---
if __name__ == "__main__":
    d = load_phase1a_data()
    for k, v in d.items():
        print(f"{k}: {tuple(v.shape)}")