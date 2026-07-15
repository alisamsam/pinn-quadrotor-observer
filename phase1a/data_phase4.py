import numpy as np
import torch

MEAS_IDX = [0, 1, 2, 6, 7, 8]

def load_phase4_data(npz_path="datasets/spiral_dataset.npz", n_train_traj=40):
    """
    Load ALL trajectories, split by TRAJECTORY (40 train / 10 test).
    Each time point is tagged with its trajectory's initial condition x0.
    """
    data = np.load(npz_path)
    T = data["T"]        # (3000,)
    X = data["X"]        # (50, 3000, 12)
    U = data["U"]        # (50, 3000, 4)

    n_traj, n_steps, _ = X.shape          # 50, 3000, 12

    # split trajectories: first 40 train, last 10 test
    train_idx = np.arange(0, n_train_traj)          # 0..39
    test_idx  = np.arange(n_train_traj, n_traj)     # 40..49

    def build(idx_list):
        Ts, X0s, Xs, Ys, Us = [], [], [], [], []
        for i in idx_list:
            x0_i = X[i, 0, :]                       # (12,) this flight's start
            for k in range(n_steps):
                Ts.append(T[k])                      # time
                X0s.append(x0_i)                     # SAME x0 for all steps of flight i
                Xs.append(X[i, k, :])                # full 12 state (eval)
                Ys.append(X[i, k, MEAS_IDX])         # 6 measured
                Us.append(U[i, k, :])                # 4 controls
        to_t = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
        return {
            "T":  to_t(Ts).unsqueeze(1),   # (N, 1)
            "X0": to_t(X0s),               # (N, 12)
            "X":  to_t(Xs),                # (N, 12) full (eval)
            "Y":  to_t(Ys),                # (N, 6) measured
            "U":  to_t(Us),                # (N, 4) controls
        }

    return build(train_idx), build(test_idx)


# --- sanity test ---
if __name__ == "__main__":
    train, test = load_phase4_data()
    print("TRAIN:")
    for k, v in train.items():
        print(f"  {k}: {tuple(v.shape)}")
    print("TEST:")
    for k, v in test.items():
        print(f"  {k}: {tuple(v.shape)}")