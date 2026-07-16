import numpy as np
import torch

MEAS_IDX = [0, 1, 2, 6, 7, 8]

DATASETS = ["datasets/circle_dataset.npz",
            "datasets/figure8_dataset.npz",
            "datasets/spiral_dataset.npz"]

def load_multi_data(n_train_traj=40):
    """
    Load circle + figure8 + spiral. Per dataset: first 40 flights train,
    last 10 test. Each sample tagged with its flight's x0.
    """
    def build(idx_list, T, X, U):
        Ts, X0s, Xs, Ys, Us = [], [], [], [], []
        n_steps = T.shape[0]
        for i in idx_list:
            x0_i = X[i, 0, :]
            for k in range(n_steps):
                Ts.append(T[k]); X0s.append(x0_i)
                Xs.append(X[i, k, :]); Ys.append(X[i, k, MEAS_IDX])
                Us.append(U[i, k, :])
        return Ts, X0s, Xs, Ys, Us

    tr = {k: [] for k in ("T","X0","X","Y","U")}
    te = {k: [] for k in ("T","X0","X","Y","U")}

    for path in DATASETS:
        d = np.load(path)
        T, X, U = d["T"], d["X"], d["U"]
        n_traj = X.shape[0]
        a = build(np.arange(0, n_train_traj), T, X, U)
        b = build(np.arange(n_train_traj, n_traj), T, X, U)
        for k, v in zip(("T","X0","X","Y","U"), a): tr[k] += v
        for k, v in zip(("T","X0","X","Y","U"), b): te[k] += v

    to_t = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
    out = lambda d: {
        "T":  to_t(d["T"]).unsqueeze(1),
        "X0": to_t(d["X0"]),
        "X":  to_t(d["X"]),
        "Y":  to_t(d["Y"]),
        "U":  to_t(d["U"]),
    }
    return out(tr), out(te)


if __name__ == "__main__":
    train, test = load_multi_data()
    print("TRAIN:")
    for k, v in train.items(): print(f"  {k}: {tuple(v.shape)}")
    print("TEST:")
    for k, v in test.items(): print(f"  {k}: {tuple(v.shape)}")