# design_gain_lqe.py  --  Stage 3: DESIGN the observer gain by control theory (LQE),
# instead of learning it. Linearize the quadrotor at hover, solve the steady-state
# Kalman (LQE) equation to get a constant gain L with eig(A0 - L C) < 0 (Hurwitz by
# design), then run the observer x_hat_dot = f(x_hat,u) + L (y - C x_hat) on the 10
# spiral test flights, starting from a WRONG guess of the hidden speeds (set to 0).
# No training -> runs in minutes. Run FROM THE REPO ROOT:  python <this>.py

import sys, os, numpy as np, torch
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "phase1a")
from scipy.linalg import solve_continuous_are
from data_phase4 import load_phase4_data, MEAS_IDX
from dynamics_torch import quadrotor_dynamics_torch

MASS, G = 1.8, 9.81
DATASET = "datasets/spiral_v2_dataset.npz"
NSTEPS = 3000
HID = [i for i in range(12) if i not in MEAS_IDX]

_, test = load_phase4_data(DATASET)

# --- linearize at hover: A0 = df/dx at x=0, u=[mg,0,0,0] ---
x_h = torch.zeros(12)
u_h = torch.tensor([MASS * G, 0.0, 0.0, 0.0])
def f_x(x): return quadrotor_dynamics_torch(x.unsqueeze(0), u_h.unsqueeze(0)).squeeze(0)
A0 = torch.autograd.functional.jacobian(f_x, x_h).numpy()          # (12,12)
C = np.zeros((6, 12))
for r, c in enumerate(MEAS_IDX):
    C[r, c] = 1.0

# observability sanity
O = np.vstack([C @ np.linalg.matrix_power(A0, k) for k in range(12)])
print("rank of observability matrix:", np.linalg.matrix_rank(O), "/ 12", flush=True)

def design_L(qscale, rscale):
    Q = qscale * np.eye(12); R = rscale * np.eye(6)
    P = solve_continuous_are(A0.T, C.T, Q, R)
    return P @ C.T @ np.linalg.inv(R)                              # (12,6)

def run_observer(Lmat, Tf, Uf, Yf, x0f):
    L = torch.tensor(Lmat, dtype=torch.float32)
    Cc = torch.tensor(C, dtype=torch.float32)
    dt = float(Tf[1] - Tf[0])
    x = x0f.clone(); x[HID] = 0.0                                  # wrong hidden start
    traj = [x.clone()]
    for k in range(len(Tf) - 1):
        uk, yk = Uf[k], Yf[k]
        def rhs(xh):
            fdyn = quadrotor_dynamics_torch(xh.unsqueeze(0), uk.unsqueeze(0)).squeeze(0)
            return fdyn + L @ (yk - Cc @ xh)
        k1 = rhs(x); k2 = rhs(x + 0.5*dt*k1); k3 = rhs(x + 0.5*dt*k2); k4 = rhs(x + dt*k3)
        x = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        traj.append(x.clone())
    return torch.stack(traj)

print("\n  q     r    ||L||   maxRe(A0-LC)  stable  hidden_RMSE  hidden_RMSE(>2s)", flush=True)
with torch.no_grad():
    for (q, r) in [(1.0, 1.0), (10.0, 1.0), (100.0, 1.0), (1000.0, 1.0)]:
        try:
            L = design_L(q, r)
        except Exception as e:
            print(f"  q={q} r={r} : ARE failed ({e})", flush=True); continue
        maxre = np.linalg.eigvals(A0 - L @ C).real.max()
        hid, hid2, nst = [], [], 0
        for j in range(10):
            a = j*NSTEPS; b = a + NSTEPS
            Tf = test["T"][a:b].view(-1); Uf = test["U"][a:b]; Yf = test["Y"][a:b]; x0f = test["X0"][a]
            tr = run_observer(L, Tf, Uf, Yf, x0f)
            if torch.isfinite(tr).all():
                nst += 1
                Xf = test["X"][a:b]
                rr = torch.sqrt(((tr - Xf) ** 2).mean(dim=0))
                hid.append(rr[HID].mean().item())
                m2 = torch.sqrt(((tr[200:] - Xf[200:]) ** 2).mean(dim=0))   # after 2 s transient
                hid2.append(m2[HID].mean().item())
        hh = sum(hid)/len(hid) if hid else float("nan")
        h2 = sum(hid2)/len(hid2) if hid2 else float("nan")
        print(f"  {q:6.0f} {r:4.0f}  {np.linalg.norm(L):6.2f}  {maxre:11.3f}   {nst:4d}/10   "
              f"{hh:9.4f}    {h2:9.4f}", flush=True)

print("\nCompare hidden RMSE to: plain PINN 0.056 | learned-gain closed loop (best) 0.132", flush=True)
