# Phase 1 Findings — Partial-Measurement Adaptive PINN Observer

*Personal working note. Records what I built, what I found, and why. To be shown to Prof. Chaibet after review.*

---

## 1. What Phase 1 set out to do

Extend my PINN observer for the 12-state quadrotor from a full-measurement *denoiser* to a proper *observer* that:

- takes **time only** as input (Farkane-style), not the measurements as input;
- measures only **6 states** (position `x, y, z` and angles `phi, theta, psi`) and must **reconstruct the 6 hidden states** (velocities `vx, vy, vz` and body rates `p, q, r`) from physics;
- is split into **Phase 1a** (partial measurement, plain physics residual) and **Phase 1b** (add the adaptive gain L).

State order: `[x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]`.
Measured indices: `[0, 1, 2, 6, 7, 8]`. Hidden indices: `[3, 4, 5, 9, 10, 11]`.

---

## 2. Setup

- **Data:** one spiral trajectory (flight 0), 3000 time points.
- **Split:** 60 / 40 **contiguous** time split — train on first 1800 points (early time), test on last 1200 (later time). This is an honest *extrapolation* test (the test region is a future time the network never trained on), not easy interpolation.
- **Losses (Farkane's three):** `MSE_y` (data, on the 6 measured states) + `MSE_g` (physics residual) + `MSE_0` (initial condition). Equal weights.
- **Two models compared:**
  - **1a** — network outputs 12 states; residual `= dx_hat/dt - f(x_hat, u)` (no gain).
  - **1b** — network outputs 12 states **+ gain L** (84 outputs = 12 + 12x6); residual `= dx_hat/dt - f(x_hat, u) - L*(y - y_hat)`.
- Both trained for **8000 epochs** (Adam, lr 1e-3) for a fair comparison.

---

## 3. Finding 1 — the adaptive gain L does NOT help (it slightly hurts)

Fair comparison, both at 8000 epochs, **TRAIN half** (seen region), hidden-state RMSE:

| hidden state | 1a (no L) | 1b (with L) |
|---|---|---|
| vx | **0.15** | 2.04 |
| vy | **0.16** | 4.55 |
| vz | **0.07** | 0.97 |
| p  | **0.01** | 0.26 |
| q  | **0.02** | 0.77 |
| r  | **0.01** | 0.13 |

The plain observer (1a) recovers the hidden states far better than the L-augmented one (1b). Adding L made estimation **worse**, not better.

Note: the physics loss itself dropped *more* with L (0.105 -> 0.015), yet the estimate got worse — a warning that **low physics loss does not mean a good estimate**.

### Why L failed (the mechanism)

- In a **classical / Farkane observer**, the observer is an ODE integrated **forward in time**:
  `dx_hat/dt = f(x_hat) + L*(y - y_hat) + Bu`.
  Here L is a **feedback gain**: it shapes the error dynamics `e_dot = (A - L C) e`, so the estimation error **decays over time** as the system evolves. L's power comes entirely from this forward-integration structure.

- In **my PINN parametrization**, the network outputs `x_hat = NN(t)` **directly** as a function of time. There is **no forward integration** and therefore **no error dynamics** for L to shape.

- Consequently `L*(y - y_hat)` degenerates into **72 extra free parameters** added to an algebraic residual. Because the network outputs *both* `x_hat` *and* `L`, it can satisfy the physics equation the **cheap** way — leave `x_hat` slightly wrong and tune `L*(y - y_hat)` to **absorb the leftover mismatch** — instead of the **honest** way of making `x_hat` physically correct. A Phase 1a network (no L) cannot do this: its only way to zero the residual is to actually be correct.

**One-line statement:** *Without forward integration, L stops being a feedback gain and becomes extra freedom that lets the network satisfy the physics residual by absorbing error rather than by producing a correct estimate — which is why adding L lowered the physics loss but degraded state estimation.*

### Honest caveat

This is shown **for my time-only, non-normalized parametrization at 8000 epochs**. It is **not** a claim that Farkane's L is wrong — their L works in their setup. Whether L helps in the *enhanced* `[t, x0]` / integrated formulation is a separate, untested question.

---

## 4. Finding 2 — the observer cannot extrapolate in time

Both models, **TEST half** (unseen later time), position RMSE:

| state | 1a (no L) | 1b (with L) |
|---|---|---|
| x | 14.7 | 15.4 |
| y | 10.4 | 11.8 |
| z | 13.1 | 15.5 |

On the seen region the estimate is near-perfect (train-half `x` RMSE ~0.07); on the unseen region the positions **explode** to 10-15, regardless of L.

### Why — and why it is NOT undertraining

- I ruled out undertraining with a fair test: training from 3000 -> 8000 epochs made the **train half** dramatically better (near-perfect) but did **nothing** for the **test half** explosion. So the failure is **not** a training-budget problem.
- It is **architectural**: the network's only input is **time**. Nothing in its input tells it what the trajectory does *after* the training window. The spiral keeps oscillating (`x = 10 sin wt`) and climbing (`z ~ 2t`), but a time-only network trained on `[0, 18s]` has no way to know that — so its estimate for `[18, 30s]` is meaningless. More epochs only make it memorize the seen window harder (and can worsen extrapolation).
- The angles (`phi, theta, psi`) stay near-constant in this spiral, so they extrapolate fine — which is *consistent* with this explanation.

---

## 5. Implication — both findings point to the enhanced architecture

Both results point to the **same next step**: the enhanced observer from the journal version of Farkane (Section 5.1):

- feed **`[t, x0]`** (time **and** initial condition) as input instead of time alone;
- **normalize** (temporal to [0,1]; standardize states) — this is the same idea as my own sigma-normalization;
- train across **multiple trajectories / initial conditions** to generalize instead of memorizing one path;
- (open design choice) move toward an **integrated** formulation, where L would recover its true feedback role.

This is why the enhanced version exists — my experiments independently reproduced the limitations that motivate it.

---

## 6. Questions for Prof. Chaibet

1. Do these two findings justify **re-prioritizing toward the enhanced `[t, x0]` architecture** (i.e. jumping from Phase 1 toward what I had planned as Phase 4)?
2. Should the enhanced version stay a **direct-output** network, or move to an **integrated** formulation where the gain L acts as a true feedback correction (and might then actually help)?
3. Is the **contiguous time-split** the right evaluation for a time-parametrized observer, given that no such network can extrapolate — or should the evaluation protocol change?

---

## 7. Status of the code

All Phase 1 files are in `phase1a/` and committed to GitHub:
`pinn_observer_v2.py` (1a network), `pinn_observer_v3.py` (1b network with L), `data_phase1a.py`, `losses_v2.py`, `dynamics_torch.py`, `physics_loss.py` (1a residual), `physics_loss_v2.py` (1b residual with correction), the training and evaluation scripts, and the trained `.pth` models (1a/1b at 3000 and 8000 epochs).

---

## 8. Finding 3 — the enhanced [t, x0] architecture SOLVES the generalization failure

After Findings 1 and 2, I rebuilt the observer with the enhanced architecture from Farkane's journal Section 5.1:

- **Input `[t, x0]`** (time **and** initial condition), not time alone — input dimension 1 -> 13.
- **Bigger network** (4 hidden layers x 128 neurons) for the harder task of learning a family of trajectories.
- **Trained across 40 trajectories, tested on 10 unseen ones** (split by trajectory, not by time) — this tests *generalization to new flights*, the right question for `[t, x0]`.
- Trained 2000 epochs on the **university GPU** (mini-batches of 4096; ~120,000 training samples). Plain physics observer (no gain L in this run).

### Result — near-identical accuracy on seen and unseen flights

| metric | TRAIN (40 seen) | TEST (10 UNSEEN) |
|---|---|---|
| avg RMSE, measured states | 0.0415 | 0.0547 |
| avg RMSE, hidden states   | 0.0328 | 0.0449 |

The unseen flights are almost as accurate as the trained ones — a tiny gap. **The observer generalizes.** Hidden states on unseen flights (vx=0.09, vy=0.11, p=0.009) are all recovered well.

### Contrast with the time-only version

- Phase 1a/1b (time-only): test-region positions **exploded to 10-17** — total failure to generalize.
- Phase 4 (`[t, x0]`): test positions x=0.12, y=0.14 — **excellent**.

So conditioning on the initial state **resolves the exact failure** of Findings 1-2. This is the principled reason Farkane's enhanced version exists, confirmed by my own experiments.

### Honest caveats

1. The 10 test flights start **near** the 40 training flights (the dataset's 50 trajectories are small variations around one spiral). "Unseen" here means "slightly different start," not a drastically different trajectory. Generalization is shown **within the training distribution**, not proven for arbitrary flights.
2. This is **one trajectory type** (spiral). True multi-trajectory generalization (circle + figure-8 + spiral in one model) is a further step.
3. **No gain L** in this run. Whether L helps in the `[t, x0]` setup (it hurt in the time-only setup) is still untested.

---

## 9. The complete story so far

1. **Finding 1:** the adaptive gain L does not help in the time-only parametrization — without forward integration it degenerates into extra parameters that absorb error rather than correct the estimate.
2. **Finding 2:** the time-only observer cannot generalize/extrapolate — architectural, not a training-budget issue.
3. **Finding 3:** feeding `[t, x0]` resolves the generalization failure — unseen flights are estimated nearly as well as trained ones.

Together: I found a limitation of the basic method, understood *why*, and fixed it with the principled enhanced architecture.

### Possible next steps
- Test whether gain L helps now that the architecture is `[t, x0]`.
- Push toward true multi-trajectory generalization (circle + figure-8 + spiral in one model).
- Test generalization to flights further from the training distribution.