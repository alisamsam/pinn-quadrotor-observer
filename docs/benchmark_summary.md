# Observer Benchmark — Spiral Trajectory (Summary)

**Setup (identical for all methods).** Same 10 unseen test flights, nominal Singha model, true initial state `x0` given to every observer, per-state RMSE (measured vs hidden). One structural difference that must be stated: the classical observers (Luenberger, EKF, UKF) consume the **live 6-state measurement stream at every step**; the PINN uses **only time `t` and the initial condition `x0`** at inference — it never sees the measurements. So they solve different estimation problems.

## Headline numbers (RMSE)

| Condition | PINN meas/hid | Luenberger | EKF | UKF |
|---|---|---|---|---|
| Clean | 0.044 / 0.056 | 0.010 / 0.009 | 0.000 / 0.008 | 0.000 / 0.013 |
| Strong noise | 0.085 / 0.072 | 0.018 / 0.037 | 0.011 / 0.017 | 0.011 / 0.020 |
| Combined-worst mismatch | 0.338 / 0.696 | 0.019 / 0.246 | 0.001 / 0.677 | 0.001 / 0.677 |

## Findings (honest)

1. **Measured states — classical observers dominate everywhere.** EKF/UKF sit at ~0 and the Luenberger at ~0.01, because they continuously correct from the live sensors. The PINN, using only `t` and `x0`, is far behind on measured states. This is expected and should be stated plainly.

2. **Clean/noise hidden states — classical still better, but the gap narrows** and the PINN needs no per-noise-level tuning, whereas a fixed-gain Luenberger amplifies noise into the hidden states if not re-tuned.

3. **Model mismatch — the picture flips on hidden states.** Measured stays ~0 for the classical filters, but on the *unmeasured* states the two Kalman filters (EKF, UKF) become **brittle**: hidden RMSE reaches ~0.49 at mass ±20% and ~0.68 at combined-worst — comparable to, and in the mass case *worse than*, the PINN (0.35). Only the well-damped Luenberger stays clearly ahead (0.12 / 0.25). So the PINN is **competitive with the EKF/UKF exactly where they are weakest**.

## Interpretation / positioning

- When **continuous, accurate measurements are available**, a classical model-based observer is simpler and more accurate — that is their setting, and the benchmark confirms it.
- The PINN's contribution is **structural**: it reconstructs the full 12-state from **only the initial condition + physics, with no live measurement stream** — a regime where the classical observers cannot operate at all. It is also **learned from data**, useful when an accurate analytic model is unavailable.

## Honest caveats

- The EKF/UKF used a fixed process-noise `Q` tuned for the nominal model; inflating `Q` under mismatch would reduce their hidden-state brittleness (i.e. their mismatch result can be improved by re-tuning). The comparison uses standard, out-of-the-box tuning.
- Classical observers use the live measurement stream; the PINN does not — a difference in task, not just accuracy.
- UKF used Euler sigma-propagation and the EKF used the standard `F = I + Δt·A` Jacobian; both are standard and their results are mutually consistent.

## Recommended future work

- An **integrated (recursive) PINN observer** that ingests measurements online — combining learned physics with continuous correction; this is the formulation in which the adaptive gain `L` (which hurt in the current direct-output PINN) would finally help.
- A **measurement-dropout** comparison: when sensors go intermittent, classical observers drift while the PINN — needing no live measurements — keeps producing estimates. This is where the PINN's structural advantage becomes a measured win.
