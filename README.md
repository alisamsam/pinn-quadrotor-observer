# PINN-Quadrotor

Physics-Informed Neural Network observer for a 12-state autonomous quadrotor.

## Context

End-of-studies internship project at **Laboratoire DRIVE**, Université Bourgogne Europe (Nevers), April–September 2026.
Supervisor: **Prof. Ahmed Chaibet**.

## Project goal

Design and train a PINN observer that estimates the full 12-state vector of an autonomous quadrotor from noisy partial sensor measurements, leveraging the physics of rigid-body flight dynamics (Singha et al. 2024) as a soft constraint inside the neural-network loss function.

## Project structure

- `data/` — Simulator and dataset generation
- `models/` — Neural network architectures (vanilla MLP + PINN)
- `training/` — Training and evaluation scripts
- `notebooks/` — Exploratory analyses
- `docs/` — Notes, derivations, references

## References

- Farkane et al. (2025), *PINN-Obs: Physics-Informed Neural Network Observer*, arXiv:2507.06712
- Singha, Thakur & Ray (2024), *Lyapunov-based trajectory tracking controller for a quadrotor UAV with nonholonomic constraints*
- Wang, Teng & Perdikaris (2021), *Understanding and Mitigating Gradient Pathologies in Physics-Informed Neural Networks*

## Related work

The pendulum prototype that validated the PINN architecture lives in a separate repository: `pinn-pendulum`.

## Author

Samsaam Ali Baig — M2 Mechatronics, Polytech Annecy.