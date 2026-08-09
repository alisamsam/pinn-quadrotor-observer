"""core/controller_v2.py

Improved trajectory-tracking controller for the Singha 12-state quadrotor.
Full definitions and formulas: docs/Controller_v2_Reference.docx.

Upgrades over core/controller.py:
  Step B  acceleration feedforward + PD   (tracks a moving target without lag)
  Step C  thrust with tilt-compensation   (U1 divided by cos(phi)cos(theta))
  Step D  exact acceleration->tilt inversion (fixes the arcsin domain bug)
  Step E  attitude loop with spin-coupling correction
  gains   tuned by a gain sweep (KP_XY=4.5, KD_XY=5.0)
  limits  realistic saturation on thrust, tilt, and torque

Trajectory-agnostic. Provide:
  ref_fn(t) -> (pos_d, vel_d, acc_d)   each a length-3 array [x,y,z]
  yaw_fn(t) -> (psi_d, psi_d_dot)      desired yaw and its rate
So it works for spiral, circle, figure-8, etc.

State  x (12): [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
Return u (4):  [U1, U2, U3, U4]  (thrust, roll/pitch/yaw torques)
"""
import numpy as np
from core.generate_quadrotor_data import m as M, g as G, l as L_ARM, Ix as IX, Iy as IY, Iz as IZ


# --- default tuned gains (from the gain sweep) ---
DEFAULT_GAINS = dict(
    kp_xy=4.5, kd_xy=5.0,
    kp_z=6.0,  kd_z=5.0,
    kp_att=25.0, kd_att=8.0,
    kp_yaw=2.0,  kd_yaw=2.0,
)

# --- realistic actuator limits (m=1.80 kg, thrust-to-weight ~2) ---
LIMITS = dict(
    u1_min=1.0, u1_max=35.0,        # thrust [N]
    tilt_max=np.radians(30.0),      # |phi_d|, |theta_d|
    tau_max=5.0,                    # |U2|, |U3|, |U4|
)


def tilt_from_accel(Uex, Uey, psi):
    """Step D: exact acceleration->tilt inversion. Returns (phi_d, theta_d).
    Derived by rotating the accel equations by yaw; arcsin args clamped to
    [-1, 1] so the map can never produce NaN.
    """
    sin_phi = np.clip(Uex * np.sin(psi) - Uey * np.cos(psi), -1.0, 1.0)
    phi_d = np.arcsin(sin_phi)
    sin_theta = np.clip((Uex * np.cos(psi) + Uey * np.sin(psi)) / np.cos(phi_d), -1.0, 1.0)
    theta_d = np.arcsin(sin_theta)
    return phi_d, theta_d


def full_control_v2(t, x, ref_fn, yaw_fn, gains=None, limits=None, saturate=True):
    """Improved position+attitude controller. Returns u = [U1, U2, U3, U4]."""
    g = DEFAULT_GAINS if gains is None else {**DEFAULT_GAINS, **gains}
    lim = LIMITS if limits is None else {**LIMITS, **limits}

    pos_d, vel_d, acc_d = ref_fn(t)
    psi_d, psi_d_dot = yaw_fn(t)
    phi, theta, psi = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]

    # Step C: thrust with tilt-compensation
    az = acc_d[2] + g['kp_z'] * (pos_d[2] - x[2]) + g['kd_z'] * (vel_d[2] - x[5])
    den = np.cos(phi) * np.cos(theta)
    den = np.sign(den) * max(abs(den), 0.25)
    U1 = M * (G + az) / den
    if saturate:
        U1 = float(np.clip(U1, lim['u1_min'], lim['u1_max']))

    # Step B: horizontal acceleration (feedforward + PD)
    ax = acc_d[0] + g['kp_xy'] * (pos_d[0] - x[0]) + g['kd_xy'] * (vel_d[0] - x[3])
    ay = acc_d[1] + g['kp_xy'] * (pos_d[1] - x[1]) + g['kd_xy'] * (vel_d[1] - x[4])

    # Step D: accel -> desired tilt (exact inversion)
    phi_d, theta_d = tilt_from_accel((M / U1) * ax, (M / U1) * ay, psi)
    if saturate:
        phi_d = float(np.clip(phi_d, -lim['tilt_max'], lim['tilt_max']))
        theta_d = float(np.clip(theta_d, -lim['tilt_max'], lim['tilt_max']))

    # Step E: torques (P + rate damping + spin-coupling correction)
    U2 = (IX / L_ARM) * (g['kp_att'] * (phi_d - phi) - g['kd_att'] * p - q * r * (IY - IZ) / IX)
    U3 = (IY / L_ARM) * (g['kp_att'] * (theta_d - theta) - g['kd_att'] * q - p * r * (IZ - IX) / IY)
    U4 = IZ * (g['kp_yaw'] * (psi_d - psi) + g['kd_yaw'] * (psi_d_dot - r) - p * q * (IX - IY) / IZ)
    if saturate:
        U2 = float(np.clip(U2, -lim['tau_max'], lim['tau_max']))
        U3 = float(np.clip(U3, -lim['tau_max'], lim['tau_max']))
        U4 = float(np.clip(U4, -lim['tau_max'], lim['tau_max']))

    return np.array([U1, U2, U3, U4])


# --- sanity test: hovering at a fixed point should command ~hover thrust, ~0 torque ---
if __name__ == "__main__":
    hover_ref = lambda t: (np.array([0.0, 0.0, 2.0]), np.zeros(3), np.zeros(3))
    hover_yaw = lambda t: (0.0, 0.0)
    x_at = np.zeros(12); x_at[2] = 2.0            # at target height, level, at rest
    u = full_control_v2(0.0, x_at, hover_ref, hover_yaw)
    print("hover u:", np.round(u, 4), " (expect U1~17.66, torques~0)")
