"""core/controller_v3.py

Same cascade as controller_v2, but the gains are chosen by a principled rule
instead of a sweep. Each loop is 2nd order, so a PD controller gives
    e'' + kd e' + kp e = 0   ->   kp = wn^2 ,  kd = 2 * zeta * wn
We pick wn (bandwidth) and zeta (damping), then COMPUTE kp, kd.

Design choices:
  * zeta = 1.0 on every loop  -> critically damped, no ringing.
  * inner (attitude) loop 3.5x faster than the outer (position) loop
    -> proper time-scale separation, so the loops don't fight (kills the
       velocity ripple / attitude ring seen with controller_v2).

  loop          wn     kp=wn^2   kd=2*wn   zeta
  position xy   2.0    4.0       4.0       1.0
  altitude z    3.0    9.0       6.0       1.0
  attitude      7.0    49.0      14.0      1.0
  yaw           3.0    9.0       6.0       1.0
"""
import numpy as np
from core.generate_quadrotor_data import m as M, g as G, l as L_ARM, Ix as IX, Iy as IY, Iz as IZ

DEFAULT_GAINS = dict(
    kp_xy=4.0,  kd_xy=4.0,     # wn=2.0, zeta=1.0
    kp_z=9.0,   kd_z=6.0,      # wn=3.0, zeta=1.0
    kp_att=49.0, kd_att=14.0,  # wn=7.0, zeta=1.0  (3.5x the position loop)
    kp_yaw=9.0,  kd_yaw=6.0,   # wn=3.0, zeta=1.0
)

LIMITS = dict(
    u1_min=1.0, u1_max=35.0,
    tilt_max=np.radians(30.0),
    tau_max=5.0,
)


def tilt_from_accel(Uex, Uey, psi):
    sin_phi = np.clip(Uex * np.sin(psi) - Uey * np.cos(psi), -1.0, 1.0)
    phi_d = np.arcsin(sin_phi)
    sin_theta = np.clip((Uex * np.cos(psi) + Uey * np.sin(psi)) / np.cos(phi_d), -1.0, 1.0)
    theta_d = np.arcsin(sin_theta)
    return phi_d, theta_d


def full_control_v3(t, x, ref_fn, yaw_fn, gains=None, limits=None, saturate=True):
    g = DEFAULT_GAINS if gains is None else {**DEFAULT_GAINS, **gains}
    lim = LIMITS if limits is None else {**LIMITS, **limits}

    pos_d, vel_d, acc_d = ref_fn(t)
    psi_d, psi_d_dot = yaw_fn(t)
    phi, theta, psi = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]

    az = acc_d[2] + g['kp_z'] * (pos_d[2] - x[2]) + g['kd_z'] * (vel_d[2] - x[5])
    den = np.cos(phi) * np.cos(theta)
    den = np.sign(den) * max(abs(den), 0.25)
    U1 = M * (G + az) / den
    if saturate:
        U1 = float(np.clip(U1, lim['u1_min'], lim['u1_max']))

    ax = acc_d[0] + g['kp_xy'] * (pos_d[0] - x[0]) + g['kd_xy'] * (vel_d[0] - x[3])
    ay = acc_d[1] + g['kp_xy'] * (pos_d[1] - x[1]) + g['kd_xy'] * (vel_d[1] - x[4])

    phi_d, theta_d = tilt_from_accel((M / U1) * ax, (M / U1) * ay, psi)
    if saturate:
        phi_d = float(np.clip(phi_d, -lim['tilt_max'], lim['tilt_max']))
        theta_d = float(np.clip(theta_d, -lim['tilt_max'], lim['tilt_max']))

    U2 = (IX / L_ARM) * (g['kp_att'] * (phi_d - phi) - g['kd_att'] * p - q * r * (IY - IZ) / IX)
    U3 = (IY / L_ARM) * (g['kp_att'] * (theta_d - theta) - g['kd_att'] * q - p * r * (IZ - IX) / IY)
    U4 = IZ * (g['kp_yaw'] * (psi_d - psi) + g['kd_yaw'] * (psi_d_dot - r) - p * q * (IX - IY) / IZ)
    if saturate:
        U2 = float(np.clip(U2, -lim['tau_max'], lim['tau_max']))
        U3 = float(np.clip(U3, -lim['tau_max'], lim['tau_max']))
        U4 = float(np.clip(U4, -lim['tau_max'], lim['tau_max']))

    return np.array([U1, U2, U3, U4])
