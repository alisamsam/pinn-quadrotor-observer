"""Phase 2: altitude-hold PD controller for the quadrotor."""
import numpy as np

# physical params (same as Singha, Step 1)
M, G = 1.80, 9.81

# ===== 2.1 the setpoint =====
Z_TARGET = 2.0          # hold altitude at 2 metres

# ===== 2.2 the PD controller (altitude only) =====
KP_Z = 4.0              # proportional gain (push toward target)
KD_Z = 3.0              # derivative gain (brake / damping)


def altitude_pd_control(x, z_target=Z_TARGET):
    """
    Compute control u = [U1, U2, U3, U4] to hold altitude z_target.
    For now: only U1 (thrust) is active; torques = 0.
    x : state vector (12,)
    """
    z    = x[2]         # current altitude
    zdot = x[5]         # current vertical velocity

    # PD correction + gravity compensation
    U1 = M * G + KP_Z * (z_target - z) - KD_Z * zdot

    # altitude-only: no attitude control yet
    return np.array([U1, 0.0, 0.0, 0.0])


# ===== quick standalone test =====
if __name__ == "__main__":
    # drone sitting at z=0, at rest -> should command MORE than hover thrust
    x_low = np.zeros(12)
    u = altitude_pd_control(x_low)
    print("At z=0 (below target):  U1 =", round(u[0], 3), " (expect > mg=17.658)")

    # drone AT target, at rest -> should command exactly hover thrust
    x_at = np.zeros(12); x_at[2] = Z_TARGET
    u = altitude_pd_control(x_at)
    print("At z=2 (at target):     U1 =", round(u[0], 3), " (expect = mg=17.658)")

    # drone ABOVE target -> should command LESS than hover
    x_high = np.zeros(12); x_high[2] = 3.0
    u = altitude_pd_control(x_high)
    print("At z=3 (above target):  U1 =", round(u[0], 3), " (expect < mg=17.658)")

# ============================================================
# B.2 — OUTER LOOP: position error -> desired tilt (phi_d, theta_d)
# ============================================================
KP_XY = 0.6        # horizontal position gain
KD_XY = 1.0        # horizontal velocity damping

X_TARGET = 0.0     # setpoint x
Y_TARGET = 0.0     # setpoint y


def outer_loop(x, U1, x_target=X_TARGET, y_target=Y_TARGET):
    """
    OUTER LOOP: from horizontal position error, compute desired tilt.
    Returns (phi_d, theta_d) — the tilt the inner loop must achieve.
    """
    px, py = x[0], x[1]          # current x, y position
    vx, vy = x[3], x[4]          # current x, y velocity
    psi    = x[8]                # current yaw

    # PD "effort" in x and y (setpoint: desired vel/accel = 0)
    Uex = (M / U1) * (KP_XY * (x_target - px) - KD_XY * vx)
    Uey = (M / U1) * (KP_XY * (y_target - py) - KD_XY * vy)

    # clip to valid arcsin domain [-1, 1] (safety against blow-up)
    Uex = np.clip(Uex, -0.99, 0.99)
    Uey = np.clip(Uey, -0.99, 0.99)

    # eqs (17)-(18): desired tilt angles
    phi_d   = np.arcsin(Uex * np.sin(psi) - Uey * np.cos(psi))
    phi     = x[6]
    theta_d = np.arcsin(np.clip(
        (Uex - np.sin(phi) * np.sin(psi)) / (np.cos(phi) * np.cos(psi)),
        -0.99, 0.99))

    return phi_d, theta_d


# --- test B.2 ---
if __name__ == "__main__":
    U1_hover = M * G

    # drone at origin, level, at rest -> desired tilt should be ~0
    x_centered = np.zeros(12)
    phi_d, theta_d = outer_loop(x_centered, U1_hover)
    print("At target, at rest:   phi_d =", round(phi_d,4), " theta_d =", round(theta_d,4), " (expect ~0)")

    # drone 1m to the +x side -> should command a tilt to push it back
    x_offset = np.zeros(12); x_offset[0] = 1.0
    phi_d, theta_d = outer_loop(x_offset, U1_hover)
    print("Offset +1m in x:      phi_d =", round(phi_d,4), " theta_d =", round(theta_d,4), " (expect nonzero)")

    # drone moving fast in +x -> damping should command opposite tilt
    x_moving = np.zeros(12); x_moving[3] = 1.0
    phi_d, theta_d = outer_loop(x_moving, U1_hover)
    print("Moving +1 m/s in x:   phi_d =", round(phi_d,4), " theta_d =", round(theta_d,4), " (expect nonzero)")

# ============================================================
# B.3 — INNER LOOP: tilt error -> torque (U2, U3, U4)
# ============================================================
IX, IY, IZ, L_ARM = 0.03, 0.03, 0.04, 0.20
KP_ATT = 8.0       # attitude (tilt) gain
KD_ATT = 4.0       # attitude rate damping
PSI_TARGET = 0.0   # desired yaw


def inner_loop(x, phi_d, theta_d, psi_d=PSI_TARGET):
    """
    INNER LOOP: from tilt error, compute torques to achieve desired tilt.
    Returns (U2, U3, U4).
    """
    phi, theta, psi = x[6], x[7], x[8]          # actual angles
    p, q, r         = x[9], x[10], x[11]        # actual angular rates

    # roll torque (eq 26): drive phi -> phi_d, with Coriolis correction
    U2 = (IX / L_ARM) * (KP_ATT*(phi_d - phi) - KD_ATT*p - q*r*(IY-IZ)/IX)
    # pitch torque (eq 33)
    U3 = (IY / L_ARM) * (KP_ATT*(theta_d - theta) - KD_ATT*q - p*r*(IZ-IX)/IY)
    # yaw torque (eq 34)
    U4 = IZ * (KP_ATT*(psi_d - psi) - KD_ATT*r - p*q*(IX-IY)/IZ)

    return U2, U3, U4


# --- test B.3 ---
if __name__ == "__main__":
    # level, at rest, no desired tilt -> torques ~0
    x0 = np.zeros(12)
    print("Level, no cmd:        ", [round(v,4) for v in inner_loop(x0, 0.0, 0.0)], "(expect ~0)")

    # commanded to tilt (phi_d=0.1) but currently level -> should command roll torque
    print("Cmd roll 0.1, at 0:   ", [round(v,4) for v in inner_loop(x0, 0.1, 0.0)], "(expect U2>0)")

    # commanded to tilt in pitch -> should command pitch torque
    print("Cmd pitch 0.1, at 0:  ", [round(v,4) for v in inner_loop(x0, 0.0, 0.1)], "(expect U3>0)")

# ============================================================
# B.4 — FULL CONTROLLER: chain U1 + outer + inner
# ============================================================
def full_control(x, x_target=X_TARGET, y_target=Y_TARGET, z_target=Z_TARGET):
    """
    Full position+attitude controller.
    Returns u = [U1, U2, U3, U4].
    """
    # 1) thrust for altitude (B.1)
    z, zdot = x[2], x[5]
    U1 = M * G + KP_Z * (z_target - z) - KD_Z * zdot

    # 2) OUTER loop: position -> desired tilt (B.2)
    phi_d, theta_d = outer_loop(x, U1, x_target, y_target)

    # 3) INNER loop: tilt -> torques (B.3)
    U2, U3, U4 = inner_loop(x, phi_d, theta_d)

    return np.array([U1, U2, U3, U4])


# --- test B.4 ---
if __name__ == "__main__":
    # at target, level, at rest -> hover thrust, ~zero torques
    x_at = np.zeros(12); x_at[2] = Z_TARGET
    u = full_control(x_at)
    print("At full target:", [round(v,4) for v in u], "(expect [17.658, ~0, ~0, ~0])")

    # off in x and below target -> thrust up + a pitch torque
    x_off = np.zeros(12); x_off[0] = 1.0
    u = full_control(x_off)
    print("Off +1m x:     ", [round(v,4) for v in u], "(expect U1>mg-ish, U3 nonzero)")