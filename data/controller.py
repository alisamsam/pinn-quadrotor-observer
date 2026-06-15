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