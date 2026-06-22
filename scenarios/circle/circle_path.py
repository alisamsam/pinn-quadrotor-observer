"""T.1: define the circular reference trajectory and its derivatives."""
import numpy as np

# circle parameters
R = 2.0          # radius (metres)
OMEGA = 0.5      # angular speed (rad/s) -> period = 2pi/omega ~ 12.6 s
Z_HEIGHT = 3.0   # constant flight height


def circle_reference(t):
    """
    Returns the desired position, velocity, acceleration at time t.
    This is the MOVING target the drone must follow.
    """
    # position (where the target is now)
    x_d = R * np.cos(OMEGA * t)
    y_d = R * np.sin(OMEGA * t)
    z_d = Z_HEIGHT

    # velocity (how fast the target is moving) - 1st derivative
    xdot_d = -R * OMEGA * np.sin(OMEGA * t)
    ydot_d =  R * OMEGA * np.cos(OMEGA * t)
    zdot_d = 0.0

    # acceleration (how the target's velocity changes) - 2nd derivative
    xddot_d = -R * OMEGA**2 * np.cos(OMEGA * t)
    yddot_d = -R * OMEGA**2 * np.sin(OMEGA * t)
    zddot_d = 0.0

    return (np.array([x_d, y_d, z_d]),
            np.array([xdot_d, ydot_d, zdot_d]),
            np.array([xddot_d, yddot_d, zddot_d]))


# --- quick test: print the target at a few times, and plot the circle ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # sanity check: print target at t=0, t=quarter, t=half period
    for t in [0.0, np.pi/(2*OMEGA), np.pi/OMEGA]:
        pos, vel, acc = circle_reference(t)
        print(f"t={t:.2f}:  pos={np.round(pos,2)}  vel={np.round(vel,2)}")

    # plot the full circle the target traces
    ts = np.linspace(0, 2*np.pi/OMEGA, 200)
    xs = [circle_reference(t)[0][0] for t in ts]
    ys = [circle_reference(t)[0][1] for t in ts]
    plt.figure(figsize=(6,6))
    plt.plot(xs, ys, 'b-')
    plt.plot(xs[0], ys[0], 'go', markersize=12, label='start')
    plt.title("Circle reference trajectory (top view)")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.axis('equal'); plt.grid(alpha=0.3); plt.legend()
    plt.savefig("docs/circle_reference.png", dpi=120, bbox_inches="tight")
    print("Saved -> docs/circle_reference.png") 