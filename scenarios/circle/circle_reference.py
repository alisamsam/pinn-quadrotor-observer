"""Singha circular trajectory: x=10sin t, y=10cos t, z=2, yaw=0.2 sin t."""
import numpy as np

R = 10.0          # radius (Singha)
OMEGA = 0.3       # angular rate (lowered so ω²R = 0.9 m/s² << g)
Z_LEVEL = 2.0


def circular_reference(t):
    w = OMEGA
    # position
    x_d = R * np.sin(w*t)
    y_d = R * np.cos(w*t)
    z_d = Z_LEVEL
    # velocity (×w)
    xdot_d =  R * w * np.cos(w*t)
    ydot_d = -R * w * np.sin(w*t)
    zdot_d = 0.0
    # acceleration (×w²)
    xddot_d = -R * w**2 * np.sin(w*t)
    yddot_d = -R * w**2 * np.cos(w*t)
    zddot_d = 0.0
    return (np.array([x_d, y_d, z_d]),
            np.array([xdot_d, ydot_d, zdot_d]),
            np.array([xddot_d, yddot_d, zddot_d]))


def yaw_reference(t):
    w = OMEGA
    psi_d = 0.2 * np.sin(w*t)
    psi_d_dot = 0.2 * w * np.cos(w*t)
    return psi_d, psi_d_dot


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    # sanity check: print target at a few times
    for t in [0.0, np.pi/2, np.pi, 3*np.pi/2]:
        pos, vel, acc = circular_reference(t)
        psi, _ = yaw_reference(t)
        print(f"t={t:.2f}:  pos={np.round(pos,2)}  vel={np.round(vel,2)}  yaw={psi:.3f}")

    # plot the circle in 3D (so it looks 3D, as you wanted)
    ts = np.linspace(0, 2*np.pi/OMEGA, 400)
    P = np.array([circular_reference(t)[0] for t in ts])
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(P[:,0], P[:,1], P[:,2], 'b-', lw=2, label="desired circle")
    ax.scatter(0, 0, 4, color='black', s=100, marker='o', label="UAV start (0,0,4)")
    ax.scatter(P[0,0], P[0,1], P[0,2], color='green', s=120, marker='*', label="circle start")
 
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title("Singha circular trajectory (3D view)")
    ax.legend()

    # --- axis tick spacing ---
    import numpy as _np
    ax.set_xticks(_np.arange(-10, 11, 5))     # x every 5
    ax.set_yticks(_np.arange(-10, 11, 5))     # y every 5
    ax.set_zticks(_np.arange(0, 5, 1))        # z every 1

    # --- viewing angle: lower + more front-on (like the paper) ---
    ax.view_init(elev=12, azim=-72)           # lower elevation = more side/front view

    plt.savefig("docs/circle_reference_3d.png", dpi=120, bbox_inches="tight")
    print("Saved -> docs/circle_reference_3d.png")