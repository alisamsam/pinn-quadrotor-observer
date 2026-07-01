"""Singha spiral trajectory: x=10sin t, y=10cos t, z=2t, yaw=0.7853 (constant)."""
import numpy as np

R = 10.0          # radius (Singha)
OMEGA = 0.6      # angular rate (raw Table 5 form; ω²R = 10 m/s² ≈ g, aggressive)
CLIMB = 2.0       # vertical climb rate : z= CLIMB * t ((ż = 2 m/s constant))
PSI_D = 0.7853      # constant desired yaw (π/4)


def spiral_reference(t):
    w = OMEGA
    # position
    x_d = R * np.sin(w*t)
    y_d = R * np.cos(w*t)
    z_d = CLIMB * t
    # velocity (×w)
    xdot_d =  R * w * np.cos(w*t)
    ydot_d = -R * w * np.sin(w*t)
    zdot_d = CLIMB
    # acceleration (×w²)
    xddot_d = -R * w**2 * np.sin(w*t)
    yddot_d = -R * w**2 * np.cos(w*t)
    zddot_d = 0.0
    return (np.array([x_d, y_d, z_d]),
            np.array([xdot_d, ydot_d, zdot_d]),
            np.array([xddot_d, yddot_d, zddot_d]))


def yaw_reference(t):
    psi_d =PSI_D
    psi_d_dot = 0.0
    return psi_d, psi_d_dot


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    # sanity check: print target at a few times
    for t in [0.0, np.pi/2, np.pi, 3*np.pi/2]:
        pos, vel, acc = spiral_reference(t)
        psi, _ = yaw_reference(t)
        print(f"t={t:.2f}:  pos={np.round(pos,2)}  vel={np.round(vel,2)}  yaw={psi:.3f}")

    # plot the spiral in 3D (so it looks 3D, as you wanted)
    T = 30.0                                   # CHANGED: 30 s flight (~4.8 loops)
    ts = np.linspace(0, T, 800)
    P = np.array([spiral_reference(t)[0] for t in ts])
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(P[:,0], P[:,1], P[:,2], 'b-', lw=2, label="desired spiral")
    ax.scatter(4, 5, 0, color='black', s=100, marker='o', label="UAV start (4,5,0)")
    ax.scatter(P[0,0], P[0,1], P[0,2], color='green', s=120, marker='*', label="spiral start")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title("Singha spiral trajectory (3D view)")
    ax.legend()
    ax.set_xticks(np.arange(-10, 11, 5))       # x every 5
    ax.set_yticks(np.arange(-10, 11, 5))       # y every 5
    ax.set_zticks(np.arange(0, 61, 10))        # CHANGED: z 0→60, every 10
    ax.view_init(elev=12, azim=-72)
    plt.savefig("docs/spiral_reference_3d.png", dpi=120, bbox_inches="tight")