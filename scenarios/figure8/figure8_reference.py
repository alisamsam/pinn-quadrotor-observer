"""Figure-8 (Tilted Eight) reference: x=10sin t, y=10cos t sin t, z=1.5+cos t."""
import numpy as np

PSI_CONST = 0.7853    # constant yaw (Table 5)
OMEGA = 0.3         # slowdown factor (peak acccel 1.9 m/s², feasible; consistent with circle)

def figure8_reference(t):
    w=OMEGA
    # position
    x_d = 10*np.sin(w*t)
    y_d = 10*np.cos(w*t)*np.sin(w*t)        # = 5 sin(2wt)
    z_d = 1.5 + np.cos(w*t)

    # velocity (1st derivative)     
    xdot_d = 10*w*np.cos(w*t)
    ydot_d = 10*w*np.cos(2*w*t)   # = 10*w*(cos^2(w*t) - sin^2(w*t)) = 10*w cos(2w*t)
    zdot_d = -w*np.sin(w*t)

    # acceleration (2nd derivative) 
    xddot_d = -10*w**2*np.sin(w*t)
    yddot_d = -20*w**2*np.sin(2*w*t)   # = -20*w^2 sin(w*t) cos(w*t)
    zddot_d = -w**2*np.cos(w*t)

    return (np.array([x_d, y_d, z_d]),
            np.array([xdot_d, ydot_d, zdot_d]),
            np.array([xddot_d, yddot_d, zddot_d]))


def yaw_reference(t):
    return PSI_CONST, 0.0      # constant yaw, zero rate

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    # print position at a few times
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t= t * 2*np.pi/OMEGA   #Sample  at 0%, 25%, 50%, 75%, 100% of the period
        pos, vel, acc = figure8_reference(t)
        print(f"t={t:.2f}: pos={np.round(pos,2)}")

    # sample one full period and plot
    ts = np.linspace(0, 2*np.pi/OMEGA, 400)
    P = np.array([figure8_reference(t)[0] for t in ts])

    fig = plt.figure(figsize=(13, 5))
    # left: top view (x-y) -> should show the "8"
    ax1 = fig.add_subplot(121)
    ax1.plot(P[:,0], P[:,1], 'b-')
    ax1.set_title("Top view (x-y) — should be a figure-8")
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)"); ax1.grid(alpha=0.3); ax1.axis('equal')
    # right: 3D (shows the z bob too)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(P[:,0], P[:,1], P[:,2], 'b-')
    ax2.set_title("3D figure-8 (tilted)")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    ax2.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.savefig("docs/figure8_reference.png", dpi=120, bbox_inches="tight")
    print("Saved -> docs/figure8_reference.png")