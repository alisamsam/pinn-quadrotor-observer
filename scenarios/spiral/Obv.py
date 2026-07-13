import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ============================================================
# PARAMÈTRES DU QUADROTOR
# ============================================================

m = 1.80
g = 9.81
l = 0.20

Ix = 0.03
Iy = 0.03
Iz = 0.04


# ============================================================
# GAINS DE COMMANDE
# ============================================================

Kpx, Kdx = 3.0, 5.0
Kpy, Kdy = 3.0, 5.0
Kpz, Kdz = 6.0, 6.0

Kp_phi, Kd_phi = 25.0, 8.0
Kp_theta, Kd_theta = 25.0, 8.0
Kp_psi, Kd_psi = 8.0, 4.0


# ============================================================
# SATURATION
# ============================================================

def sat(x, xmin, xmax):
    return np.minimum(np.maximum(x, xmin), xmax)


# ============================================================
# TRAJECTOIRE DÉSIRÉE : HÉLICE VERTICALE
# ============================================================

def desired_trajectory(t):

    R = 8.0
    w = 0.35
    vz = 1.2

    xd = R * np.cos(w * t)
    yd = R * np.sin(w * t)
    zd = vz * t

    xd_dot = -R * w * np.sin(w * t)
    yd_dot = R * w * np.cos(w * t)
    zd_dot = vz

    xd_ddot = -R * w**2 * np.cos(w * t)
    yd_ddot = -R * w**2 * np.sin(w * t)
    zd_ddot = 0.0

    psi_d = w * t
    psi_d_dot = w
    psi_d_ddot = 0.0

    return (
        xd, yd, zd,
        xd_dot, yd_dot, zd_dot,
        xd_ddot, yd_ddot, zd_ddot,
        psi_d, psi_d_dot, psi_d_ddot
    )


# ============================================================
# CONTRÔLEUR
# ============================================================

def controller(t, X):

    x, y, z = X[0], X[1], X[2]
    vx, vy, vz = X[3], X[4], X[5]
    phi, theta, psi = X[6], X[7], X[8]
    p, q, r = X[9], X[10], X[11]

    (
        xd, yd, zd,
        xd_dot, yd_dot, zd_dot,
        xd_ddot, yd_ddot, zd_ddot,
        psi_d, psi_d_dot, psi_d_ddot
    ) = desired_trajectory(t)

    ex = x - xd
    ey = y - yd
    ez = z - zd

    evx = vx - xd_dot
    evy = vy - yd_dot
    evz = vz - zd_dot

    ax_cmd = xd_ddot - Kpx * ex - Kdx * evx
    ay_cmd = yd_ddot - Kpy * ey - Kdy * evy
    az_cmd = zd_ddot - Kpz * ez - Kdz * evz

    ax_cmd = sat(ax_cmd, -12.0, 12.0)
    ay_cmd = sat(ay_cmd, -12.0, 12.0)
    az_cmd = sat(az_cmd, -10.0, 10.0)

    den = np.cos(phi) * np.cos(theta)
    den = np.sign(den) * max(abs(den), 0.25)

    U1 = m * (g + az_cmd) / den
    U1 = sat(U1, 1.0, 45.0)

    phi_d = (1.0 / g) * (
        ax_cmd * np.sin(psi) - ay_cmd * np.cos(psi)
    )

    theta_d = (1.0 / g) * (
        ax_cmd * np.cos(psi) + ay_cmd * np.sin(psi)
    )

    phi_d = sat(phi_d, -0.65, 0.65)
    theta_d = sat(theta_d, -0.65, 0.65)

    U2 = Ix / l * (
        -Kp_phi * (phi - phi_d)
        -Kd_phi * p
        -q * r * ((Iy - Iz) / Ix)
    )

    U3 = Iy / l * (
        -Kp_theta * (theta - theta_d)
        -Kd_theta * q
        -p * r * ((Iz - Ix) / Iy)
    )

    U4 = Iz * (
        -Kp_psi * (psi - psi_d)
        -Kd_psi * (r - psi_d_dot)
        -p * q * ((Ix - Iy) / Iz)
    )

    U2 = sat(U2, -5.0, 5.0)
    U3 = sat(U3, -5.0, 5.0)
    U4 = sat(U4, -5.0, 5.0)

    return U1, U2, U3, U4, phi_d, theta_d, psi_d


# ============================================================
# MODÈLE DYNAMIQUE
# ============================================================

def quadrotor_dynamics(t, X):

    x, y, z = X[0], X[1], X[2]
    vx, vy, vz = X[3], X[4], X[5]
    phi, theta, psi = X[6], X[7], X[8]
    p, q, r = X[9], X[10], X[11]

    U1, U2, U3, U4, _, _, _ = controller(t, X)

    Ux = (
        np.cos(psi) * np.sin(theta) * np.cos(phi)
        + np.sin(psi) * np.sin(phi)
    )

    Uy = (
        np.sin(psi) * np.sin(theta) * np.cos(phi)
        - np.cos(psi) * np.sin(phi)
    )

    x_ddot = Ux * U1 / m
    y_ddot = Uy * U1 / m
    z_ddot = np.cos(phi) * np.cos(theta) * U1 / m - g

    phi_ddot = q * r * ((Iy - Iz) / Ix) + l * U2 / Ix
    theta_ddot = p * r * ((Iz - Ix) / Iy) + l * U3 / Iy
    psi_ddot = p * q * ((Ix - Iy) / Iz) + U4 / Iz

    return np.array([
        vx, vy, vz,
        x_ddot, y_ddot, z_ddot,
        p, q, r,
        phi_ddot, theta_ddot, psi_ddot
    ])


# ============================================================
# SIMULATION
# ============================================================

t0 = 0.0
tf = 65.0
t_eval = np.linspace(t0, tf, 6500)

R = 8.0
w = 0.35
vz0 = 1.2

X0 = np.array([
    R, 0.0, 0.0,
    0.0, R * w, vz0,
    0.0, 0.0, 0.0,
    0.0, 0.0, w
])

sol = solve_ivp(
    quadrotor_dynamics,
    [t0, tf],
    X0,
    t_eval=t_eval,
    method="RK45",
    rtol=1e-6,
    atol=1e-8
)

t = sol.t
X = sol.y.T

x = X[:, 0]
y = X[:, 1]
z = X[:, 2]

phi = X[:, 6]
theta = X[:, 7]
psi = X[:, 8]


# ============================================================
# RÉFÉRENCES ET COMMANDES
# ============================================================

xd, yd, zd = [], [], []
U1, U2, U3, U4 = [], [], [], []
phi_d, theta_d, psi_d = [], [], []

for i, ti in enumerate(t):

    ref = desired_trajectory(ti)

    xd.append(ref[0])
    yd.append(ref[1])
    zd.append(ref[2])

    u = controller(ti, X[i, :])

    U1.append(u[0])
    U2.append(u[1])
    U3.append(u[2])
    U4.append(u[3])

    phi_d.append(u[4])
    theta_d.append(u[5])
    psi_d.append(u[6])

xd = np.array(xd)
yd = np.array(yd)
zd = np.array(zd)

U1 = np.array(U1)
U2 = np.array(U2)
U3 = np.array(U3)
U4 = np.array(U4)

phi_d = np.array(phi_d)
theta_d = np.array(theta_d)
psi_d = np.array(psi_d)


# ============================================================
# FIGURE 1 : TRAJECTOIRE 3D
# ============================================================

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot(
    xd, yd, zd,
    color="blue",
    linestyle="-",
    linewidth=2.5,
    label="Desired trajectory"
)

ax.plot(
    x, y, z,
    color="red",
    linestyle="--",
    linewidth=2.2,
    label="Actual trajectory"
)

ax.set_xlabel("X(m)")
ax.set_ylabel("Y(m)")
ax.set_zlabel("Z(m)")
ax.set_title("Helical trajectory tracking")

ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([0, 80])

ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# FIGURE 2 : POSITIONS
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(t, xd, color="blue", linestyle="-", label="xd")
plt.plot(t, x, color="red", linestyle="--", label="x")

plt.plot(t, yd, color="green", linestyle="-", label="yd")
plt.plot(t, y, color="orange", linestyle="--", label="y")

plt.plot(t, zd, color="black", linestyle="-", label="zd")
plt.plot(t, z, color="magenta", linestyle="--", label="z")

plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Positions tracking")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# FIGURE 3 : ERREURS
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(t, x - xd, color="red", label="Error x")
plt.plot(t, y - yd, color="blue", label="Error y")
plt.plot(t, z - zd, color="green", label="Error z")

plt.xlabel("Time (s)")
plt.ylabel("Error (m)")
plt.title("Tracking errors")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# FIGURE 4 : ANGLES
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(t, phi_d, color="blue", linestyle="-", label="phi desired")
plt.plot(t, phi, color="red", linestyle="--", label="phi actual")

plt.plot(t, theta_d, color="green", linestyle="-", label="theta desired")
plt.plot(t, theta, color="orange", linestyle="--", label="theta actual")

plt.plot(t, psi_d, color="black", linestyle="-", label="psi desired")
plt.plot(t, psi, color="magenta", linestyle="--", label="psi actual")

plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title("Attitude tracking")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# FIGURE 5 : COMMANDES
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(t, U1, color="blue", label="U1")
plt.plot(t, U2, color="red", label="U2")
plt.plot(t, U3, color="green", label="U3")
plt.plot(t, U4, color="orange", label="U4")

plt.xlabel("Time (s)")
plt.ylabel("Control inputs")
plt.title("Control inputs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# PERFORMANCE
# ============================================================

rmse_x = np.sqrt(np.mean((x - xd) ** 2))
rmse_y = np.sqrt(np.mean((y - yd) ** 2))
rmse_z = np.sqrt(np.mean((z - zd) ** 2))

print("===== Performance =====")
print(f"RMSE x = {rmse_x:.4f} m")
print(f"RMSE y = {rmse_y:.4f} m")
print(f"RMSE z = {rmse_z:.4f} m")