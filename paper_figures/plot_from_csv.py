import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

D = 'paper_figures/'
o = np.genfromtxt(D + 'fig_spiral_observer_states.csv', delimiter=',', names=True)
c = np.genfromtxt(D + 'fig_spiral_controller_tracking.csv', delimiter=',', names=True)

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def save(fig, name):
    fig.tight_layout()
    fig.savefig(D + name + '.pdf')            # vector, HD -> use in LaTeX
    fig.savefig(D + name + '.png', dpi=300)   # raster preview
    plt.close(fig)
    print('saved', name)

# 1) Observer 3D trajectory: true vs estimated position
perr = np.sqrt(np.mean((o['x_true']-o['x_est'])**2 + (o['y_true']-o['y_est'])**2
                       + (o['z_true']-o['z_est'])**2))
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot(o['x_true'], o['y_true'], o['z_true'], color='red',   lw=1.5, label='true')
ax.plot(o['x_est'],  o['y_est'],  o['z_est'], '--', color='green', lw=1.5, label='estimated')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
ax.set_title(f'PINN observer - 3D trajectory   position RMSE={perr:.3f} m')
ax.legend()
save(fig, 'spiral_observer_3d')

# 2) Observer 12 states: true vs estimated (paper style, image16)
names = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
units = ['m','m','m','m/s','m/s','m/s','rad','rad','rad','rad/s','rad/s','rad/s']
meas  = {'x','y','z','phi','theta','psi'}          # positions + angles = measured
fig, axes = plt.subplots(4, 3, figsize=(15, 10))
fig.suptitle('PINN observer - 12 states on an unseen test flight: true vs estimated',
             fontsize=14)
for i, n in enumerate(names):
    a = axes.flat[i]
    a.plot(o['t'], o[n+'_true'], color='red',   lw=1.5, label='true')
    a.plot(o['t'], o[n+'_est'], '--', color='green', lw=1.5, label='estimated')
    e = rmse(o[n+'_true'], o[n+'_est'])
    tag = 'meas' if n in meas else 'HID'
    a.set_title(f'{n} ({tag})  RMSE={e:.3f} {units[i]}')
    a.set_ylabel(f'{n} ({units[i]})')
    a.set_xlabel('time t  (s)')
    a.grid(alpha=0.3); a.legend(fontsize=8)
save(fig, 'spiral_observer_12states')

# 3) Controller tracking 3D: desired (blue) vs actual (red)
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot(c['x_des'], c['y_des'], c['z_des'], color='blue', lw=2.0, label='desired')
ax.plot(c['x_act'], c['y_act'], c['z_act'], '--', color='red', lw=1.8, label='actual')
ax.scatter(c['x_act'][0], c['y_act'][0], c['z_act'][0], color='black', s=60,
           label='start (4,5,0)')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
ax.set_title('Spiral tracking: desired vs actual')
ax.legend()
save(fig, 'spiral_tracking_3d')