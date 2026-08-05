import numpy as np

base = np.load('datasets/spiral_dataset.npz')
X_true = base['X'][40:50]

nom = np.load('datasets/mismatch/mass_0.npz')
X_nom = nom['X']

diff = np.abs(X_true - X_nom)
print('max abs difference: ', diff.max())
print('mean abs difference:', diff.mean())
names = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
for i, nm in enumerate(names):
    print('  %6s: %.2e' % (nm, diff[:,:,i].max()))