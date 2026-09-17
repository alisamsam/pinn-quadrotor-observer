import numpy as np

d = np.load('docs/eval_spiral_best_data.npz')
names = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']

cols, hdr = [d['t']], ['t']
for i, n in enumerate(names):
    cols += [d['true'][:, i], d['est'][:, i]]
    hdr  += [n + '_true', n + '_est']

arr = np.column_stack(cols)                      # t + 24 columns, side by side
np.savetxt('docs/fig_spiral_observer_states.csv', arr,
           delimiter=',', header=','.join(hdr), comments='', fmt='%.6g')
print('wrote docs/fig_spiral_observer_states.csv', arr.shape)