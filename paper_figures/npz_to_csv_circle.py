import numpy as np

d = np.load('docs/eval_circle_best_data.npz')
names = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
meas_idx = list(d['meas_idx'])              # indices of the 6 measured states
meas_names = [names[i] for i in meas_idx]

cols, hdr = [d['t']], ['t']
for i, n in enumerate(names):
    cols += [d['true'][:, i], d['est'][:, i]]
    hdr  += [n + '_true', n + '_est']

# append noisy measurements (6 cols) so the 12-state plot can show grey dots
for j, n in enumerate(meas_names):
    cols += [d['ynoisy'][:, j]]
    hdr  += [n + '_noisy']

arr = np.column_stack(cols)
np.savetxt('paper_figures/fig_circle_observer_states.csv', arr,
           delimiter=',', header=','.join(hdr), comments='', fmt='%.6g')
print('wrote paper_figures/fig_circle_observer_states.csv', arr.shape)
print('measured states:', meas_names)
