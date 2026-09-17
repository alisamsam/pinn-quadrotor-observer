import sys, os, numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'scenarios/spiral')
from core.generate_quadrotor_data import quadrotor_dynamics
from core.controller_v2 import full_control_v2
from spiral_reference import spiral_reference, yaw_reference

x0 = np.zeros(12); x0[:3] = [4.0, 5.0, 0.0]
t_eval = np.arange(0.0, 30.0, 0.01)
ctrl = lambda t, x: full_control_v2(t, x, spiral_reference, yaw_reference)
sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, ctrl(t, x)),
                (0.0, 30.0), x0, t_eval=t_eval, max_step=0.02)
t, Y = sol.t, sol.y
pos_d = np.array([spiral_reference(tt)[0] for tt in t])
arr = np.column_stack([t, pos_d[:,0], pos_d[:,1], pos_d[:,2], Y[0], Y[1], Y[2]])
np.savetxt('paper_figures/fig_spiral_controller_tracking.csv', arr, delimiter=',',
           header='t,x_des,y_des,z_des,x_act,y_act,z_act', comments='', fmt='%.6g')
print('wrote paper_figures/fig_spiral_controller_tracking.csv', arr.shape)