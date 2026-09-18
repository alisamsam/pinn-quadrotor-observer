import sys, os, numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'scenarios/circle')
sys.path.insert(0, 'scenarios/figure8')
from core.generate_quadrotor_data import quadrotor_dynamics
from core.controller_v2 import full_control_v2
from circle_reference import circular_reference, yaw_reference as circ_yaw
from figure8_reference import figure8_reference, yaw_reference as f8_yaw

T = 21.0                      # one full period (2*pi/0.3 = 20.94 s)
t_eval = np.arange(0.0, T, 0.01)

def run(ref_pos, ref_yaw, x0, out):
    ctrl = lambda t, x: full_control_v2(t, x, ref_pos, ref_yaw)
    sol = solve_ivp(lambda t, x: quadrotor_dynamics(t, x, ctrl(t, x)),
                    (0.0, T), x0, t_eval=t_eval, max_step=0.02)
    t, Y = sol.t, sol.y
    pos_d = np.array([ref_pos(tt)[0] for tt in t])
    arr = np.column_stack([t, pos_d[:,0], pos_d[:,1], pos_d[:,2], Y[0], Y[1], Y[2]])
    np.savetxt(out, arr, delimiter=',',
               header='t,x_des,y_des,z_des,x_act,y_act,z_act', comments='', fmt='%.6g')
    print('wrote', out, arr.shape)

# circle: starts at (0,10,2)
x0c = np.zeros(12); x0c[:3] = circular_reference(0.0)[0]
run(circular_reference, circ_yaw, x0c, 'paper_figures/fig_circle_controller_tracking.csv')

# figure-8: starts at (0,0,2.5)
x0f = np.zeros(12); x0f[:3] = figure8_reference(0.0)[0]
run(figure8_reference, f8_yaw, x0f, 'paper_figures/fig_figure8_controller_tracking.csv')
print('DONE')
