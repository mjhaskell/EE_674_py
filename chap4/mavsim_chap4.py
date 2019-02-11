import sys
sys.path.append('..')

from chap2.mav_viewer import mav_viewer
from chap3.data_viewer import data_viewer
from chap4.mav_dynamics import mav_dynamics
from chap4.wind_simulation import wind_simulation
from message_types.msg_state import msg_state
import params.sim_params as SIM
import numpy as np

mav_view = mav_viewer()
data_view = data_viewer()

wind = wind_simulation(SIM.ts_simulation)
mav = mav_dynamics(SIM.ts_simulation)

sim_time = SIM.start_time

# main simulation loop
while sim_time < SIM.end_time:
    #-------set control surfaces-------------
    delta_e = -0.2
    delta_t = 0.5
    delta_a = 0.0
    delta_r = 0.0 #0.005

    if sim_time < 15:
        delta_r = 0.002
    elif sim_time < 25:
        delta_a = 0.025
    elif sim_time < 29.5:
        delta_a = -.025
    else:
        delta_t = 1.0

    delta = np.array([delta_e,delta_t,delta_a,delta_r]).T

    #-------physical system------------
    current_wind = wind.update(mav._Va)
    mav.update_state(delta, current_wind)

    #-------update viewer--------------
    mav_view.update(mav.msg_true_state)
    data_view.update(mav.msg_true_state, # true states
                     mav.msg_true_state, # estimated states
                     mav.msg_true_state, # commanded states
                     SIM.ts_simulation)

    #-------increment time-------------
    sim_time += SIM.ts_simulation

print('Simulation has ended')
