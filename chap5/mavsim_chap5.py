
"""
mavSimPy 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        1/1/2019 - RWB
        1/29/2019 - RWB
        2/2/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import params.sim_params as SIM

from chap2.mav_viewer import mav_viewer
from chap3.data_viewer import data_viewer
from chap4.mav_dynamics import mav_dynamics
from chap4.wind_simulation import wind_simulation
from chap5.trim import compute_trim
from chap5.compute_models import compute_tf_model#, compute_ss_model

mav_view = mav_viewer()  # initialize the mav viewer
data_view = data_viewer()  # initialize view of data plots

# initialize elements of the architecture
wind = wind_simulation(SIM.ts_simulation)
mav = mav_dynamics(SIM.ts_simulation)

# use compute_trim function to compute trim state and trim input
Va = 25.
gamma = 0.*np.pi/180.
trim_state, trim_input = compute_trim(mav, Va, gamma)
mav._state = trim_state  # set the initial state of the mav to the trim state
delta = trim_input  # set input to constant constant trim input

# # compute the state space model linearized about trim
#A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, \
T_h_Va, T_Va_delta_t, T_Va_theta, T_beta_delta_r \
    = compute_tf_model(mav, trim_state, trim_input)
print('T_phi_delta_a: \n',T_phi_delta_a)
print('T_chi_phi: \n',T_chi_phi)
print('T_theta_delta_e: \n',T_theta_delta_e)
print('T_h_theta: \n',T_h_theta)
print('T_h_Va: \n',T_h_Va)
print('T_Va_delta_t: \n',T_Va_delta_t)
print('T_Va_theta: \n',T_Va_theta)
print('T_beta_delta_r: \n',T_beta_delta_r)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:

    #-------physical system-------------
    #current_wind = wind.update()  # get the new wind vector
    current_wind = np.zeros((6,1))
    mav.update_state(delta, current_wind)  # propagate the MAV dynamics

    #-------update viewer-------------
    mav_view.update(mav.msg_true_state)  # plot body of MAV
    data_view.update(mav.msg_true_state, # true states
                     mav.msg_true_state, # estimated states
                     mav.msg_true_state, # commanded states
                     SIM.ts_simulation)

    #-------increment time-------------
    sim_time += SIM.ts_simulation

input('Press ENTER to close...')
