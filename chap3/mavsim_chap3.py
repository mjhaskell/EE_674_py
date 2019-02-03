import sys
sys.path.append('..')

from chap3.mav_dynamics import mav_dynamics as Dynamics
from chap2.mav_viewer import mav_viewer
import params.sim_params as SIM
from message_types.msg_state import msg_state
import numpy as np

state = msg_state()  # instantiate state message
mav_view = mav_viewer()
sim_time = SIM.start_time
dyn = Dynamics(SIM.ts_simulation)

# main simulation loop
T = 2.5
while sim_time < SIM.end_time:
    #-------vary states to check viewer-------------
    fx = 0
    fy = 0
    fz = 0
    l = 0
    m = 0
    n = 0
    if sim_time < SIM.end_time/6:
        fx = 50
    elif sim_time < 2*SIM.end_time/6:
        fy = 50
        dyn._state[3] = 0
    elif sim_time < 3*SIM.end_time/6:
        fz = 50
        dyn._state[4] = 0
    elif sim_time < 4*SIM.end_time/6:
        l = 0.05
        dyn._state[5] = 0
    elif sim_time < 5*SIM.end_time/6:
        m = 0.05
        dyn._state[10] = 0
        dyn._state[12] = 0
        if sim_time <= 4*SIM.end_time/6 + 0.02:
            dyn._state[6:10] = np.array([1,0,0,0]).reshape((4,1))
    else:
        n = 0.05
        dyn._state[11] = 0
        if sim_time <= 5*SIM.end_time/6 + 0.02:
            dyn._state[6:10] = np.array([1,0,0,0]).reshape((4,1))

    U = np.array([fx,fy,fz,l,m,n])
    dyn.update_state(U)
    state = dyn.msg_true_state
    mav_view.update(state)

    #-------increment time-------------
    sim_time += SIM.ts_simulation

print('Simulation has ended')
