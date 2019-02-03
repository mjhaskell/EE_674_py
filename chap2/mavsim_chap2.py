import sys
sys.path.append('..')

from chap2.mav_viewer import mav_viewer
import params.sim_params as SIM
from message_types.msg_state import msg_state

state = msg_state()  # instantiate state message
mav_view = mav_viewer()
sim_time = SIM.start_time

# main simulation loop
T = 2.5
while sim_time < SIM.end_time:
    #-------vary states to check viewer-------------
    if sim_time < SIM.end_time/6:
        state.pn += 10*SIM.ts_simulation
    elif sim_time < 2*SIM.end_time/6:
        state.pe += 10*SIM.ts_simulation
    elif sim_time < 3*SIM.end_time/6:
        state.h += 10*SIM.ts_simulation
    elif sim_time < 4*SIM.end_time/6:
        state.phi += 0.1*SIM.ts_simulation
    elif sim_time < 5*SIM.end_time/6:
        state.theta += 0.1*SIM.ts_simulation
    else:
        state.psi += 0.1*SIM.ts_simulation

    #-------update viewer and video-------------
    mav_view.update(state)
#    if VIDEO == True: video.update(sim_time)

    #-------increment time-------------
    sim_time += SIM.ts_simulation

print('Simulation has ended')
