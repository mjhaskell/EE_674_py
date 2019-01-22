from mav_viewer import mav_viewer
from msg_state import msg_state

mv = mav_viewer()
state = msg_state()

for i in range(1000):
    state.psi = state.psi + 0.01
    mv.update(state)
