import sys
sys.path.append('..')

from chap2.mav_viewer import mav_viewer
from message_types.msg_state import msg_state

mv = mav_viewer()
state = msg_state()

for i in range(500):
    state.pn = state.pn + 0.1
    mv.update(state)

for i in range(500):
    state.pe = state.pe + 0.1
    mv.update(state)

for i in range(500):
    state.h = state.h + 0.1
    mv.update(state)

for i in range(1000):
    state.psi = state.psi + 3.14159*2 / 1000.0
    mv.update(state)

for i in range(1000):
    state.theta = state.theta + 3.14159*2 / 1000.0
    mv.update(state)

for i in range(1000):
    state.phi = state.phi + 3.14159*2 / 1000.0
    mv.update(state)
