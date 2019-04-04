import numpy as np
from math import sin, cos, atan, atan2
import sys

sys.path.append('..')
from message_types.msg_autopilot import msg_autopilot

class path_follower:
    def __init__(self):
        self.chi_inf = np.deg2rad(80)  # approach angle for large distance from straight-line path
        self.k_path = 0.01  # proportional gain for straight-line path following
        self.k_orbit = 2.5  # proportional gain for orbit following
        self.gravity = 9.8
        self.autopilot_commands = msg_autopilot()  # message sent to autopilot

    def update(self, path, state):
        if path.flag=='line':
            self._follow_straight_line(path, state)
        elif path.flag=='orbit':
            self._follow_orbit(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self, path, state):
        Vg = state.Vg
        chi = state.chi
        qn = path.line_direction.item(0)
        qe = path.line_direction.item(1)
        qd = path.line_direction.item(2)

        chi_q = atan2(qe,qn)
        chi_q = self._wrap(chi_q, chi)

        Ri_p = np.array([[cos(chi_q),sin(chi_q),0],
                        [-sin(chi_q),cos(chi_q),0],
                         [0,0,1]])

        p_i = np.array([[state.pn,state.pe,-state.h]]).T
        r_i = path.line_origin

        # chi_c
        ep_i = p_i-r_i
        ep = Ri_p @ ep_i
        epy = ep.item(1) 

        chi_d = -self.chi_inf*2.0/np.pi*atan(self.k_path*epy) 
        chi_c = chi_q + chi_d

        # h_c
        q = np.array([[qn,qe,qd]]).T
        n_num = np.cross(q.reshape(3), np.array([0,0,1])).reshape((3,1))
        n = n_num / np.linalg.norm(n_num)
        s = ep_i - (ep_i.T @ n)*n
        sn = s.item(0)
        se = s.item(1)
        sd = qd * np.sqrt((sn**2+se**2)/(qn**2+qe**2))
        h_c = -r_i.item(2) - sd

        self.autopilot_commands.airspeed_command = path.airspeed
        self.autopilot_commands.course_command = chi_c
        self.autopilot_commands.altitude_command = h_c
        self.autopilot_commands.phi_feedforward = 0.0

    def _follow_orbit(self, path, state):
        Vg = state.Vg
        psi = state.psi
        chi = state.chi
        p_i = np.array([[state.pn, state.pe, -state.h]]).T
        d = p_i - path.orbit_center
        d_norm = np.linalg.norm(d)
        R = path.orbit_radius

        if path.orbit_direction == 'CW':
            direction = 1
        else:
            direction = -1

        # chi command
        var_phi = atan2(d.item(1), d.item(0))
        chi0 = var_phi + direction * np.pi/2.0
        chi_c = chi0 + direction*atan(self.k_orbit * (d_norm-R)/R)

        # phi feedforward
        phi_ff = direction*atan2(Vg**2 , (self.gravity*R*cos(chi-psi)))

        self.autopilot_commands.airspeed_command = path.airspeed
        self.autopilot_commands.course_command = chi_c
        self.autopilot_commands.altitude_command = -path.orbit_center.item(2)
        self.autopilot_commands.phi_feedforward = phi_ff

    def _wrap(self, chi_c, chi):
        while chi_c-chi > np.pi:
            chi_c = chi_c - 2.0 * np.pi
        while chi_c-chi < -np.pi:
            chi_c = chi_c + 2.0 * np.pi
        return chi_c
