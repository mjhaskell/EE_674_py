import numpy as np
import sys
sys.path.append('..')
from chap11.dubins_params import dubins_params
from message_types.msg_path import msg_path

class path_manager:
    def __init__(self):
        # message sent to path follower
        self.path = msg_path()
        # pointers to previous, current, and next waypoints
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2
        self.ptrs_updated = True
        # flag that request new waypoints from path planner
        self.flag_need_new_waypoints = True
        self.num_waypoints = 0
        self.halfspace_n = np.inf * np.ones((3,1))
        self.halfspace_r = np.inf * np.ones((3,1))
        # state of the manager state machine
        self.manager_state = 1
        # dubins path parameters
        self.dubins_path = dubins_params()
        self.dubins_state_changed = True

    def update(self, waypoints, radius, state):
        if waypoints.flag_waypoints_changed:
            waypoints.flag_waypoints_changed = False
            self.num_waypoints = waypoints.num_waypoints
            self.initialize_pointers()
            self.manager_state = 1
            self.flag_need_new_waypoints = False

        if self.path.flag_path_changed:
            self.path.flag_path_changed = False

        if waypoints.num_waypoints == 0:
            waypoints.flag_manager_requests_waypoints = True
        else:
            if waypoints.type == 'straight_line':
                self.line_manager(waypoints, state)
            elif waypoints.type == 'fillet':
                self.fillet_manager(waypoints, radius, state)
            elif waypoints.type == 'dubins':
                self.dubins_manager(waypoints, radius, state)
            else:
                print('Error in Path Manager: Undefined waypoint type.')
        return self.path

    def line_manager(self, waypoints, state):
        p = np.array([[state.pn, state.pe, -state.h]]).T

        w_im1 = waypoints.ned[:,self.ptr_previous].reshape(3,1)
        w_i = waypoints.ned[:,self.ptr_current].reshape(3,1)
        w_ip1 = waypoints.ned[:,self.ptr_next].reshape(3,1)
        
        q_im1 = w_i -w_im1
        q_im1 /= np.linalg.norm(q_im1)
        q_i = w_ip1 - w_i
        q_i /= np.linalg.norm(q_i)

        n_i = q_im1 + q_i
        n_i /= np.linalg.norm(n_i)

        self.halfspace_r = w_i
        self.halfspace_n = n_i

        self.path.airspeed = waypoints.airspeed.item(self.ptr_current)
        self.path.flag = 'line'

        if self.inHalfSpace(p):
            self.increment_pointers()
            self.path.flag_path_changed = True
            
            self.path.line_origin = w_i
            self.path.line_direction = q_i
        else:
            self.path.flag_path_changed = False

            self.path.line_origin = w_im1
            self.path.line_direction = q_im1


    def fillet_manager(self, waypoints, radius, state):
        p = np.array([[state.pn, state.pe, -state.h]]).T

        w_im1 = waypoints.ned[:,self.ptr_previous].reshape(3,1)
        w_i = waypoints.ned[:,self.ptr_current].reshape(3,1)
        w_ip1 = waypoints.ned[:,self.ptr_next].reshape(3,1)
        
        q_im1 = w_i -w_im1
        q_im1 /= np.linalg.norm(q_im1)
        q_i = w_ip1 - w_i
        q_i /= np.linalg.norm(q_i)

        var_phi = np.arccos(-q_im1.T @ q_i)

        if self.manager_state == 1:
            self.path.flag = 'line'
            self.path.line_origin = w_im1
            self.path.line_direction = q_im1
            self.path.airspeed = waypoints.airspeed.item(self.ptr_current)

            z = w_i - (radius/np.tan(var_phi/2.0))*q_im1
            self.halfspace_r = z
            self.halfspace_n = q_im1
            
            if self.inHalfSpace(p):
                self.manager_state = 2
                self.path.flag_path_changed = True
            else:
                self.path.flag_path_changed = False

        else:
            direction = (q_im1-q_i)
            direction /= np.linalg.norm(direction)
            c = w_i - (radius/np.sin(var_phi/2.0))*direction
            lam = np.sign(q_im1.item(0)*q_i.item(1)-q_im1.item(1)*q_i.item(0))

            self.path.flag = 'orbit'
            self.path.airspeed = waypoints.airspeed.item(self.ptr_current)
            self.path.orbit_center = c
            self.path.orbit_radius = radius
            if lam > 0:
                self.path.orbit_direction = 'CW'
            else:
                self.path.orbit_direction = 'CCW'

            z = w_i + (radius/np.tan(var_phi/2.0))*q_i
            self.halfspace_r = z
            self.halfspace_n = q_i

            if self.inHalfSpace(p):
                self.increment_pointers()
                self.manager_state = 1
                self.path.flag_path_changed = True
            else:
                self.path.flag_path_changed = False

    def dubins_manager(self, waypoints, radius, state):
        p = np.array([[state.pn, state.pe, -state.h]]).T
        self.path.airspeed = waypoints.airspeed.item(self.ptr_current)

        if self.ptrs_updated:
            self.ptrs_updated = False
            ps = waypoints.ned[:,self.ptr_previous].reshape(3,1)
            pe = waypoints.ned[:,self.ptr_current].reshape(3,1)
            chis = waypoints.course.item(self.ptr_previous)
            chie = waypoints.course.item(self.ptr_current)
            self.dubins_path.update(ps,chis,pe,chie,radius)

        if self.manager_state == 1:
            self.path.flag_path_changed = self.dubins_state_changed
            self.dubins_state_changed = False
            self.path.flag = 'orbit'
            self.path.orbit_center = self.dubins_path.center_s
            self.path.orbit_radius = radius
            if self.dubins_path.dir_s > 0:
                self.path.orbit_direction = 'CW'
            else:
                self.path.orbit_direction = 'CCW'

            self.halfspace_n = self.dubins_path.n1
            self.halfspace_r = self.dubins_path.r1
            if self.inHalfSpace(p):
                print('entered state 2')
                self.manager_state = 2
                self.dubins_state_changed = True
        elif self.manager_state == 2:
            self.path.flag_path_changed = self.dubins_state_changed
            self.dubins_state_changed = False
            self.path.flag = 'line'
            self.path.line_origin = self.dubins_path.r1
            self.path.line_direction = self.dubins_path.n1

            self.halfspace_n = self.dubins_path.n1
            self.halfspace_r = self.dubins_path.r1
            if self.inHalfSpace(p):
                print('entered state 3')
                self.manager_state = 3
                self.dubins_state_changed = True
        elif self.manager_state == 3:
            self.path.flag_path_changed = self.dubins_state_changed
            self.dubins_state_changed = False
            self.path.flag = 'line'
            self.path.line_origin = self.dubins_path.r1
            self.path.line_direction = self.dubins_path.n1

            self.halfspace_n = self.dubins_path.n1
            self.halfspace_r = self.dubins_path.r2
            if self.inHalfSpace(p):
                print('entered state 4')
                self.manager_state = 4
                self.dubins_state_changed = True
        elif self.manager_state == 4:
            self.path.flag_path_changed = self.dubins_state_changed
            self.dubins_state_changed = False
            self.path.flag = 'orbit'
            self.path.orbit_center = self.dubins_path.center_e
            if self.dubins_path.dir_e > 0:
                self.path.orbit_direction = 'CW'
            else:
                self.path.orbit_direction = 'CCW'

            self.halfspace_n = self.dubins_path.n3
            self.halfspace_r = self.dubins_path.r3
            if self.inHalfSpace(p):
                print('entered state 5')
                self.manager_state = 5
                self.path.dubins_state_changed = True
        else:
            self.path.flag_path_changed = self.dubins_state_changed
            self.dubins_state_changed = False
            self.halfspace_n = self.dubins_path.n3
            self.halfspace_r = self.dubins_path.r3
            if self.inHalfSpace(p):
                self.manager_state = 1
                self.path.dubins_state_changed = True
                self.increment_pointers()

    def initialize_pointers(self):
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2
        self.ptrs_updated = True

    def increment_pointers(self):
            self.ptr_previous += 1
            if self.ptr_previous >= self.num_waypoints:
                self.ptr_previous = 0
            self.ptr_current += 1
            if self.ptr_current >= self.num_waypoints:
                self.ptr_current = 0
            self.ptr_next += 1
            if self.ptr_next >= self.num_waypoints:
                self.ptr_next = 0
            self.ptrs_updated = True

    def inHalfSpace(self, pos):
        if (pos-self.halfspace_r).T @ self.halfspace_n >= 0:
            return True
        else:
            return False
