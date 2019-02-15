"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
"""
import sys
sys.path.append('..')
import numpy as np

# load message types
from message_types.msg_state import msg_state

import params.aerosonde_params as MAV
from tools.tools import Quaternion2Rotation, Quaternion2Euler
from math import exp

class mav_dynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.pn0],  # (0)
                               [MAV.pe0],   # (1)
                               [MAV.pd0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0]])   # (12)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        self._update_velocity_data()
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # initialize true_state message
        self.msg_true_state = msg_state()

    ###################################
    # public functions
    def update_state(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_e, delta_t, delta_a, delta_r) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step/2.*k1, forces_moments)
        k3 = self._derivatives(self._state + time_step/2.*k2, forces_moments)
        k4 = self._derivatives(self._state + time_step*k3, forces_moments)
        self._state += time_step/6.0 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)

        # update the message class for the true state
        self._update_msg_true_state()

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        pn = state.item(0)
        pe = state.item(1)
        pd = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        #   extract forces/moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        # position kinematics
        Rv_b = Quaternion2Rotation(state[6:10])
        
        pos_dot = Rv_b @ np.array([u,v,w]).T
        pn_dot = pos_dot.item(0)
        pe_dot = pos_dot.item(1)
        pd_dot = pos_dot.item(2)
         

        # position dynamics
        u_dot = r*v - q*w + 1/MAV.mass * fx
        v_dot = p*w - r*u + 1/MAV.mass * fy
        w_dot = q*u - p*v + 1/MAV.mass * fz

        # rotational kinematics
        e0_dot = (-p * e1 - q * e2 - r * e3) * 0.5
        e1_dot = (p * e0 + r * e2 - q * e3) * 0.5
        e2_dot = (q * e0 - r * e1 + p * e3) * 0.5
        e3_dot = (r * e0 + q * e1 - p * e2) * 0.5

        # rotatonal dynamics
        p_dot = MAV.gamma1 * p * q - MAV.gamma2 * q * r + MAV.gamma3 * l + MAV.gamma4 * n
        q_dot = MAV.gamma5 * p * r - MAV.gamma6 * (p**2 - r**2) + 1/MAV.Jy * m
        r_dot = MAV.gamma7 * p * q - MAV.gamma1 * q * r + MAV.gamma4 * l + MAV.gamma8 * n

        # collect the derivative of the states
        x_dot = np.array([[pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot

    def _update_velocity_data(self, wind=np.zeros((6,1))):
        Rb_v = Quaternion2Rotation(self._state[6:10])

        self._wind = Rb_v.T @ wind[:3] + wind[3:]
        V = self._state[3:6]
        Vr = V - self._wind

        # compute airspeed
        self._Va = np.linalg.norm(Vr)

        ur = Vr.item(0)
        vr = Vr.item(1)
        wr = Vr.item(2)

        # compute angle of attack
        if ur == 0:
            self._alpha = np.sign(wr) * np.pi/2.0
        else:
            self._alpha = np.arctan2(wr,ur)

        # compute sideslip angle
        if ur == 0 and wr == 0:
            self._beta = np.sign(vr) * np.pi/2.0
        else:
            self._beta = np.arcsin(vr/self._Va)

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        fb_grav = Quaternion2Rotation(self._state[6:10]).T @ np.array([[0,0,
                            MAV.mass*MAV.gravity]]).T

        # longitudinal forces and moments
        fx,fz,m = self.calcLonDynamics(delta.item(0))
        fx += fb_grav.item(0)
        fz += fb_grav.item(2)

        # lateral forces and moments
        fy,l,n = self.calcLatDynamics(delta.item(2),delta.item(3))
        fy += fb_grav.item(1)

        # propeller/motor forces and moments
        fp,mp = self.calcMotorDynamics(self._Va, delta.item(1))
        fx += fp
        l -= mp

        self._forces[0] = fx
        self._forces[1] = fy
        self._forces[2] = fz
        return np.array([[fx, fy, fz, l, m, n]]).T

    def _update_msg_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self.msg_true_state.pn = self._state.item(0)
        self.msg_true_state.pe = self._state.item(1)
        self.msg_true_state.h = -self._state.item(2)
        self.msg_true_state.Va = self._Va
        self.msg_true_state.alpha = self._alpha
        self.msg_true_state.beta = self._beta
        self.msg_true_state.phi = phi
        self.msg_true_state.theta = theta
        self.msg_true_state.psi = psi
        self.msg_true_state.Vg = np.linalg.norm(self._state[3:6])
        self.msg_true_state.gamma = np.arctan2(-self._state.item(5),self._state.item(3))
        self.msg_true_state.chi = np.arctan2(self._state.item(4),self._state.item(3))
        self.msg_true_state.p = self._state.item(10)
        self.msg_true_state.q = self._state.item(11)
        self.msg_true_state.r = self._state.item(12)
        self.msg_true_state.wn = self._wind.item(0)
        self.msg_true_state.we = self._wind.item(1)


    def calcLonDynamics(self, de):
        M = MAV.M
        alpha = self._alpha
        alpha0 = MAV.alpha0
        rho = MAV.rho
        Va = self._Va
        S = MAV.S_wing
        q = self._state.item(11)
        c = MAV.c

        e_neg_M = exp(-M * (alpha - alpha0))
        e_pos_M = exp(M * (alpha + alpha0))

        sigma = (1 + e_neg_M + e_pos_M) / ((1 + e_neg_M) * (1 + e_pos_M))
        CL = (1-sigma)*(MAV.C_L_0 + MAV.C_L_alpha*alpha) + \
                sigma*(2*np.sign(alpha)*(np.sin(alpha)**2)*np.cos(alpha))
        CD = MAV.C_D_p+((MAV.C_L_0+MAV.C_L_alpha*alpha)**2)/(np.pi*MAV.e*MAV.AR)
        
        pVa2S_2 = 0.5*rho*(Va**2)*S
        c_2Va = c / (2*Va)

        Lift = pVa2S_2*(CL + MAV.C_L_q*c_2Va*q + MAV.C_L_delta_e*de)
        Drag = pVa2S_2*(CD+MAV.C_D_q*c_2Va*q + MAV.C_D_delta_e*de)

        fx = -Drag*np.cos(alpha) + Lift*np.sin(alpha)
        fz = -Drag*np.sin(alpha) - Lift*np.cos(alpha)

        m = pVa2S_2*c * (MAV.C_m_0 + MAV.C_m_alpha*alpha + MAV.C_m_q*c_2Va*q +\
                MAV.C_m_delta_e*de)

        return fx,fz,m

    def calcLatDynamics(self,da,dr):
        b = MAV.b
        Va = self._Va
        beta = self.msg_true_state.beta
        p = self._state.item(10)
        r = self._state.item(12)
        rho = MAV.rho
        S = MAV.S_wing

        pVa2S_2 = 0.5*rho*(Va**2)*S
        b_2Va = b / (2*Va)

        fy = pVa2S_2 * (MAV.C_Y_0 + MAV.C_Y_beta*beta + MAV.C_Y_p*b_2Va*p + \
                MAV.C_Y_r*b_2Va*r + MAV.C_Y_delta_a*da + MAV.C_Y_delta_r*dr) 

        l = pVa2S_2*b*(MAV.C_ell_0 + MAV.C_ell_beta*beta + MAV.C_ell_p*b_2Va*p+\
                MAV.C_ell_r*b_2Va*r + MAV.C_ell_delta_a*da+MAV.C_ell_delta_r*dr)

        n = pVa2S_2*b * (MAV.C_n_0 + MAV.C_n_beta*beta + MAV.C_n_p*b_2Va*p + \
                MAV.C_n_r*b_2Va*r + MAV.C_n_delta_a*da + MAV.C_n_delta_r*dr)
        
        return fy, l, n

    def calcMotorDynamics(self, Va, dt):
        rho = MAV.rho
        D = MAV.D_prop
#        Va = self._Va

        V_in = MAV.V_max * dt

        a = (rho * D**5) / ((2*np.pi)**2) * MAV.C_Q0
        b = (rho * (D**4) * MAV.C_Q1 * Va)/(2*np.pi) + (MAV.KQ**2)/MAV.R_motor
        c = rho * (D**3) * MAV.C_Q2 * (Va**2) - (MAV.KQ*V_in)/MAV.R_motor + \
                MAV.KQ * MAV.i0

        Omega_op = (-b + np.sqrt((b**2) - 4*a*c)) / (2*a)
        J_op = (2 * np.pi * Va) / (Omega_op * D)

        CT = MAV.C_T2 * (J_op**2) + MAV.C_T1 * J_op + MAV.C_T0
        CQ = MAV.C_Q2 * (J_op**2) + MAV.C_Q1 * J_op + MAV.C_Q0

        n = Omega_op / (2*np.pi)

        Qp = rho * (n**2) * (D**5) * CQ
        Tp = rho * (n**2) * (D**4) * CT

        return Tp, Qp

