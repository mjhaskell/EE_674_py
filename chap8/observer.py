"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
"""
import sys
import numpy as np
sys.path.append('..')
import params.control_params as CTRL
import params.sim_params as SIM
import params.sensor_params as SENSOR
import params.aerosonde_params as MAV
from tools.tools import Euler2Rotation

from message_types.msg_state import msg_state

class observer:
    def __init__(self, ts_control):
        # initialized estimated state message
        self.estimated_state = msg_state()
        # use alpha filters to low pass filter gyros and accels
        self.lpf_gyro_x = alpha_filter(alpha=0.5)
        self.lpf_gyro_y = alpha_filter(alpha=0.5)
        self.lpf_gyro_z = alpha_filter(alpha=0.5)
        self.lpf_accel_x = alpha_filter(alpha=0.5)
        self.lpf_accel_y = alpha_filter(alpha=0.5)
        self.lpf_accel_z = alpha_filter(alpha=0.5)
        # use alpha filters to low pass filter static and differential pressure
        self.lpf_static = alpha_filter(alpha=0.9)
        self.lpf_diff = alpha_filter(alpha=0.5)
        # ekf for phi and theta
        self.attitude_ekf = ekf_attitude()
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = ekf_position()

    def update(self, measurements):

        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        bx = self.estimated_state.bx
        by = self.estimated_state.by
        bz = self.estimated_state.bz
        self.estimated_state.p = self.lpf_gyro_x.update(measurements.gyro_x-bx)
        self.estimated_state.q = self.lpf_gyro_y.update(measurements.gyro_y-by)
        self.estimated_state.r = self.lpf_gyro_z.update(measurements.gyro_z-bz)

        # invert sensor model to get altitude and airspeed
        rho = MAV.rho
        g = MAV.gravity
        lpf_static = self.lpf_static.update(measurements.static_pressure)
        lpf_diff = self.lpf_diff.update(measurements.diff_pressure)
        self.estimated_state.h = lpf_static / (rho*g)
        self.estimated_state.Va = np.sqrt(2.0 * lpf_diff / rho)

        # estimate phi and theta with simple ekf
        self.attitude_ekf.update(self.estimated_state, measurements)

        # estimate pn, pe, Vg, chi, wn, we, psi
        self.position_ekf.update(self.estimated_state, measurements)

        # not estimating these
        self.estimated_state.alpha = self.estimated_state.theta
        self.estimated_state.beta = 0.0
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0

        return self.estimated_state

class alpha_filter:
    # alpha filter implements a simple low pass filter
    # y[k] = alpha * y[k-1] + (1-alpha) * u[k]
    def __init__(self, alpha=0.5, y0=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0  # initial condition

    def update(self, u):
        self.y = self.alpha*self.y + (1.0-self.alpha)*u
        return self.y

class ekf_attitude:
    # implement continous-discrete EKF to estimate roll and pitch angles
    def __init__(self):
        self.Q = np.eye(2) * 1e-6
        self.Q_gyro = np.eye(3)*SENSOR.gyro_sigma**2
        self.R_accel = np.eye(3)*SENSOR.accel_sigma**2
        self.N = 10  # number of prediction step per sample
        self.xhat = np.array([[0.0, 0.0]]).T # initial state: phi, theta
        self.P = np.eye(2)*0.1
        self.Ts = SIM.ts_control/self.N

    def update(self, state, measurement):
        self.propagate_model(state)
        self.measurement_update(state, measurement)
        state.phi = self.xhat.item(0)
        state.theta = self.xhat.item(1)

    def f(self, x, state):
        # system dynamics for propagation model: xdot = f(x, u)
        phi = x.item(0)
        theta = x.item(1)
        p = state.p
        q = state.q
        r = state.r

        _f = np.array([
            [p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)],
            [q*np.cos(phi) - r*np.sin(phi)]
            ])
        return _f

    def h(self, x, state):
        # measurement model y
        phi = x.item(0)
        theta = x.item(1)
        p = state.p
        q = state.q
        r = state.r
        Va = state.Va
        g = MAV.gravity

        _h = np.array([[(q*Va+g)*np.sin(theta)],
            [r*Va*np.cos(theta)-p*Va*np.sin(theta)-g*np.cos(theta)*np.sin(phi)],
            [-q*Va*np.cos(theta) - g*np.cos(theta)*np.cos(phi)]])
        return _h

    def propagate_model(self, state):
        # model propagation
        for i in range(0, self.N):
            phi = self.xhat.item(0)
            theta = self.xhat.item(1)
             # propagate model
            self.xhat = self.xhat + self.Ts*self.f(self.xhat, state)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state)
            # compute G matrix for gyro noise
            G = np.array([
                [1.0, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                [0.0, np.cos(phi), -np.sin(phi)]
                ])
            # update P with continuous time model
            # self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q + G @ self.Q_gyro @ G.T)
            # convert to discrete time models
            A_d = np.eye(2) + A*self.Ts + (A@A)*(self.Ts**2) / 2.0
            G_d = G * self.Ts
            # update P with discrete time model
            self.P = A_d@self.P@A_d.T + G_d@self.Q_gyro@G_d.T + \
                     self.Q*self.Ts**2

    def measurement_update(self, state, measurement):
        # measurement updates
        threshold = 2.0
        h = self.h(self.xhat, state)
        C = jacobian(self.h, self.xhat, state)
        y = np.array([[measurement.accel_x, measurement.accel_y,
                       measurement.accel_z]]).T

        L = self.P@C.T @ np.linalg.inv(self.R_accel + C@self.P@C.T)
        self.P = (np.eye(2) - L@C) @ self.P @ (np.eye(2) - L@C).T + \
                  L@self.R_accel@L.T
        self.xhat += L @ (y - h)

class ekf_position:
    # implement continous-discrete EKF to estimate pn, pe, chi, Vg
    def __init__(self):
        #self.Q = np.eye(7) * 0.1**2
        self.Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        self.R_gps = np.diag([SENSOR.gps_n_sigma**2, SENSOR.gps_e_sigma**2,
                          SENSOR.gps_Vg_sigma**2, SENSOR.gps_course_sigma**2])
        self.R_pseudo = np.diag([0.01,0.01])
        self.N = 25  # number of prediction step per sample
        self.Ts = (SIM.ts_control / self.N)
        self.xhat = np.array([[0.,0.,25.,0.,0.,0.,0.]]).T
        self.P = np.eye(7) * 0.5
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999

    def update(self, state, measurement):
        self.propagate_model(state)
        self.measurement_update(state, measurement)
        state.pn = self.xhat.item(0)
        state.pe = self.xhat.item(1)
        state.Vg = self.xhat.item(2)
        state.chi = self.xhat.item(3)
        state.wn = self.xhat.item(4)
        state.we = self.xhat.item(5)
        state.psi = self.xhat.item(6)

    def f(self, x, state):
        # system dynamics for propagation model: xdot = f(x, u)
        Vg = x.item(2)
        chi = x.item(3)
        wn = x.item(4)
        we = x.item(5)
        psi = x.item(6)
        Va = state.Va
        phi = state.phi
        theta = state.theta
        q = state.q
        r = state.r

        psi_d = q*np.sin(phi)/np.cos(theta) + r*np.cos(phi)/np.cos(theta)

        _f = np.array([
            [Vg*np.cos(chi)],
            [Vg*np.sin(chi)],
            [((Va*np.cos(psi)+wn)*(-Va*psi_d*np.sin(psi))+(Va*np.sin(psi)+we)*(Va*psi_d*np.cos(psi)))/Vg],
            [MAV.gravity/Vg * np.tan(phi)*np.cos(chi-psi)],
            [0.0],
            [0.0],
            [psi_d]
            ])
        return _f

    def h_gps(self, x, state):
        # measurement model for gps measurements
        _h = x[0:4,0].reshape((4,1))

        return _h

    def h_pseudo(self, x, state):
        # measurement model for wind triangale pseudo measurement
        Va = state.Va
        Vg = x.item(2)
        chi = x.item(3)
        wn = x.item(4)
        we = x.item(5)
        psi = x.item(6)

        _h = np.array([[Va*np.cos(psi)+wn-Vg*np.cos(chi)],
                       [Va*np.sin(psi)+we-Vg*np.sin(chi)]])
        return _h

    def propagate_model(self, state):
        # model propagation
        for i in range(0, self.N):
            # propagate model
            self.xhat += self.Ts * self.f(self.xhat,state) 
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state)
            # update P with continuous time model
            # self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q + G @ self.Q_gyro @ G.T)
            # convert to discrete time models
            A_d = np.eye(7) + A*self.Ts + A@A * self.Ts**2/2
            # update P with discrete time model
            self.P = A_d@self.P@A_d.T + self.Q*self.Ts**2

    def measurement_update(self, state, measurement):
        # always update based on wind triangle pseudu measurement
        h = self.h_pseudo(self.xhat, state)
        C = jacobian(self.h_pseudo, self.xhat, state)
        y = np.array([[0, 0]]).T

        L = self.P @ C.T @ np.linalg.inv(self.R_pseudo+ C @ self.P @ C.T)
        self.P = (np.eye(7) - L @ C) @ self.P @ (np.eye(7) - L @ C).T + \
                  L @ self.R_pseudo @ L.T
        self.xhat += L@(y - h)

        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):

            h = self.h_gps(self.xhat, state)
            C = jacobian(self.h_gps, self.xhat, state)
            y = np.array([[measurement.gps_n, measurement.gps_e,
                           measurement.gps_Vg, measurement.gps_course]]).T

            L = self.P @ C.T @ np.linalg.inv(self.R_gps + C @ self.P @ C.T)

            y[3,0] = self.wrap(y[3,0], h[3,0])

            self.P = (np.eye(7) - L @ C) @ self.P @ (np.eye(7) - L @ C).T + \
                      L @ self.R_gps @ L.T
            self.xhat += L @ (y - h)
             
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course

    def wrap(self, chi_c, chi):
        while chi_c-chi > np.pi:
            chi_c = chi_c - 2.0 * np.pi
        while chi_c-chi < -np.pi:
            chi_c = chi_c + 2.0 * np.pi
        return chi_c

def jacobian(fun, x, state):
    # compute jacobian of fun with respect to x
    f = fun(x, state)
    m = f.shape[0]
    n = x.shape[0]
    eps = 0.01  # deviation
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i][0] += eps
        f_eps = fun(x_eps, state)
        df = (f_eps - f) / eps
        J[:, i] = df[:, 0]
    return J
