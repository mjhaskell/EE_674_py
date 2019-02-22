import sys
import numpy as np
sys.path.append('..')

class pid_control:
    def __init__(self,kp=0.0,ki=0.0,kd=0.0,limits=[-1.0,1.0],Ts=0.01,sigma=0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.Ts = Ts
        self.low_limit = limits[0]
        self.up_limit  = limits[1]

        self.error_int = 0.0
        self.y_dot = 0.0
        self.y_d1 = 0.0
        # gains for differentiator
        self.a1 = (2.0 * sigma - Ts) / (2.0 * sigma + Ts)
        self.a2 = 2.0 / (2.0 * sigma + Ts)

    def update(self, y_ref, y, wrap_flag=False):
        error = y_ref - y
        if wrap_flag:
            while error > np.pi:
                error -= 2*np.pi
            while error < np.pi:
                error += 2*np.pi
        self._integrateError(error)
        self._differentiateY(y)

        u_unsat = self.kp*error + self.ki*self.error_int + self.kd*self.y_dot
        u_sat = self._saturate(u_unsat)

        return u_sat

    def update_with_rate(self, y_ref, y, ydot, reset_flag=False):
        error = y_ref - y
        self._integrateError(error)

        u_unsat = self.kp*error + self.ki*self.error_int + self.kd*ydot
        u_sat = self._saturate(u_unsat)

        return u_sat

    def _saturate(self, u):
        if u > self.up_limit:
            u_sat = self.up_limit
        elif u < -self.low_limit:
            u_sat = -self.low_limit
        else:
            u_sat = u
        return u_sat

    def _integrateError(self, error):
        self.error_int += self.Ts/2.0 * (error _ self.error_d1)
        self.error_d1 = error

    def _differentiateY(self, y):
        self.y_dot = self.a1*self.y_dot + self.a2*(y - self.y_d1)
        self.y_d1 = y

