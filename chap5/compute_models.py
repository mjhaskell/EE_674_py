"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.tools import Euler2Quaternion, Quaternion2Euler
from control import TransferFunction as TF
import params.aerosonde_params as MAV
from params.sim_params import ts_simulation as Ts

def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    rho = MAV.rho
    Va = mav._Va
    S = MAV.S_wing
    b = MAV.b
    c = MAV.c
    Cpp = MAV.C_p_p
    Cpb = MAV.C_p_beta
    Cp0 = MAV.C_p_0
    b_2Va = b / (2*Va)
    c_2Va = c / (2*Va)
    beta = mav._beta
    alpha = mav._alpha
    de = trim_input.item(0)
    dt = trim_input.item(1)
    da = trim_input.item(2)
    dr = trim_input.item(3)
    phi,theta,psi = Quaternion2Euler(trim_state[6:10])

    # lateral transfer functions
    a_phi_1 = 0.5 * rho * (Va**2) * S * b * Cpp * b_2Va
    a_phi_2 = 0.5 * rho * (Va**2) * S * b * MAV.C_p_delta_a

    T_phi_delta_a = TF(np.array([a_phi_2]),np.array([1,a_phi_1,0]))
    T_chi_phi = TF(np.array([MAV.gravity/Va]),np.array([1,0]))

    frac = (rho*Va*S)/(2*MAV.mass*np.cos(beta))

    a_beta_1 = - frac * MAV.C_Y_beta
    a_beta_2 = frac * MAV.C_Y_delta_r

    T_beta_delta_r = TF(np.array([a_beta_2]),np.array([1,a_beta_1]))

    # Longitudinal transfer functions
    frac = (rho*(Va**2)*c*S) / (2*MAV.Jy)

    a_theta_1 = -frac * MAV.C_m_q * c_2Va
    a_theta_2 = -frac * MAV.C_m_alpha
    a_theta_3 = frac * MAV.C_m_delta_e

    T_theta_delta_e = TF(np.array([a_theta_3]),np.array([1,a_theta_1,a_theta_2]))
    T_h_theta = TF(np.array([Va]),np.array([1,0]))
    T_h_Va = TF(np.array([theta]),np.array([1,0]))

    frac = rho*Va*S / MAV.mass
    a_V_1 = frac * (MAV.C_D_0 + MAV.C_D_alpha*alpha + MAV.C_D_delta_e*de)
    a_V_1 -= dT_dVa(mav,Va,dt)
    a_V_2 = dT_ddelta_t(mav,Va,dt)
    a_V_3 = MAV.gravity * np.cos(theta - alpha)

    T_Va_delta_t = TF(np.array([a_V_2]),np.array([1,a_V_1]))
    T_Va_theta = TF(np.array([-a_V_3]),np.array([1,a_V_1]))

    return T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, T_h_Va, T_Va_delta_t, T_Va_theta, T_beta_delta_r

def compute_ss_model(mav, trim_state, trim_input):

     return A_lon, B_lon, A_lat, B_lat

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    phi, theta, psi = Quaternion2Euler(x_quat[6:10])
    x_euler = np.empty((12,1))
    x_euler[:6] = x_quat[:6]
    x_euler[6:9] = np.array([[phi],[theta],[psi]])
    x_euler[9:12] = x_quat[10:13]

    return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions
    phi = x_euler.item(6)
    theta = x_euler.item(7)
    psi = x_euler.item(8)
    q = Euler2Quaternion(phi,theta,psi)
    x_quat = np.empty((13,1))
    x_quat[:6] = x_euler[:6]
    x_quat[6:10] = q
    x_quat[10:13] = x_euler[9:12]

    return x_quat

def f_euler(mav, x_euler, input):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state
    return f_euler_

def df_dx(mav, x_euler, input):
    # take partial of f_euler with respect to x_euler
    return A

def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to delta
    return B

def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    h = 0.01
    f_f,_ = mav.calcMotorDynamics(Va + h,delta_t)
    f_b,_ = mav.calcMotorDynamics(Va - h,delta_t)

    dThrust = (f_f - f_b) / (2.0 * h)
    return dThrust

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    h = 0.001
    f_f,_ = mav.calcMotorDynamics(Va,delta_t + h)
    f_b,_ = mav.calcMotorDynamics(Va,delta_t - h)

    dThrust = (f_f - f_b) / (2.0 * h)
    return dThrust

if __name__ == "__main__":
    from chap4.mav_dynamics import mav_dynamics
    from chap5.trim import compute_trim
    
    dyn = mav_dynamics(0.02)
    Va = 25.
    gamma = 0. * np.pi/180.0
    trim_state, trim_input = compute_trim(dyn, Va, gamma)
    dyn._state = trim_state

    T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, \
    T_h_Va, T_Va_delta_t, T_Va_theta, T_beta_delta_r \
        = compute_tf_model(dyn, trim_state, trim_input)
    print('T_phi_delta_a: \n',T_phi_delta_a)
    print('T_chi_phi: \n',T_chi_phi)
    print('T_theta_delta_e: \n',T_theta_delta_e)
    print('T_h_theta: \n',T_h_theta)
    print('T_h_Va: \n',T_h_Va)
    print('T_Va_delta_t: \n',T_Va_delta_t)
    print('T_Va_theta: \n',T_Va_theta)
    print('T_beta_delta_r: \n',T_beta_delta_r)

    q = np.array([[1,0,0,0]]).T
    x_e = np.array([[1,2,3,4,5,6,0,np.pi/6.0,0,11,12,13]]).T
    x_q = quaternion_state(x_e)
    print('quat: \n',x_q)
    x_e_test = euler_state(x_q)
    print('euler: \n',x_e_test)
