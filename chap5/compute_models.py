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
    euler_trim = euler_state(trim_state)
    A = np.empty((13,12))
    for x in range(len(euler_trim)):
        A[:,x] = f_euler(mav,euler_trim,trim_input)


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

def dtheta_dq(quat):
    h = 0.005
    Jacobian = np.zeros((3,4))
    for i in range(4):
        q_p = np.copy(quat)
        q_p[i][0] += h
        q_m = np.copy(quat)
        q_m[i][0] -= h

        phi,theta,psi = Quaternion2Euler(q_p)
        f_p = np.array([[phi,theta,psi]]).T
        
        phi,theta,psi = Quaternion2Euler(q_m)
        f_m = np.array([[phi,theta,psi]]).T

        Jac_col_i = (f_p - f_m) / (2 * h)
        Jacobian[:,i] = Jac_col_i[:,0]

    return Jacobian

def f_euler(mav, x_euler, delta):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state
    u = mav._forces_moments(delta)
    x_quat = quaternion_state(x_euler)
    f_euler_ = mav._derivatives(x_quat,u)

    return f_euler_

def df_dx(mav, x_euler, delta):
    # take partial of f_euler with respect to x_euler
    x_quat = quaternion_state(x_euler)
    dT_dx = np.zeros((12,13))
    dT_dx[:6,:6] = np.eye(6)
    dT_dx[9:12,10:13] = np.eye(3)
    dT_dx[6:9,6:10] = dtheta_dq(x_quat[6:10])

#    u = mav._forces_moments(delta)
    
    h = 0.005
    Jacobian = np.zeros((13,12))
    for i in range(12):
        x_p = np.copy(x_euler)
        x_p[i][0] += h
        x_m = np.copy(x_euler)
        x_m[i][0] -= h

#        f_p = mav._derivatives(x_p,u)
#        f_m = mav._derivatives(x_m,u)

        f_p = f_euler(mav,x_p,delta)
        f_m = f_euler(mav,x_m,delta)

        Jac_col_i = (f_p - f_m) / (2 * h)
        Jacobian[:,i] = Jac_col_i[:,0]

    A = dT_dx @ Jacobian
    
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

#    T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, \
#    T_h_Va, T_Va_delta_t, T_Va_theta, T_beta_delta_r \
#        = compute_tf_model(dyn, trim_state, trim_input)
#    print('T_phi_delta_a: \n',T_phi_delta_a)
#    print('T_chi_phi: \n',T_chi_phi)
#    print('T_theta_delta_e: \n',T_theta_delta_e)
#    print('T_h_theta: \n',T_h_theta)
#    print('T_h_Va: \n',T_h_Va)
#    print('T_Va_delta_t: \n',T_Va_delta_t)
#    print('T_Va_theta: \n',T_Va_theta)
#    print('T_beta_delta_r: \n',T_beta_delta_r)

#    q = np.array([[1,0,0,0]]).T
#    x_e = np.array([[1,2,3,4,5,6,0,np.pi/6.0,0,11,12,13]]).T
#    x_q = quaternion_state(x_e)
#    print('quat: \n',x_q)
#    x_e_test = euler_state(x_q)
#    print('euler: \n',x_e_test)

#    angle = np.pi / 6.0
#    axis = np.array([[0,1,0]]).T
#    q0 = np.cos(angle/2.0)
#    q1 = np.sin(angle/2.0)*axis.item(0)
#    q2 = np.sin(angle/2.0)*axis.item(1)
#    q3 = np.sin(angle/2.0)*axis.item(2)
#    quat = np.array([[q0,q1,q2,q3]]).T
#    A = dtheta_dq(quat)
#    print('dTheta_dQuat: \n',A)

    x_euler = np.array([[2,0,-2,1,0,1,0,np.pi/4,0,0,0,0]]).T
    delta = np.array([[-0.1,0.7,trim_input.item(2),trim_input.item(3)]]).T
    wrench = dyn._forces_moments(delta)
    A = df_dx(dyn,x_euler,wrench)
    print('df_dx: \n',A)