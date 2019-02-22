"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import sys
sys.path.append('..')
import yaml
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
    a_phi_1 = -0.5 * rho * (Va**2) * S * b * Cpp * b_2Va
    a_phi_2 = 0.5 * rho * (Va**2) * S * b * MAV.C_p_delta_a

    T_phi_delta_a = TF(np.array([a_phi_2]),np.array([1,a_phi_1,0]))
    T_chi_phi = TF(np.array([MAV.gravity/Va]),np.array([1,0]))

    frac = (rho*Va*S)/(2*MAV.mass*np.cos(beta))

    a_beta_1 = - frac * MAV.C_Y_beta
    a_beta_2 = frac * MAV.C_Y_delta_r 

    T_beta_delta_r = TF(np.array([a_beta_2]),np.array([1,a_beta_1]))
    T_v_delta_r = TF(np.array([a_beta_2*Va*np.cos(beta)]),np.array([1,a_beta_1]))

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
    a_V_1 -= dT_dVa(mav,Va,dt) / MAV.mass
    a_V_2 = dT_ddelta_t(mav,Va,dt) / MAV.mass
    a_V_3 = MAV.gravity * np.cos(theta - alpha)

    T_Va_delta_t = TF(np.array([a_V_2]),np.array([1,a_V_1]))
    T_Va_theta = TF(np.array([-a_V_3]),np.array([1,a_V_1]))

    outfile = open('tf_params.yaml','w')
    data = {
           'a_phi_1': float(a_phi_1),
           'a_phi_2': float(a_phi_2),
           'a_beta_1': float(a_beta_1),
           'a_beta_2': float(a_beta_2),
           'a_theta_1': float(a_theta_1),
           'a_theta_2': float(a_theta_2),
           'a_theta_3': float(a_theta_3),
           'a_V_1': float(a_V_1),
           'a_V_2': float(a_V_2),
           'a_V_3': float(a_V_3)
           }
    yaml.dump(data,outfile,default_flow_style=False)

    return T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, T_h_Va, \
            T_Va_delta_t, T_Va_theta, T_beta_delta_r, T_v_delta_r

def compute_ss_model(mav, trim_state, trim_input):
    euler_trim = euler_state(trim_state)
    A = df_dx(mav, euler_trim, trim_input)
    B = df_du(mav, euler_trim, trim_input)

#    print('A: \n',A)

    pn = 0
    pe = 1
    h = 2
    u = 3
    v = 4
    w = 5
    phi = 6
    th = 7
    psi = 8
    p = 9
    q = 10
    r = 11

    de = 0
    dt = 1
    da = 2
    dr = 3

    A_lon = np.array([
            [A[u,u],  A[u,w],  A[u,q],  A[u,th],  A[u,h]],
            [A[w,u],  A[w,w],  A[w,q],  A[w,th],  A[w,h]],
            [A[q,u],  A[q,w],  A[q,q],  A[q,th],  A[q,h]],
            [A[th,u], A[th,w], A[th,q], A[th,th], A[th,h]],
            [A[h,u],  A[h,w],  A[h,q],  A[h,th],  A[h,h]]
            ])
    A_lon[4,:] = -A_lon[4,:]
#    A_lon = getALon(mav,trim_state,trim_input)

    B_lon = np.array([
            [B[u,de],  B[u,dt]],
            [B[w,de],  B[w,dt]],
            [B[q,de],  B[q,dt]],
            [B[th,de], B[th,dt]],
            [B[h,de],  B[h,dt]]
            ])

    A_lat = np.array([
            [A[v,v],   A[v,p],   A[v,r],   A[v,phi],   A[v,psi]],
            [A[p,v],   A[p,p],   A[p,r],   A[p,phi],   A[p,psi]],
            [A[r,v],   A[r,p],   A[r,r],   A[r,phi],   A[r,psi]],
            [A[phi,v], A[phi,p], A[phi,r], A[phi,phi], A[phi,psi]],
            [A[psi,v], A[psi,p], A[psi,r], A[psi,phi], A[psi,psi]]
            ])
#    A_lat = getALat(mav,trim_state,trim_input)

    B_lat = np.array([
            [B[v,da],   B[v,dr]],
            [B[p,da],   B[p,dr]],
            [B[r,da],   B[r,dr]],
            [B[phi,da], B[phi,dr]],
            [B[psi,da], B[psi,dr]]
            ])

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

def dq_dtheta(euler):
    h = 0.005
    Jacobian = np.zeros((4,3))
    for i in range(3):
        e_p = np.copy(euler)
        e_p[i][0] += h
        e_m = np.copy(euler)
        e_m[i][0] -= h
        
        f_p = Euler2Quaternion(e_p.item(0),e_p.item(1),e_p.item(2))
        f_m = Euler2Quaternion(e_m.item(0),e_m.item(1),e_m.item(2))

        Jac_col_i = (f_p - f_m) / (2 * h)
        Jacobian[:,i] = Jac_col_i[:,0]

    return Jacobian

def f_euler(mav, x_euler, delta):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state
    x_quat = quaternion_state(x_euler)
    mav._state = x_quat
    mav._update_velocity_data()
    u = mav._forces_moments(delta)
    f_euler_ = mav._derivatives(x_quat,u)

    return f_euler_

def f_quat(mav, x_quat, delta):
    # return 13x1 dynamics (as if state were Euler state)
    # compute f at euler_state
    mav._state = x_quat
    mav._update_velocity_data()
#    print('Va: \n',mav._Va)
    u = mav._forces_moments(delta)
    f_quat_ = mav._derivatives(x_quat,u)

    return f_quat_

def df_dx(mav, x_euler, delta):
    # take partial of f_euler with respect to x_euler
    x_quat = quaternion_state(x_euler)
    dT_dx = np.zeros((12,13))
    dT_dx[:6,:6] = np.eye(6)
    dT_dx[9:12,10:13] = np.eye(3)
    dT_dx[6:9,6:10] = dtheta_dq(x_quat[6:10])

#    print('dT_dx: \n',dT_dx[6:9,6:10])

    dTinv_dx = np.zeros((13,12))
    dTinv_dx[:6,:6] = np.eye(6)
    dTinv_dx[10:13,9:12] = np.eye(3)
    dTinv_dx[6:10,6:9] = dq_dtheta(x_euler[6:9])

    print('dTinv_dx: \n',dTinv_dx)

    h = 0.005
    Jacobian = np.zeros((13,13))
    for i in range(13):
        x_p = np.copy(x_quat)
        x_p[i][0] += h
        x_m = np.copy(x_quat)
        x_m[i][0] -= h

#        print('x_p: \n',x_p)

        f_p = f_quat(mav,x_p,delta)
        f_m = f_quat(mav,x_m,delta)

        Jac_col_i = (f_p - f_m) / (2 * h)
        Jacobian[:,i] = Jac_col_i[:,0]

    A = dT_dx @ Jacobian @ dTinv_dx

    return A

def df_dx_euler(mav, x_euler, delta):
    # take partial of f_euler with respect to x_euler
    h = 0.005
    A = np.zeros((12,12))
    for i in range(12):
        x_p = np.copy(x_euler)
        x_p[i][0] += h
        x_m = np.copy(x_euler)
        x_m[i][0] -= h

        f_p = f_euler(mav,x_p,delta)
        f_m = f_euler(mav,x_m,delta)

        Jac_col_i = (f_p - f_m) / (2 * h)
        A[:,i] = Jac_col_i[:,0]
    
    return A

def getALat(mav, trim_state, trim_input):
    # take partial of f_euler with respect to x_euler
    rho = MAV.rho
    m = MAV.mass
    u = mav._state.item(3)
    w = mav._state.item(5)
    Va = mav._Va
    S = MAV.S_wing
    b = MAV.b
    c = MAV.c
    Cpp = MAV.C_p_p
    Cpb = MAV.C_p_beta
    Cp0 = MAV.C_p_0
#    b_2Va = b / (2*Va)
#    c_2Va = c / (2*Va)
#    beta = mav._beta
#    alpha = mav._alpha
    phi,theta,psi = Quaternion2Euler(trim_state[6:10])

    frac = rho*S/2
    Y_v = frac*MAV.C_Y_beta/m * Va
    Y_p = w + frac*Va*b/(2*m) * MAV.C_Y_p
    Y_r = -u + frac*Va*b/(2*m) * MAV.C_Y_r
    Y_da = frac*(Va**2)/m * MAV.C_Y_delta_a
    Y_dr = frac*(Va**2)/m * MAV.C_Y_delta_r

    L_v = frac*b*Cpb*Va
    L_p = frac*Va*(b**2)/2.0 * Cpp
    L_r = frac*Va*(b**2)/2.0 * MAV.C_p_r
    L_da = frac*(Va**2)*b * MAV.C_p_delta_a
    L_dr = frac*(Va**2)*b * MAV.C_p_delta_r

    N_v = frac*b*MAV.C_r_beta * Va
    N_p = frac*Va*(b**2)/2.0 * MAV.C_r_p
    N_r = frac*Va*(b**2)/2.0 * MAV.C_r_r
    N_da = frac*(Va**2)*b * MAV.C_r_delta_a
    N_dr = frac*(Va**2)*b * MAV.C_r_delta_r

    A_lat = np.array([
            [Y_v, Y_p, Y_r, MAV.gravity*np.cos(theta)*np.cos(phi), 0],
            [L_v, L_p, L_r, 0, 0],
            [N_v, N_p, N_r, 0, 0],
            [0, 1, np.cos(phi)*np.tan(theta), 0, 0],
            [0, 0, np.cos(phi)/np.cos(theta), 0, 0],
            ])

    return A_lat

def getALon(mav,trim_state,trim_input):
    rho = MAV.rho
    m = MAV.mass
    u = mav._state.item(3)
    w = mav._state.item(5)
    Va = mav._Va
    S = MAV.S_wing
    b = MAV.b
    c = MAV.c
    Cpp = MAV.C_p_p
    Cpb = MAV.C_p_beta
    Cp0 = MAV.C_p_0
#    b_2Va = b / (2*Va)
#    c_2Va = c / (2*Va)
    de = trim_input.item(0)
    beta = mav._beta
    alpha = mav._alpha
    phi,theta,psi = Quaternion2Euler(trim_state[6:10])

    

#    X_u = u*rho*S/m * (MAV.C_X_0 + MAV.C_X_alpha*alpha + MAV.C_X_delta_e*de) \

    A_lon - 0
    return A_lon

def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to delta
    x_quat = quaternion_state(x_euler)
    dT_dx = np.zeros((12,13))
    dT_dx[:6,:6] = np.eye(6)
    dT_dx[9:12,10:13] = np.eye(3)
    dT_dx[6:9,6:10] = dtheta_dq(x_quat[6:10])

    h = 0.005
    Jacobian = np.zeros((13,4))
    for i in range(4):
        u_p = np.copy(delta)
        u_p[i][0] += h
        u_m = np.copy(delta)
        u_m[i][0] -= h

        f_p = f_euler(mav,x_euler,u_p)
        f_m = f_euler(mav,x_euler,u_m)

        Jac_col_i = (f_p - f_m) / (2 * h)
        Jacobian[:,i] = Jac_col_i[:,0]

    B = dT_dx @ Jacobian

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

#    trim_euler = euler_state(trim_state)
#    trim_quat = quaternion_state(trim_euler)
#    print('trim: \n',trim_state)
#    print('euler: \n',trim_euler)
#    print('quat: \n',trim_quat)
   
#    A_lat = getALat(dyn,trim_state,trim_input)
#    A_lon, B_lon, A_lat, B_lat = compute_ss_model(dyn, trim_state, trim_input)
#    print('A_lon: \n',A_lon)
#    print('B_lon: \n',B_lon)
#    print('A_lat: \n',A_lat)
#    print('B_lat: \n',B_lat)

    T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, \
    T_h_Va, T_Va_delta_t, T_Va_theta, T_beta_delta_r, T_v_delta_r \
        = compute_tf_model(dyn, trim_state, trim_input)
#    print('T_phi_delta_a: \n',T_phi_delta_a)
#    print('T_chi_phi: \n',T_chi_phi)
#    print('T_theta_delta_e: \n',T_theta_delta_e)
#    print('T_h_theta: \n',T_h_theta)
#    print('T_h_Va: \n',T_h_Va)
#    print('T_Va_delta_t: \n',T_Va_delta_t)
#    print('T_Va_theta: \n',T_Va_theta)
#    print('T_beta_delta_r: \n',T_beta_delta_r)
#    print('T_v_delta_r: \n',T_v_delta_r)

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

#    x_euler = np.array([[2,0,-2,1,0,1,0,np.pi/4,0,0,0,0]]).T
#    delta = np.array([[-0.1,0.7,trim_input.item(2),trim_input.item(3)]]).T
#    A = df_dx_euler(dyn,x_euler,delta)
#    print('df_dx: \n',A)
