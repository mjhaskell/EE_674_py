import sys
sys.path.append('..')
import numpy as np
#import chap5.transfer_function_coef as TF
import yaml

param_file = open('../params/tf_params.yaml','r')
params = yaml.load(param_file)
a_phi_1 = params.get('a_phi_1')
a_phi_2 = params.get('a_phi_2')
a_beta_1 = params.get('a_beta_1')
a_beta_2 = params.get('a_beta_2')
a_theta_1 = params.get('a_theta_1')
a_theta_2 = params.get('a_theta_2')
a_theta_3 = params.get('a_theta_3')
a_V_1 = params.get('a_V_1')
a_V_2 = params.get('a_V_2')
a_V_3 = params.get('a_V_3')

gravity = 9.8
#sigma = 
Va0 = 25

#----------roll loop-------------
tr = 0.5
wn = 2.2 / tr
h = 0.707

roll_kp = (wn**2) / a_phi_2
roll_kd = (2*h*wn - a_phi_1) / a_phi_2

#----------course loop-------------
W_X = 10
wn = wn / W_X
h = 5.0

course_kp = 2*h*wn * Va0/gravity
course_ki = (wn**2) * Va0/gravity

#----------sideslip loop-------------
dr_max = 1.0
e_max = 0.5
h = 5.0

sideslip_kp = dr_max / e_max
wn = (a_beta_1 + a_beta_2*sideslip_kp) / (2*h)
sideslip_ki = (wn**2) / a_beta_2

#----------yaw damper-------------
yaw_damper_tau_r = 0.05
yaw_damper_kp = 1

#----------pitch loop-------------
tr = 1
wn = 2.2 / tr
h = 0.707

pitch_kp = (wn**2 - a_theta_2) / a_theta_3
pitch_kd = (2*h*wn - a_theta_1) / a_theta_3
K_theta_DC = (pitch_kp*a_theta_3) / (a_theta_2 + pitch_kp*a_theta_3)

#----------altitude loop-------------
W_h = 10
wn = wn / W_h
h = 0.707

altitude_kp = (2*h*wn) / (K_theta_DC*Va0)
altitude_ki = (wn**2) / (K_theta_DC*Va0)
altitude_zone = 2

#---------airspeed hold using throttle---------------
tr = 1
wn = 2.2 / tr
h = 5.0

airspeed_throttle_kp = (2*h*wn - a_V_1) / a_V_2
airspeed_throttle_ki = wn**2 / a_V_2
