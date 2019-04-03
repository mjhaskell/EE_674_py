# dubins_parameters
#   - Dubins parameters that define path between two configurations
#
# mavsim_matlab 
#     - Beard & McLain, PUP, 2012
#     - Update history:  
#         3/26/2019 - RWB

import numpy as np
import sys
sys.path.append('..')

class dubins_params:
    def __init__(self):
        self.p_s = np.inf*np.ones((3,1))  # the start position in re^3
        self.chi_s = np.inf  # the start course angle
        self.p_e = np.inf*np.ones((3,1))  # the end position in re^3
        self.chi_e = np.inf  # the end course angle
        self.radius = np.inf  # turn radius
        self.length = np.inf  # length of the Dubins path
        self.center_s = np.inf*np.ones((3,1))  # center of the start circle
        self.dir_s = np.inf  # direction of the start circle
        self.center_e = np.inf*np.ones((3,1))  # center of the end circle
        self.dir_e = np.inf  # direction of the end circle
        self.r1 = np.inf*np.ones((3,1))  # vector in re^3 defining half plane H1
        self.r2 = np.inf*np.ones((3,1))  # vector in re^3 defining position of half plane H2
        self.r3 = np.inf*np.ones((3,1))  # vector in re^3 defining position of half plane H3
        self.n1 = np.inf*np.ones((3,1))  # unit vector in re^3 along straight line path
        self.n3 = np.inf*np.ones((3,1))  # unit vector defining direction of half plane H3

    def update(self, ps, chis, pe, chie, R):
        ell = np.linalg.norm(ps[0:2] - pe[0:2])
        if ell < 2 * R:
            print('Error in Dubins Parameters: The distance between nodes must be larger than 2R.')
        else:
            Cxs = np.cos(chis)
            Sxs = np.sin(chis)
            Cxe = np.cos(chie)
            Sxe = np.sin(chie)
            c_rs = ps + R*rotz(np.pi/2) @ np.array([[Cxs,Sxs,0]]).T
            c_ls = ps + R*rotz(-np.pi/2) @ np.array([[Cxs,Sxs,0]]).T
            c_re = pe + R*rotz(np.pi/2) @ np.array([[Cxe,Sxe,0]]).T
            c_le = pe + R*rotz(-np.pi/2) @ np.array([[Cxe,Sxe,0]]).T
            
            pi = np.pi
            
            # compute L1
            theta = np.arctan2(c_re.item(1)-c_rs.item(1),c_re.item(0)-c_rs.item(0))
            L1 = np.linalg.norm(c_rs-c_re) + R*mod(2*pi+mod(theta-pi/2)-\
                    mod(chis-pi/2)) + R*mod(2*pi+mod(chie-pi/2)-mod(theta-pi/2))

            # compute L2
            ell = np.linalg.norm(c_rs - c_le)
            theta = np.arctan2(c_le.item(1)-c_rs.item(1),c_le.item(0)-c_rs.item(0))
            theta2 = theta - pi/2 + np.arcsin(2*R/ell)
            sqrt = np.sqrt(ell**2 - 4*R**2)
            L2 = sqrt + R*mod(2*pi+mod(theta-theta2)-mod(chis-pi/2)) +\
                          R*mod(2*pi+mod(theta2+pi)-mod(chie+pi/2))

            # compute L3
            ell = np.linalg.norm(c_ls - c_re)
            theta = np.arctan2(c_re.item(1)-c_ls.item(1),c_re.item(0)-c_ls.item(0))
            theta2 = np.arcsin(2*R/ell)
            sqrt = np.sqrt(ell**2 - 4*R**2)
            L3 = sqrt + R*mod(2*pi+mod(chis+pi/2)-mod(theta+theta2))+R*mod(\
                    2*pi+mod(chie-pi/2)-mod(theta+theta2-pi))

            # compute L4
            ell = np.linalg.norm(c_ls - c_le)
            theta = np.arctan2(c_le.item(1)-c_ls.item(1),c_le.item(0)-c_ls.item(0))
            L4 = ell + R*mod(2*pi+mod(chis+pi/2)-mod(theta+pi/2))+\
                    R*mod(2*pi+mod(theta+pi/2)-mod(chie+pi/2))

            L_list = [L1, L2, L3, L4]
            index = np.argmin(L_list)
            
            e1 = np.array([[1,0,0]]).T
            r3 = pe
            n3 = rotz(chie) @ e1
            if index == 0:
                cs = c_rs
                ce = c_re
                lam_s = 1
                lam_e = 1
                ell = np.linalg.norm(ce - cs)
                n1 = ce - cs
                n1 /= np.linalg.norm(n1)
                R_Rz_n1 = R*rotz(-pi/2) @ n1
                r1 = cs + R_Rz_n1
                r2 = ce + R_Rz_n1
            elif index == 1:
                cs = c_rs
                ce = c_le
                lam_s = 1
                lam_e = -1
                diff = ce - cs
                ell = np.linalg.norm(diff)
                theta = np.arctan2(diff.item(1),diff.item(0))
                theta2 = theta - pi/2 + np.arcsin(2*R/ell)
                n1 = rotz(theta2+pi/2) @ e1
                r1 = cs + R*rotz(theta2) @ e1
                r2 = ce + R*rotz(theta2+pi) @ e1
            elif index == 2:
                cs = c_ls
                ce = c_re
                lam_s = -1
                lam_e = 1
                diff = ce - cs
                ell = np.linalg.norm(diff)
                theta = np.arctan2(diff.item(1),diff.item(0))
                theta2 = np.arccos(2*R/ell)
                n1 = rotz(theta+theta2-pi/2) @ e1
                r1 = cs + R*rotz(theta+theta2) @ e1
                r2 = ce + R*rotz(theta+theta2-pi) @ e1
            else:
                cs = c_ls
                ce = c_le
                lam_s = -1
                lam_e = -1
                ell = np.linalg.norm(ce - cs)
                n1 = ce - cs
                n1 /= np.linalg.norm(n1)
                R_Rz_n1 = R*rotz(pi/2) @ n1
                r1 = cs + R_Rz_n1
                r2 = ce + R_Rz_n1

            self.p_s = ps
            self.chi_s = chis
            self.p_e = pe
            self.chi_e = chie
            self.radius = R
            self.length = ell
            self.center_s = cs
            self.dir_s = lam_s
            self.center_e = ce
            self.dir_e = lam_e
            self.r1 = r1
            self.n1 = n1
            self.r2 = r2
            self.r3 = r3
            self.n3 = n3

def rotz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])

def mod(x):
    return x % 2*np.pi
