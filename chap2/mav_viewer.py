import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.Vector as Vector

class mav_viewer():
    def __init__(self):
        # initialize Qt gui application and window
        self.app = pg.QtGui.QApplication([])  # initialize QT
        self.window = gl.GLViewWidget()  # initialize the view object
        self.window.setWindowTitle('Fixedwing Viewer')
        self.window.setGeometry(0, 0, 1000, 1000)  # args: upper_left_x, upper_right_y, width, height
        grid = gl.GLGridItem() # make a grid to represent the ground
        grid.scale(20, 20, 20) # set the size of the grid (distance between each line)
        self.window.addItem(grid) # add grid to viewer
        self.window.setCameraPosition(distance=200) # distance from center of plot to camera
        self.window.setBackgroundColor('k')  # set background color to black
        self.window.show()  # display configured window
        self.window.raise_() # bring window to the front
        self.plot_initialized = False # has the spacecraft been plotted yet?
        # get points that define the non-rotated, non-translated spacecraft and the mesh colors
        self.points, self.mesh_colors = self.getMAVPoints()

    ###################################
    # public functions
    def update(self, state):
        """
        Update the drawing of the spacecraft.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed to be:
            state.pn  # north position
            state.pe  # east position
            state.h   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        mav_position = np.array([[state.pn], [state.pe], [-state.h]])  # NED coordinates
        # attitude of spacecraft as a rotation matrix R from body to inertial
        R = self.Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining spacecraft
        rotated_points = self.rotatePoints(self.points, R)
        translated_points = self.translatePoints(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self.pointsToMesh(translated_points)

        # initialize the drawing the first time update() is called
        if not self.plot_initialized:
            # initialize drawing of triangular mesh.
            self.body = gl.GLMeshItem(vertexes=mesh,  # defines the triangular mesh (Nx3x3)
                                      vertexColors=self.mesh_colors, # defines mesh colors (Nx1)
                                      drawEdges=True,  # draw edges between mesh elements
                                      smooth=False,  # speeds up rendering
                                      computeNormals=False)  # speeds up rendering
            self.window.addItem(self.body)  # add body to plot
            self.plot_initialized = True

        # else update drawing on all other calls to update()
        else:
            # reset mesh using rotated and translated points
            self.body.setMeshData(vertexes=mesh, vertexColors=self.mesh_colors)

        # update the center of the camera view to the spacecraft location
        view_location = Vector(state.pe, state.pn, state.h)  # defined in ENU coordinates
        self.window.opts['center'] = view_location
        # redraw
        self.app.processEvents()

    ###################################
    # private functions
    def rotatePoints(self, points, R):
        "Rotate points by the rotation matrix R"
        rotated_points = R @ points
        return rotated_points

    def translatePoints(self, points, translation):
        "Translate points by the vector translation"
        translated_points = points + np.dot(translation, np.ones([1,points.shape[1]]))
        return translated_points

    def getMAVPoints(self):
        """"
            Points that define the spacecraft, and the colors of the triangular mesh
            Define the points on the aircraft following diagram in Figure C.3
        """
        #points are in NED coordinates
        points = np.array([[3.5, 0.0, 0.0],     # point 1
                           [2.0, 1.0, -1.0],    # point 2
                           [2.0, -1.0, -1.0],   # point 3
                           [2.0, -1.0, 1.0],    # point 4
                           [2.0, 1.0, 1.0],     # point 5
                           [-7.0, 0.0, 0.0],    # point 6
                           [0.0, 5.0, 0.0],     # point 7
                           [-2.0, 5.0, 0.0],    # point 8
                           [-2.0, -5.0, 0.0],   # point 9
                           [0.0, -5.0, 0.0],    # point 10
                           [-5.0, 3.0, 0.0],    # point 11
                           [-7.0, 3.0, 0.0],    # point 12
                           [-7.0, -3.0, 0.0],   # point 13
                           [-5.0, -3.0,0.0],    # point 14
                           [-5.0, 0.0, 0.0],    # point 15
                           [-7.0, 0.0, -3.0]    # point 16
                           ]).T

        # scale points for better rendering
        scale = 5
        points = scale * points

        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        mesh_colors = np.empty((13, 3, 4), dtype=np.float32)
        mesh_colors[0] = yellow  # front
        mesh_colors[1] = yellow  # front
        mesh_colors[2] = yellow  # back
        mesh_colors[3] = yellow  # back
        mesh_colors[4] = blue  # right
        mesh_colors[5] = blue  # right
        mesh_colors[6] = blue  # left
        mesh_colors[7] = blue  # left
        mesh_colors[8] = green  # top
        mesh_colors[9] = green  # top
        mesh_colors[10] = red  # bottom
        mesh_colors[11] = red  # bottom
        mesh_colors[12] = yellow  # bottom
        return points, mesh_colors

    def pointsToMesh(self, points):
        """"
        Converts points to triangular mesh
        Each mesh face is defined by three 3D points
          (a rectangle requires two triangular mesh faces)
        """
        points=points.T
        mesh = np.array([[points[0], points[1], points[2]],  # front
                         [points[0], points[1], points[4]],  # front
                         [points[0], points[4], points[3]],  # back
                         [points[0], points[3], points[2]],  # back
                         [points[1], points[2], points[5]],  # right
                         [points[2], points[3], points[5]],  # right
                         [points[3], points[4], points[5]],  # left
                         [points[4], points[1], points[5]],  # left
                         [points[7], points[6], points[9]],  # top
                         [points[7], points[8], points[9]],  # top
                         [points[11], points[10], points[13]],  # bottom
                         [points[11], points[12], points[13]],  # bottom
                         [points[5], points[14], points[15]]  # bottom
                         ])
        return mesh

    def Euler2Rotation(self, phi, theta, psi):
        """
        Converts euler angles to rotation matrix (R_b^i, i.e., body to inertial)
        """
        # only call sin and cos once for each angle to speed up rendering
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)

        R_roll = np.array([[1, 0, 0],
                           [0, c_phi, s_phi],
                           [0, -s_phi, c_phi]])
        R_pitch = np.array([[c_theta, 0, -s_theta],
                            [0, 1, 0],
                            [s_theta, 0, c_theta]])
        R_yaw = np.array([[c_psi, s_psi, 0],
                          [-s_psi, c_psi, 0],
                          [0, 0, 1]])

        R = R_roll @ R_pitch @ R_yaw # inertial to body (Equation 2.4 in book)
        return R.T  # transpose to return body to inertial

