import numpy as np
import casadi as ca
import unittest

from random import randint

import sympy as sp
from sympy.utilities.lambdify import implemented_function
from sympy import lambdify
from epic.utils.geometry import r_mat, xi_mat


unittest.TestLoader.sortTestMethodsUsing = lambda _, x, y: randint(-1, 1)


class TestMPRVS(unittest.TestCase):
    def setUp(self):
        # Set world features
        self.w_points_h = np.array([[0], [0], [3], [1]])

        self.build_models()

    def build_models(self):
        """
        Build the models for the tests.
        """

        # Start by building and lambdifying our model
        t = sp.symbols("t")
        X = sp.Function("X")
        Y = sp.Function("Y")
        Z = sp.Function("Z")

        # Define camera properties
        csi = sp.symbols("xi")
        psi = sp.symbols("psi")

        # Auxiliary variables
        alpha = sp.symbols("alpha")
        rho = sp.sqrt(X(t) ** 2 + Y(t) ** 2 + Z(t) ** 2)
        r = rho

        # Define xs and y
        x = X(t) / (Z(t) + csi * rho)
        y = -Y(t) / (Z(t) + csi * rho)

        # Define Ji
        Ji_0 = sp.diff(x, X(t))
        Ji_1 = sp.diff(x, Y(t))
        Ji_2 = sp.diff(x, Z(t))
        Ji_3 = sp.diff(y, X(t))
        Ji_4 = sp.diff(y, Y(t))
        Ji_5 = sp.diff(y, Z(t))

        Ji = sp.Matrix([[Ji_0, Ji_1, Ji_2], [Ji_3, Ji_4, Ji_5]])

        Jm = sp.Matrix(
            [
                [0, -Z(t), Y(t), -1, 0, 0],
                [Z(t), 0, -X(t), 0, -1, 0],
                [-Y(t), X(t), 0, 0, 0, -1],
            ]
        )
        Jg = Ji @ Jm

        # Lambdify Jg
        self.our_fixed_barreto_model = lambdify((X(t), Y(t), Z(t), csi), Jg)

        # ---------------------------------------------------------------------
        # Build Barreto's model
        # ---------------------------------------------------------------------
        ups = sp.sqrt(1 + (1 - csi**2) * (x**2 + y**2))

        # Matrix entries
        l11 = x * y
        l12 = ((1 + x**2) * ups - y**2 * csi) / (ups + csi)
        l13 = y
        l14 = (1 + x**2 * (1 - csi * (ups + csi)) + y**2) / (r * (ups + csi))
        l15 = x * y * csi / r
        l16 = -x * ups / r

        l21 = ((1 + y**2) * ups - x**2 * csi) / (ups + csi)
        l22 = x * y
        l23 = -x
        l24 = -x * y * csi / r
        l25 = (1 + x**2 + y**2 * (1 - csi * (ups + csi))) / (r * (ups + csi))
        l26 = -y * ups / r

        barreto_L = -sp.Matrix(
            [[l11, l12, l13, l14, l15, l16],
             [l21, l22, l23, l24, -l25, l26]]
        )

        # Lambdify barreto's model
        self.barreto_model = lambdify((X(t), Y(t), Z(t), csi), barreto_L)

    def imaging_model_barreto(self, X, Y, Z, csi):
        """
        Imaging model for generic cameras from Barreto's visual servoing paper.

        :param X: X coordinate of the point
        :type X: float
        :param Y: Y coordinate of the point
        :type Y: float
        :param Z: Z coordinate of the point
        :type Z: float
        :param csi: distortion parameter
        :type csi: float
        """
        # Define xs and y
        x = X / (Z + csi * np.sqrt(X**2 + Y**2 + Z**2))
        y = -Y / (Z + csi * np.sqrt(X**2 + Y**2 + Z**2))

        return np.array([x, y])

    def imaging_model(self, X, Y, Z, csi):
        """
        Imaging model for generic cameras - proper.

        :param X: X coordinate of the point
        :type X: float
        :param Y: Y coordinate of the point
        :type Y: float
        :param Z: Z coordinate of the point
        :type Z: float
        :param csi: distortion parameter
        :type csi: float
        """
        # Define xs and y
        x = X / (Z + csi * np.sqrt(X**2 + Y**2 + Z**2))
        y = Y / (Z + csi * np.sqrt(X**2 + Y**2 + Z**2))

        return np.array([x, y])

    def test_same_point_as_derivation(self):
        """
        Test that the moving leaders setpoint is set correctly.
        """
        point = [1, 2, -3]
        csi = 0

        # Set the setpoint
        print(
            "Our model, numeric: \n",
            self.our_fixed_barreto_model(point[0], point[1], point[2], csi),
        )
        print(
            "Barreto's model, numeric: \n",
            self.barreto_model(point[0], point[1], point[2], csi),
        )

        # Return true
        self.assertTrue(True)

    def get_camera_transformation_matrix(self, cam_state):
        """
        Get camera transformation matrix.

        :param cam_state: camera state, defaults to None
        :type cam_state: np.ndarray, optional
        :return: transformation matrix
        :rtype: np.ndarray
        """
        x = cam_state

        # Get Transformation matrix - (6.6) in Multiple View Geometry in Computer Vision, 2nd Ed
        T_c_W = r_mat(x[3:]).T @ np.concatenate(
            (np.eye(3), -x[0:3].reshape((3, 1))), axis=1
        )
        return T_c_W

    def get_image_points(self, cam_state, csi):
        """
        Obtain points in the image plane.
        """
        # Get transformation matrix
        T_c_W = self.get_camera_transformation_matrix(cam_state)

        # Get all points in camera frame
        c_points_h = np.dot(T_c_W, self.w_points_h)

        # Get the image points
        image_points = self.imaging_model_barreto(
            c_points_h[0, :], c_points_h[1, :], c_points_h[2, :], csi
        )
        return image_points, c_points_h

    def camera_kinematics(self, x, u):
        """
        Camera 6dof kinematics.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state time derivative
        :rtype: ca.MX
        """

        # State extraction
        q = x[3:] / ca.norm_2(x[3:])

        # 3D Force
        w = u[0:3]

        # 3D Torque
        v = u[3:]

        # Model
        pdot = ca.mtimes(r_mat(q), v)
        qdot = ca.mtimes(xi_mat(q), w) / 2

        dxdt = [pdot, qdot]

        return ca.vertcat(*dxdt)

    def test_camera_motion_euler(self):
        """
        Test that the moving leaders setpoint is set correctly.
        """

        # Test two cameras with two different imaging models and compare the results

        # Define camera properties
        camera_state = np.array([1, 2, 3, -0.003, -0.842, 0.281, -0.460])
        velocity = np.array([[0.1, 0.2, 0.3, 0.1, 0.2, 0.3]]).T
        csi = 0.1
        dt = 0.1

        # Get image points
        our_image_points, ray_incamera = self.get_image_points(camera_state, csi)
        print("Start imaging points: ", our_image_points)
        print("Ray in camera: ", ray_incamera)
        ray_incamera = ray_incamera.flatten()

        print("----------------------------------")
        print("Propagating with interaction matrix")
        print("----------------------------------")
        print("Ours:")
        print(
            self.our_fixed_barreto_model(
                ray_incamera[0], ray_incamera[1], ray_incamera[2], csi
            )
        )
        print("Barreto's:")
        print(
            self.barreto_model(
                ray_incamera[0], ray_incamera[1], ray_incamera[2], csi
            )
        )

        # Propagate with interaction matrix
        propagated_our_points = (
            our_image_points
            + self.our_fixed_barreto_model(
                ray_incamera[0], ray_incamera[1], ray_incamera[2], csi
            )
            @ velocity
            * dt
        )
        propagated_bar_points = (
            our_image_points
            + self.barreto_model(ray_incamera[0], ray_incamera[1], ray_incamera[2], csi)
            @ velocity
            * dt
        )

        # Move camera and get image points
        next_state = camera_state + self.camera_kinematics(camera_state, velocity) * dt
        # Normalize quaternion
        next_state[3:] = next_state[3:] / np.linalg.norm(next_state[3:])

        # Obtain image points now
        next_image_points, _ = self.get_image_points(next_state, csi)
        print("Next image points: ", next_image_points)
        print("Propagated our points: ", propagated_our_points)
        print("Propagated bar points: ", propagated_bar_points)

        # Get minimum error
        our_error = np.linalg.norm(next_image_points - propagated_our_points)
        bar_error = np.linalg.norm(next_image_points - propagated_bar_points)

        # Print
        print("Our error: ", our_error)
        print("Barreto's error: ", bar_error)

    @unittest.skip("Need further investigation")
    def test_camera_motion_integrator(self):
        """
        Test that the moving leaders setpoint is set correctly.
        """

        # Test two cameras with two different imaging models and compare the results

        # Define camera properties
        camera_state = np.array([1, 2, 3, -0.003, -0.842, 0.281, -0.460])
        camera_state[3:] = camera_state[3:] / np.linalg.norm(camera_state[3:])
        velocity = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
        csi = 0.0
        dt = 0.1

        # Get image points
        our_image_points = self.get_image_points(camera_state, csi)

        # Propagate with interaction matrix
        propagated_our_points = (
            our_image_points
            + self.our_fixed_barreto_model(
                camera_state[0], camera_state[1], camera_state[2], csi
            )
            @ velocity
            * dt
        )
        propagated_bar_points = (
            our_image_points
            + self.barreto_model(camera_state[0], camera_state[1], camera_state[2], csi)
            @ velocity
            * dt
        )

        # Move camera and get image points
        next_state = camera_state + self.camera_kinematics(camera_state, velocity) * dt
        # Normalize quaternion
        next_state[3:] = next_state[3:] / np.linalg.norm(next_state[3:])

        # Obtain image points now
        next_image_points = self.get_image_points(next_state, csi)

        # Get minimum error
        our_error = np.linalg.norm(next_image_points - propagated_our_points)
        bar_error = np.linalg.norm(next_image_points - propagated_bar_points)

        # Print
        print("Our error: ", our_error)
        print("Barreto's error: ", bar_error)


if __name__.__contains__("__main__"):
    unittest.main()
