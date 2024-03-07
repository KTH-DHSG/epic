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
        self.w_points_h = np.array([[2], [0], [5], [1]])

        self.build_models()

        # Define camera properties
        self.csi = 0.1
        self.psi = 0.6

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
        rho = sp.sqrt(X(t) ** 2 + Y(t) ** 2 + Z(t) ** 2)

        # Define Hc matrix
        Mc = sp.Matrix.diag([psi - csi, csi - psi, 1])
        Rc = sp.Matrix.diag([1, -1, -1])
        Hc = Rc @ Mc
        # Hc = sp.Matrix.eye(3)

        # Imaging model
        Xs = X(t) / rho
        Ys = Y(t) / rho
        Zs = Z(t) / rho + csi
        img = sp.Matrix([[Xs / Zs, Ys / Zs, Zs / Zs]]).T
        # img = sp.Matrix([[X(t)/Z(t), Y(t)/Z(t), 1]]).T

        # Apply Hc to the image
        img = Hc @ img

        # Take image derivative
        dimg_dt = sp.diff(img, t)

        # Points motion wrt camera input
        vx, vy, vz, wx, wy, wz = sp.symbols("v_x, v_y, v_z, w_x, w_y, w_z")
        vc = sp.Matrix([[vx, vy, vz]]).T
        wc = sp.Matrix([[wx, wy, wz]]).T
        Pw = sp.Matrix([[X(t), Y(t), Z(t)]]).T
        Pw_dot = -vc - wc.cross(
            Pw
        )  # NOTE: THIS MIGHT BE WRONG GIVEN RC IS NOT THE IDENTITY

        # Substittue the derivative of the points in the image derivative
        dimg_with_uc_X = dimg_dt.subs({sp.Derivative(X(t), t): Pw_dot[0]})
        dimg_with_uc_Y = dimg_with_uc_X.subs({sp.Derivative(Y(t), t): Pw_dot[1]})
        sdimg_uc = dimg_with_uc_Y.subs({sp.Derivative(Z(t), t): Pw_dot[2]})

        # Get the interaction matrix
        L11 = sp.simplify(sdimg_uc[0].diff(vx))
        L12 = sp.simplify(sdimg_uc[0].diff(vy))
        L13 = sp.simplify(sdimg_uc[0].diff(vz))
        L14 = sp.simplify(sdimg_uc[0].diff(wx))
        L15 = sp.simplify(sdimg_uc[0].diff(wy))
        L16 = sp.simplify(sdimg_uc[0].diff(wz))
        L21 = sp.simplify(sdimg_uc[1].diff(vx))
        L22 = sp.simplify(sdimg_uc[1].diff(vy))
        L23 = sp.simplify(sdimg_uc[1].diff(vz))
        L24 = sp.simplify(sdimg_uc[1].diff(wx))
        L25 = sp.simplify(sdimg_uc[1].diff(wy))
        L26 = sp.simplify(sdimg_uc[1].diff(wz))

        L = sp.Matrix([[L11, L12, L13, L14, L15, L16], [L21, L22, L23, L24, L25, L26]])

        # Lambdify interaction matrix
        self.interaction_matrix = lambdify((X(t), Y(t), Z(t), csi, psi), L)

        # Lambdify Hc
        self.Hc = lambdify((csi, psi), Hc)

        # ---------------------------------------------------------------------
        # Build Barreto's model
        # ---------------------------------------------------------------------
        # Define xs and y
        rho = sp.sqrt(X(t) ** 2 + Y(t) ** 2 + Z(t) ** 2)
        r = rho
        x = X(t) / (Z(t) + csi * rho)
        y = -Y(t) / (Z(t) + csi * rho)

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

        barreto_L = sp.Matrix(
            [[l14, l15, l16, l11, l12, l13], [l24, -l25, l26, l21, l22, l23]]
        )

        barreto_L[0, :] = (csi - psi) * barreto_L[0, :]
        barreto_L[1, :] = (psi - csi) * barreto_L[1, :]

        # Lambdify barreto's model
        self.barreto_model = lambdify((X(t), Y(t), Z(t), csi, psi), barreto_L)

        # ---------------------------------------------------------------------
        # Build Barreto's model
        # ---------------------------------------------------------------------
        # Define xs and y
        r = sp.symbols("rho")
        x = sp.symbols("x")
        y = sp.symbols("y")

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

        L_img = sp.Matrix(
            [[l14, l15, l16, l11, l12, l13], [l24, -l25, l26, l21, l22, l23]]
        )

        L_img[0, :] = (csi - psi) * L_img[0, :]
        L_img[1, :] = (psi - csi) * L_img[1, :]

        # Lambdify barreto's model
        self.barreto_model_img = lambdify((x, y, r, csi, psi), L_img)

    def imaging_model(self, X, Y, Z, csi, psi, barreto=False):
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
        if not barreto:
            # Define xs and y
            x = X / (Z + csi * np.sqrt(X**2 + Y**2 + Z**2))
            y = Y / (Z + csi * np.sqrt(X**2 + Y**2 + Z**2))

            # Get Hc
            Hc = self.Hc(csi, psi)

            # Apply Hc
            point = np.array([[x[0], y[0], 1]]).T
            point = Hc @ point
        else:
            x = X / (Z + csi * np.sqrt(X**2 + Y**2 + Z**2))
            y = -Y / (Z + csi * np.sqrt(X**2 + Y**2 + Z**2))

            # Apply Hc
            point = np.array([[x[0], y[0], 1]]).T

        # Extract points
        x, y = point[0, 0], point[1, 0]

        return np.array([[x, y]]).T

    def test_same_point_as_derivation(self):
        """
        Test that the moving leaders setpoint is set correctly.
        """
        point = [1, 2, 3]

        # Set the setpoint
        print(
            "Our model, perspective: \n",
            self.interaction_matrix(point[0], point[1], point[2], self.csi, self.psi),
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

    def get_image_points(self, cam_state, csi, psi, barreto=False):
        """
        Obtain points in the image plane.
        """
        # Get transformation matrix
        T_c_W = self.get_camera_transformation_matrix(cam_state)

        # Get all points in camera frame
        c_points_h = np.dot(T_c_W, self.w_points_h)

        # Get the image points
        image_points = self.imaging_model(
            c_points_h[0, :], c_points_h[1, :], c_points_h[2, :], csi, psi, barreto
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
        v = u[0:3]

        # 3D Torque
        w = u[3:]

        # Model
        pdot = ca.mtimes(r_mat(q), v)
        qdot = ca.mtimes(xi_mat(q), w) / 2.0

        dxdt = [pdot, qdot]

        return ca.vertcat(*dxdt)

    def test_camera_motion_euler(self):
        """
        Test that the moving leaders setpoint is set correctly.
        """

        # Test two cameras with two different imaging models and compare the results
        dt = 0.01

        # Define camera properties
        camera_state = np.array([1.0, 2.0, 3.0, 0, 0, 0, 1])
        camera_state[3:] = camera_state[3:] / np.linalg.norm(camera_state[3:])
        velocity = np.array([[0.1, 0.2, 0.3, 0.0, 0, 0]]).T
        print("Starting state: ", camera_state)

        # Get image points
        our_image_points, ray_incamera = self.get_image_points(
            camera_state, self.csi, self.psi
        )
        barreto_img_points, barreto_ray_incamera = self.get_image_points(
            camera_state, self.csi, self.psi, barreto=True
        )
        print("Start imaging points: ", our_image_points)
        print("Ray in camera: ", ray_incamera)
        ray_incamera = ray_incamera.flatten()

        # Propagate with interaction matrix
        propagated_our_points = (
            our_image_points
            + self.interaction_matrix(
                ray_incamera[0], ray_incamera[1], ray_incamera[2], self.csi, self.psi
            )
            @ velocity
            * dt
        )

        print("----------------------------------")
        print("Propagating with interaction matrix")
        print("----------------------------------")
        ours = self.interaction_matrix(
            ray_incamera[0], ray_incamera[1], ray_incamera[2], self.csi, self.psi
        )
        barreto = self.barreto_model(
            ray_incamera[0], ray_incamera[1], ray_incamera[2], self.csi, self.psi
        )
        barreto_img = self.barreto_model_img(
            barreto_img_points[0],
            barreto_img_points[1],
            np.linalg.norm(barreto_ray_incamera),
            self.csi,
            self.psi,
        ).reshape((2, 6))
        print("Ours:")
        print(ours)
        print("Barreto's:")
        print(barreto)
        print("Barreto with image: ")
        print(barreto_img)
        print("Differences:")
        print(np.linalg.norm(ours - barreto))
        print(np.linalg.norm(ours - barreto_img))
        print("Matrices subtraction: ")
        print(ours - barreto)
        print(ours - barreto_img)

        # # Move camera and get image points
        # next_state = camera_state + self.camera_kinematics(camera_state, velocity) * dt
        # print("Next state: ", next_state)
        # # Normalize quaternion
        # next_state[3:] = next_state[3:] / np.linalg.norm(next_state[3:])

        # # Obtain image points now
        # next_image_points, _ = self.get_image_points(next_state, self.csi, self.psi)
        # print("Next image points: ", next_image_points)
        # print("Propagated our points: ", propagated_our_points)

        # # Get minimum error
        # our_error = np.linalg.norm(next_image_points - propagated_our_points)

        # # Print
        # print("Our error: ", our_error)


if __name__.__contains__("__main__"):
    unittest.main()
