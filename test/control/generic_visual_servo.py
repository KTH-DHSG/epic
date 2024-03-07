import numpy as np
import casadi as ca
import unittest

from random import randint
import time

import sympy as sp
from sympy import lambdify
from epic.utils.geometry import r_mat, xi_mat

import matplotlib.pyplot as plt


unittest.TestLoader.sortTestMethodsUsing = lambda _, x, y: randint(-1, 1)


class TestMPRVS(unittest.TestCase):
    def setUp(self):
        # testing flags:
        self.generic_model = True  # Perspective or generic model
        self.use_negative_Z_points = True
        self.use_Hc = False
        self.use_raw_model = False
        self.vs_gain = 10

        # Simulation parameters
        self.dt = 0.01
        sim_length = 1.0
        self.sim_steps = int(sim_length / self.dt)
        self.online_plot = True
        self.wait_for_input = False

        # Debug conditions:
        self.start_at = 2  # 0: start at target, 1: start at translation error, 2: start at random pose

        # Set world features
        if self.use_negative_Z_points:
            self.w_points_h = np.array(
                [
                    [2, 3, 4, 9, -10],
                    [0, -1, 5, -3, 7],
                    [-5, -10, -2, -7, -7],
                    [1, 1, 1, 1, 1],
                ]
            )
        else:
            self.w_points_h = np.array(
                [
                    [2, 3, 4, 9, -5],
                    [0, -1, 5, -3, 7],
                    [5, 10, 2, 7, 15],
                    [1, 1, 1, 1, 1],
                ]
            )

        self.build_caron_model()
        self.buid_barreto_model()
        self.build_model_raw()
        self.set_integrator()

        # Define camera properties
        if self.generic_model:
            self.csi = 1.0
            self.psi = 0.0
        else:
            self.csi = 0.0
            self.psi = 1.0

    def build_caron_model(self):
        """
        Build the models for the tests.
        """
        # Define xs and y
        r = sp.symbols("rho")
        xs = sp.symbols("x")
        ys = sp.symbols("y")

        # Define camera properties
        csi = sp.symbols("xi")

        # ups = sp.symbols("alpha")
        ups = sp.sqrt(1 + (1 - csi**2) * (xs**2 + ys**2))

        # Matrix entries
        l11 = -(1 + xs**2 * (1 - csi * (ups + csi)) + ys**2) / (r * (ups + csi))
        l12 = xs * ys * csi / r
        l13 = xs * ups / r
        l14 = xs * ys
        l15 = -((1 + xs**2) * ups - ys**2 * csi) / (ups + csi)
        l16 = ys

        l21 = xs * ys * csi / r
        l22 = -(1 + xs**2 + ys**2 * (1 - csi * (ups + csi))) / (r * (ups + csi))
        l23 = ys * ups / r
        l24 = ((1 + ys**2) * ups - xs**2 * csi) / (ups + csi)
        l25 = -xs * ys
        l26 = -xs

        caron_L = sp.Matrix(
            [[l11, l12, l13, l14, l15, l16], [l21, l22, l23, l24, l25, l26]]
        )

        # Lambdify caron's model
        if self.use_Hc:
            psi = sp.symbols("psi")
            caron_L[0, :] = caron_L[0, :] * (psi - csi)
            caron_L[1, :] = caron_L[1, :] * (csi - psi)
            self.img_interaction_matrix = lambdify((xs, ys, r, csi, psi), caron_L)
        else:
            self.img_interaction_matrix = lambdify((xs, ys, r, csi), caron_L)

    def buid_barreto_model(self):
        # Define xs and y
        r = sp.symbols("rho")
        xs = sp.symbols("x")
        ys = sp.symbols("y")

        # Define camera properties
        csi = sp.symbols("xi")

        # ups = sp.symbols("Upsilon")
        ups = sp.sqrt(1 + (1 - csi**2) * (xs**2 + ys**2))

        # Matrix entries
        l11 = xs * ys
        l12 = ((1 + xs**2) * ups - ys**2 * csi) / (ups + csi)
        l13 = ys
        l14 = (1 + xs**2 * (1 - csi * (ups + csi)) + ys**2) / (r * (ups + csi))
        l15 = xs * ys * csi / r
        l16 = -xs * ups / r

        l21 = ((1 + ys**2) * ups - xs**2 * csi) / (ups + csi)
        l22 = xs * ys
        l23 = -xs
        l24 = -xs * ys * csi / r
        l25 = (1 + xs**2 + ys**2 * (1 - csi * (ups + csi))) / (r * (ups + csi))
        l26 = -ys * ups / r

        # Note that barreto has v and w swapped, so entries need to be swapped
        barreto_L = sp.Matrix(
            [[l14, l15, l16, l11, l12, l13], [l24, l25, l26, l21, l22, l23]]
        )

        # Lambdify barreto's model
        self.barreto_interaction_matrix = lambdify((xs, ys, r, csi), barreto_L)

    def build_model_raw(self):
        # Start by building and lambdifying our model
        t = sp.symbols("t")
        X = sp.Function("X")
        Y = sp.Function("Y")
        Z = sp.Function("Z")

        # Define camera properties
        csi = sp.symbols("xi")
        psi = sp.symbols("psi")

        # Define Hc matrix
        Mc = sp.Matrix.diag([psi - csi, csi - psi, 1])
        Rc = sp.Matrix.diag([1, 1, 1])  # No mirror rotation, canonical
        Hc = Rc @ Mc
        self.Hc = lambdify((csi, psi), Hc)

        # Auxiliary variables
        rho = sp.sqrt(X(t) ** 2 + Y(t) ** 2 + Z(t) ** 2)

        # Imaging model
        Xs = X(t) / rho
        Ys = Y(t) / rho
        Zs = Z(t) / rho + csi
        img = sp.Matrix([[Xs / Zs, Ys / Zs, Zs / Zs]]).T

        # Apply Hc to the image
        if self.use_Hc:
            img = Hc @ img

        # Take image derivative
        dimg_dt = sp.diff(img, t)

        # Points motion wrt camera input
        vx, vy, vz, wx, wy, wz = sp.symbols("v_x, v_y, v_z, w_x, w_y, w_z")
        vc = sp.Matrix([[vx, vy, vz]]).T
        wc = sp.Matrix([[wx, wy, wz]]).T
        Pw = sp.Matrix([[X(t), Y(t), Z(t)]]).T
        Pw_dot = -vc - wc.cross(Pw)

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
        if self.use_Hc:
            self.interaction_matrix = lambdify((X(t), Y(t), Z(t), csi, psi), L)
        else:
            self.interaction_matrix = lambdify((X(t), Y(t), Z(t), csi), L)

    @unittest.skip("Ensuring that matrices match first.")
    def test_interaction_matrices(self):
        """
        Basic test for matching interaction matrices.
        """
        # Define one world point
        self.w_points_h = self.w_points_h[:, 0].reshape((4, 1))

        # Get image points
        camera_state = np.array([1.0, 2.0, 3.0, 0, 0, 0, 1])
        img_points, ray_incamera, depth = self.get_image_points(camera_state, self.csi)

        # Get interaction matrices
        if self.use_Hc:
            ours = self.interaction_matrix(
                ray_incamera[0], ray_incamera[1], ray_incamera[2], self.csi, self.psi
            ).reshape((2, 6))
            caron_img = self.img_interaction_matrix(
                img_points[0], img_points[1], depth, self.csi, self.psi
            ).reshape((2, 6))
            print("Ours: \n", ours)
            print("Caron's: \n", caron_img)
            print("Difference: \nO/C: ", np.linalg.norm(ours - caron_img))
        else:
            ours = self.interaction_matrix(
                ray_incamera[0], ray_incamera[1], ray_incamera[2], self.csi
            ).reshape((2, 6))
            caron_img = self.img_interaction_matrix(
                img_points[0], img_points[1], depth, self.csi
            ).reshape((2, 6))
            barreto_img = self.barreto_interaction_matrix(
                img_points[0], img_points[1], depth, self.csi
            ).reshape((2, 6))
            print("Ours: \n", ours)
            print("Caron's: \n", caron_img)
            print("Barreto's: \n", barreto_img)
            print(
                "Difference: \nO/C: ",
                np.linalg.norm(ours - caron_img),
                "       O/B:",
                np.linalg.norm(ours - barreto_img),
            )

    def imaging_model(self, X_vector, Y_vector, Z_vector, csi, psi=None):
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
        points = np.empty((3, 0))
        depths = np.empty((1, 0))

        for i in range(X_vector.shape[0]):
            # Get point
            X = X_vector[i]
            Y = Y_vector[i]
            Z = Z_vector[i]

            # Define x and y
            rho = np.sqrt(X**2 + Y**2 + Z**2)
            x = X / (Z + csi * rho)

            # Negated y coordinate for barreto's model
            y = Y / (Z + csi * rho)

            point = np.array([[x, y, 1]]).T

            if self.use_Hc and psi is not None:
                point = self.Hc(csi, psi) @ point

            # Append
            points = np.append(points, point, axis=1)
            depths = np.append(depths, rho)

        return points, depths

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

    def get_image_points(self, cam_state, csi, psi=None):
        """
        Obtain points in the image plane.
        """
        # Get transformation matrix
        T_c_W = self.get_camera_transformation_matrix(cam_state)

        # Get all points in camera frame
        c_points_h = np.dot(T_c_W, self.w_points_h)

        # Get the image points
        image_points, depths = self.imaging_model(
            c_points_h[0, :], c_points_h[1, :], c_points_h[2, :], csi, psi
        )
        return image_points, c_points_h, depths

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

    def set_integrator(self):
        """
        Set casadi integrator for camera kinematics.
        """
        x = ca.MX.sym("x", 7, 1)  # [x, q]
        u = ca.MX.sym("u", 6, 1)  # [v, w]
        dae = {
            "x": x,
            "ode": self.camera_kinematics(x, u),
            "p": ca.vertcat(u),
        }
        options = {
            "tf": self.dt,
        }
        self.camera_motion_integrator = ca.integrator(
            "integrator", "cvodes", dae, options
        )

        return

    def discrete_dynamics(self, x, u):
        """
        Discrete dynamics for the camera.
        """
        # call casadi integrator
        next_state = self.camera_motion_integrator(x0=x, p=ca.vertcat(u))["xf"]

        # normalize quaternion
        next_state[3:] = next_state[3:] / np.linalg.norm(next_state[3:])

        return next_state

    def visual_servo_law(self, cam_state, target_image_points, target_depths):
        # Get image points
        image_points, rays, depths = self.get_image_points(cam_state, self.csi)

        # Get error
        image_error = image_points - target_image_points

        # Get interaction matrix
        L = np.empty((0, 6))
        for i in range(image_points.shape[1]):
            # A few selections are done here
            if self.use_raw_model:
                if self.use_Hc:
                    Li = self.interaction_matrix(
                        rays[0, i],
                        rays[1, i],
                        rays[2, i],
                        self.csi,
                        self.psi,
                    )
                else:
                    Li = self.interaction_matrix(
                        rays[0, i], rays[1, i], rays[2, i], self.csi
                    )
            else:
                if self.use_Hc:
                    Li = self.img_interaction_matrix(
                        image_points[0, i],
                        image_points[1, i],
                        depths[i],
                        self.csi,
                        self.psi,
                    )
                else:
                    Li = self.img_interaction_matrix(
                        image_points[0, i], image_points[1, i], depths[i], self.csi
                    )
            L = np.append(L, Li, axis=0)

        # Take first two components of the error and stack it
        error = image_error[0:2, :].reshape((-1, 1), order="F")

        # Get pseudo inverse
        L_pinv = np.linalg.pinv(L)

        # Get velocity
        velocity = -self.vs_gain * L_pinv @ error

        return velocity, image_points, image_error

    # @unittest.skip("Ensuring that matrices match first.")
    def test_visual_servoing(self):
        """
        Test that the moving leaders setpoint is set correctly.
        """

        # Define target camera pose
        camera_target = np.array([0.0, 0.0, 0.0, 0, 0, 0, 1])
        target_image_points, _, depths = self.get_image_points(camera_target, self.csi)

        # Define starting pose
        if self.start_at == 0:
            camera_state = camera_target
        elif self.start_at == 2:
            camera_state = np.array([0.1, 0.1, 0.1, 0.345, 0.111, 0.245, 0.972])
            camera_state[3:] = camera_state[3:] / np.linalg.norm(camera_state[3:])
        elif self.start_at == 1:
            camera_state = camera_target
            camera_state[0] += 0.1
        else:
            raise ValueError("Invalid start_at value.")

        fig = plt.figure()
        ax = fig.add_subplot(141)
        ax2 = fig.add_subplot(142)
        ax3 = fig.add_subplot(143)
        ax4 = fig.add_subplot(144)

        plt.ion()

        # Do servoing over a few steps:
        clctd_points = np.empty((3, self.w_points_h.shape[1], 0))
        cltcd_error = np.empty((3, self.w_points_h.shape[1], 0))
        cltcd_error_norm = np.empty((1, 0))
        cltcd_error_2d = np.empty((2 * self.w_points_h.shape[1], 0))
        cltcd_input = np.empty((6, 0))
        for ts in range(self.sim_steps):
            velocity, img_pts, error = self.visual_servo_law(
                camera_state, target_image_points, depths
            )

            # Move camera and get image points
            camera_state = self.discrete_dynamics(camera_state, velocity)

            # Update
            print("State: ", camera_state)

            # Collect points
            clctd_points = np.append(
                clctd_points, img_pts.reshape((3, self.w_points_h.shape[1], 1)), axis=2
            )
            cltcd_error = np.append(
                cltcd_error, error.reshape((3, self.w_points_h.shape[1], 1)), axis=2
            )
            cltcd_error_2d = np.append(
                cltcd_error_2d, error[0:2, :].reshape((-1, 1), order="F"), axis=1
            )
            cltcd_error_norm = np.append(
                cltcd_error_norm, np.linalg.norm(np.linalg.norm(error, axis=0))
            )
            cltcd_input = np.append(cltcd_input, velocity.reshape((6, 1)), axis=1)

            # Online Plot
            if self.online_plot:
                # Clear
                ax.clear()
                ax2.clear()
                ax3.clear()
                ax4.clear()

                # Plot
                ax.scatter(clctd_points[0, :, -1], clctd_points[1, :, -1], c="r")
                ax.scatter(target_image_points[0, :], target_image_points[1, :], c="b")
                ax.set_aspect("equal")

                ax2.plot(
                    self.dt * np.array(range(cltcd_error.shape[2])),
                    cltcd_error_norm,
                    c="k",
                )
                # plot zero
                ax2.plot(
                    self.dt * np.array(range(cltcd_error.shape[2])),
                    np.zeros((cltcd_error.shape[2], 1)),
                    "--k",
                )
                ax3.plot(
                    self.dt * np.array(range(cltcd_error.shape[2])),
                    cltcd_error_2d.T
                )
                ax4.plot(
                    self.dt * np.array(range(cltcd_input.shape[1])),
                    cltcd_input.T
                )

                # Draw
                plt.draw()
                plt.pause(0.001)

                if self.wait_for_input:
                    input("Press Enter to continue...")


if __name__.__contains__("__main__"):
    unittest.main()
