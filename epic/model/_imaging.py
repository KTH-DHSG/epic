import casadi as ca
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

from epic.utils.geometry import r_mat
from epic.utils.veronese import lift_mat_to_veronese


class Imaging:
    """
    This class follows the implementation in
    1) "Geometric Properties of Central Catadioptric Line Images and Their Application in Calibration",
    by João P. Barreto and Helder Araujo, in IEEE Transactions on Pattern Analysis and Machine Intelligence,  VOL. 27, NO. 8, AUGUST 2005
    2) "Epipolar Geometry of Central Projection Systems Using Veronese Maps" by João P. Barreto and Kostas Daniilidis,
    in 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06)
    """

    def set_intrinsics(self, K=None):
        """
        Set intrinsics matrix.

        :param K: intrinsics matrix
        :type K: numpy array 3x3
        """

        # Consider same camera for all camera types
        if K is None:
            self.K = np.eye(3)
            self.K[0, 0] = 758.595136
            self.K[0, 2] = 305.937566
            self.K[1, 1] = 762.985127
            self.K[1, 2] = 220.271156
        else:
            self.K = K

        # Set resolution based on camera center
        self.K_inv = np.linalg.inv(self.K)
        self.res_x = self.K[0, 2] * 2.0
        self.res_y = self.K[1, 2] * 2.0

        # Set camera csi with d and p
        # d Is the Distance between Foci and 4p Is the Latus Rectum
        print("Camera type: ", self.type)
        if self.type == "perspective":
            self.csi = 0

        elif self.type == "hyperbolic":
            self.csi = self.cam_d / (np.sqrt(self.cam_d**2 + 4 * self.cam_p**2))
            print("csi: ", self.csi)
            self.csi = 0.95
        elif self.type == "parabolic":
            self.csi = 1.0

        elif self.type == "distortion":
            self.csi = -0.2

        else:
            raise ValueError("Camera type not supported")

        # Create Hc matrix - let it be identity, i.e., assume calibrated camera
        self.Rc = np.eye(3)
        self.Mc = np.eye(3)
        self.Hc = self.K @ self.Rc @ self.Mc
        self.Hc_inv = np.linalg.inv(self.Hc)
        self.Hc_inv_lifted = lift_mat_to_veronese(np.linalg.inv(self.Hc))

        # Create calibrated Hc matrix
        self.Hc_n = np.eye(3)
        self.Hc_n_inv = np.eye(3)
        self.Hc_n_inv_lifted = lift_mat_to_veronese(np.linalg.inv(self.Hc_n))

        # Create augment K matrix
        self.K_inv_aug = np.zeros((6, 6))
        self.K_inv_aug[3:, 3:] = self.K_inv

        # Create calibrated K matrices
        self.K_n = np.eye(3)
        self.K_n_inv_aug = np.zeros((6, 6))
        self.K_n_inv_aug[3:, 3:] = np.eye(3)

        # Set mirror parameters for lifted coordinates
        if self.type == "hyperbolic":
            self.Delta_c = np.eye(6)
            self.Delta_c[0, 0] = self.Delta_c[1, 1] = self.Delta_c[2, 2] = (
                1 - self.csi**2
            )
            self.Delta_c[0, 5] = self.Delta_c[2, 5] = -self.csi**2

        elif self.type == "parabolic":
            c1 = np.diag([2, 2, 1])
            c2 = np.zeros((3, 6))
            c2[0, 3] = c2[1, 4] = c2[2, 5] = 1
            c2[2, 0] = c2[2, 2] = -1
            self.Theta = np.concatenate((np.zeros((3, 6)), np.dot(c1, c2)), axis=0).T

        elif self.type == "distortion":
            # TODO: Note, distortion model not yet available for servoing
            raise NotImplementedError("Not yet implemented")
            c1 = np.diag([2, 2, 1])
            c2 = np.zeros((3, 6))
            c2[0, 3] = c2[1, 4] = 0.5
            c2[2, 5] = 1
            c2[2, 0] = c2[2, 2] = self.csi
            self.Psi = np.concatenate((np.zeros((3, 6)), np.dot(c1, c2)), axis=0).T

    def get_K(self):
        """
        Returns the camera intrinsics matrix.

        :return: camera intrinsics K.
        :rtype: numpy array
        """
        return self.K

    def h(self, x, csi):
        """
        H mapping.

        :param x: point
        :type x: np.ndarray
        :param csi: distortion parameter
        :type csi: float
        :return: mapped point
        :rtype: np.ndarray
        """
        # Apply h mapping
        xp = np.array(
            [[x[0], x[1], x[2] + csi * np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)]]
        ).T
        return xp

    def h_inv(self, xp, csi):
        """
        Inverse H mapping.

        :param xp: point in image plane
        :type xp: np.ndarray
        :param csi: distortion paramater
        :type csi: float
        :return: image ray
        :rtype: np.ndarray
        """
        x1 = (
            (
                xp[2] * csi
                + np.sqrt(xp[2] ** 2 + (1 - csi**2) * (xp[0] ** 2 + xp[1] ** 2))
            )
            / (xp[0] ** 2 + xp[1] ** 2 + xp[2] ** 2)
        ) * xp[0]
        x2 = (
            (
                xp[2] * csi
                + np.sqrt(xp[2] ** 2 + (1 - csi**2) * (xp[0] ** 2 + xp[1] ** 2))
            )
            / (xp[0] ** 2 + xp[1] ** 2 + xp[2] ** 2)
        ) * xp[1]
        x3 = (
            (
                xp[2] * csi
                + np.sqrt(xp[2] ** 2 + (1 - csi**2) * (xp[0] ** 2 + xp[1] ** 2))
            )
            / (xp[0] ** 2 + xp[1] ** 2 + xp[2] ** 2)
        ) * xp[2] - csi

        return np.array([x1, x2, x3])

    def delta(self, x, csi):
        """
        Divison model for radial distortion cameras.

        :param x: ray
        :type x: np.ndarray
        :param csi: distortion parameter
        :type csi: float
        :return: image point
        :rtype: np.ndarray
        """
        raise NotImplementedError("Not yet implemented")

    def delta_inv(self, xp, csi):
        """
        Inverse division model for radial distortion cameras.

        :param xp: image point
        :type xp: np.ndarray
        :param csi: distortion parameter
        :type csi: float
        :return: image ray
        :rtype: np.ndarray
        """
        raise NotImplementedError("Not yet implemented")

    def general_projection_model(self, x):
        """
        Projects to the image plane the given point x.

        :param x: Homogeneous point to be projected
        :type x: np.ndarray
        :raises TypeError: camera type is invalid
        :return: projected point
        :rtype: np.ndarray
        """

        if self.type == "perspective":
            xp = self.K @ self.h(x, self.csi)
            up = self.h(x, self.csi)
        elif self.type == "hyperbolic" or self.type == "parabolic":
            xp = self.Hc @ self.h(x, self.csi)
            up = self.Hc_n @ self.h(x, self.csi)
        elif self.type == "distortion":
            xp = self.delta(self.K @ x, self.csi)
            up = self.delta(x, self.csi)
        else:
            raise ValueError("Camera type not supported")
        return xp, up

    def general_inverse_projection_model(self, xp):
        """
        Performs the inverse projection, back to the camera plane,
        of the given point x''.

        :param x: Homogeneous point to be projected
        :type x: np.ndarray
        :raises TypeError: wrong camera type
        :return: image-plane point
        :rtype: np.ndarray
        """
        if self.type == "perspective":
            x = self.h_inv(self.K_inv @ xp, self.csi)
        elif self.type == "hyperbolic" or self.type == "parabolic":
            x = self.h_inv(self.Hc_inv @ xp, self.csi)
        elif self.type == "distortion":
            x = self.K_inv @ self.delta_inv(xp, self.csi)
        else:
            raise ValueError("Camera type not supported")

        # x = x / np.linalg.norm(x)
        x = x / x[2]

        return x

    def general_calibrated_inverse_projection_model(self, xp):
        """
        Performs the inverse projection, back to the camera plane,
        of the given point x''.

        :param x: Homogeneous point to be projected
        :type x: np.ndarray
        :raises TypeError: wrong camera type
        :return: image-plane point
        :rtype: np.ndarray
        """
        if self.type == "perspective":
            x = self.h_inv(xp, self.csi)
        elif self.type == "hyperbolic" or self.type == "parabolic":
            x = self.h_inv(self.Hc_n_inv @ xp, self.csi)
        elif self.type == "distortion":
            x = self.delta_inv(xp, self.csi)
        else:
            raise ValueError("Camera type not supported")

        # x = x / np.linalg.norm(x)
        x = x / x[2]

        return x

    def get_camera_transformation_matrix(self, cam_state=None):
        """
        Get camera transformation matrix.

        :param cam_state: camera state, defaults to None
        :type cam_state: np.ndarray, optional
        :return: transformation matrix
        :rtype: np.ndarray
        """

        if cam_state is None:
            x = self.state
        else:
            x = cam_state

        # Get Transformation matrix - (6.6) in Multiple View Geometry in Computer Vision, 2nd Ed
        T_c_W = r_mat(x[3:]).T @ np.concatenate(
            (np.eye(3), -x[0:3].reshape((3, 1))), axis=1
        )
        return T_c_W

    def get_image_points(self, p_w_points_h, cam_state=None):
        """
        Get image points given camera state and points position.

        :param x: camera state in world coordinates
        :type x: ca.DM
        :param p_w_points_h: (4,N) world point-features, in homogenous coordinates
        :type p_w_points_h: ca.DM or np.array
        :return: image (pixels and normalized) points w/ depth (N*3, 1)
        :rtype: ca.DM
        """

        # Get transformation matrix
        T_c_W = self.get_camera_transformation_matrix(cam_state)

        # Get all points in camera frame
        c_points_h = np.dot(T_c_W, p_w_points_h)
        c_depths = np.linalg.norm(c_points_h, axis=0)  # rho of each feature

        # Apply unified projection model
        px_points_h = np.empty((3, 0))
        u_points_h = np.empty((3, 0))
        depth = np.array([]).reshape(1, 0)
        delta = np.array([]).reshape(1, 0)
        for i in range(c_points_h.shape[1]):
            p, u = self.general_projection_model(c_points_h[:, i])
            delta_i = u[2]
            depth_i = c_depths[i]

            # Normalize by last coordinate
            p = p / p[2]
            u = u / delta_i

            # Save the point
            px_points_h = np.hstack((px_points_h, p.reshape(3, 1)))  # Px points
            u_points_h = np.hstack((u_points_h, u.reshape(3, 1)))  # calibrated points
            depth = np.hstack((depth, depth_i.reshape((1, 1))))
            delta = np.hstack((delta, delta_i.reshape((1, 1))))

        return px_points_h, u_points_h, (depth, delta)

    def get_image_ray_from_calibrated_points(self, u_points, depth, camera=None):
        """
        Get image rays from normalized points.

        :param u_points: calibrated image points
        :type u_points: np.ndarray
        :param depth: depth of these points
        :type depth: np.ndarray
        """
        world_points_h = np.empty((3, 0))
        for i in range(u_points.shape[1]):
            delta = depth[1][0, i]
            if camera is None:
                ray = self.general_calibrated_inverse_projection_model(
                    u_points[:, i] * delta
                )
            else:
                ray = camera.general_calibrated_inverse_projection_model(
                    u_points[:, i] * delta
                )
            world_points_h = np.hstack((world_points_h, ray.reshape(3, 1)))

        return world_points_h

    def get_camera_frame_points(self, px_points_h, depth_in, cam_state=None):
        """
        Get image points given camera state and points position.

        :param x: camera state in world coordinates
        :type x: ca.DM
        :param p_w_points_h: (4,N) world point-features, in homogenous coordinates
        :type p_w_points_h: ca.DM or np.array
        :return: image (pixels and normalized) points w/ depth (N*3, 1)
        :rtype: ca.DM
        """
        raise NotImplementedError("Not yet implemented")
        T_c_W = self.get_camera_transformation_matrix(cam_state)
        T_c_W_h = np.concatenate((T_c_W, np.array([[0, 0, 0, 1]])), axis=0)
        T_c_W_inv = np.linalg.inv(T_c_W_h)

        # Create returnin structures
        world_points_h = np.empty((3, 0))
        depth = np.array([]).reshape(1, 0)
        for i in range(px_points_h.shape[1]):
            # Multiply with delta
            delta = depth_in[1][0, i]
            wp = self.general_inverse_projection_model(px_points_h[:, i] * delta)

            # Multiply with depth
            wp = wp * depth_in[0][0, i]

            # Multiply by inverse transformation matrix
            wp = np.concatenate((wp.reshape((3, 1)), np.array([[1]])), axis=0)
            wp = np.dot(T_c_W_inv, wp)

            # Save the point
            world_points_h = np.hstack((world_points_h, wp[:3, :].reshape(3, 1)))
            depth = np.hstack((depth, depth_in[0][0, i].reshape((1, 1))))

        return world_points_h, depth

    def get_visible_points(self, w_points, cam_state=None):
        """
        Get the points in the camera frame (pixel and normalized)
        that are visible given a cam_state.

        :param w_points: world points in homogenous format
        :type w_points: numpy array, 4xN
        :param cam_state: camera state, optional
        :type cam_state: numpy array
        :return: points (pixel and normalized), world points index
        :rtype: numpy array 4xN, 4xN, list
        """

        px, u, z = self.get_image_points(w_points, cam_state)

        vis_px_points = np.array([]).reshape(3, 0)
        vis_u_points = np.array([]).reshape(3, 0)
        vis_z = np.array([]).reshape(1, 0)
        vis_idx = []

        for i in range(px.shape[1]):
            # Grab index point)
            px_pt = px[:, i]

            u_pt = u[:, i]
            z_pt = z[0][0, i]

            if (
                px_pt[0] > 0
                and px_pt[0] <= self.res_x
                and px_pt[1] > 0
                and px_pt[1] <= self.res_y
                and z_pt > 0
            ):
                vis_px_points = np.hstack((vis_px_points, px_pt.reshape(3, 1)))
                vis_u_points = np.hstack((vis_u_points, u_pt.reshape(3, 1)))
                vis_z = np.hstack((vis_z, z_pt.reshape(1, 1)))
                vis_idx.append(i)

        return vis_px_points, vis_u_points, vis_idx
