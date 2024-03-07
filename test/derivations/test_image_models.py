import numpy as np
import unittest

from epic.utils.geometry import get_relative_pose, skew, r_mat
from random import randint
from math import sqrt
from scipy.spatial.transform import Rotation as Rotations

unittest.TestLoader.sortTestMethodsUsing = lambda _, x, y: randint(-1, 1)


class TestMPRVS(unittest.TestCase):

    def setUp(self):
        # Set print options
        np.set_printoptions(precision=10)

        # Set world points
        self.w_points_h = np.array(
            [
                [0, 0.5, -0.5, 0, 0.5],
                [0, -0.5, 0.5, 0.5, 0.1],
                [5, 5, 5, 5, 5],
                [1, 1, 1, 1, 1]
            ]
        )

        # Leader control input
        self.leader_u = np.array([[0.0, 0.0, 0.0, 0, 0, 0]]).T
        pass

    def h(self, x, csi, rho):
        return np.array([x[0], x[1], x[2] + csi * rho])

    def h_inv(self, xp, csi):
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

    def get_image_with_h(self, x, csi):
        rho = np.linalg.norm(x)
        xp = self.h(x, csi, rho)
        xp = xp / xp[2]
        return xp

    @unittest.skip("Successful.")
    def test_direct_inv_h(self):
        p1 = np.array([0.1, 0.2, 0.3])
        depth = np.linalg.norm(p1)
        p1_s = p1 / depth
        # Get image point
        x = self.h(p1_s, 0.8)

        # Get inverse
        p1_inv = self.h_inv(x, 0.8)
        p1_inv = p1_inv * depth

        # Check if inverse is correct
        print("p1: ", p1.T)
        print("p1_inv: ", p1_inv.T)

        self.assertTrue(np.allclose(p1, p1_inv))

    def test_lifted_epipolar_condition_perspective_parabolic(self):
        K = np.eye(3)
        K_inv = np.eye(3)

        if False:
            leader = np.array([0.0, 0, 0, 0, 0, 0, 1])
            f1 = np.array([0.2, 0, 0, 0, 0, 0, 1])
        else:
            # Generate random pose
            leader_p = np.random.randn(3, 1)
            leader_q = np.random.randn(4, 1)
            leader_q = leader_q / np.linalg.norm(leader_q)

            f1_p = np.random.randn(3, 1)
            f1_q = np.random.randn(4, 1)
            f1_q = f1_q / np.linalg.norm(f1_q)

            leader = np.concatenate((leader_p, leader_q), axis=0).reshape((7,))
            f1 = np.concatenate((f1_p, f1_q), axis=0).reshape((7,))

        # Get relative pose
        R, t = get_relative_pose(leader, f1)

        # According to wikipedia this is correct
        # According to his paper, it is not (probably different frames)
        E = R @ skew(t)

        # Project points to each camera
        # Leader: Perspective
        csi1 = 0.0

        # Follower: parabolic
        csi2 = 1.0
        Hc = np.eye(3)
        Hc_inv = np.eye(3)
        # Print

        # Get projection matrices
        P1 = r_mat(leader[3:]).T @ np.concatenate(
            (np.eye(3), -leader[0:3].reshape((3, 1))), axis=1
        )
        P2 = r_mat(f1[3:]).T @ np.concatenate(
            (np.eye(3), -f1[0:3].reshape((3, 1))), axis=1
        )

        # Get points in leader camera
        x = self.w_points_h[:, :]
        xp_v = np.empty((3, 0))
        yp_v = np.empty((3, 0))

        # Save rays
        x_n_ray = np.empty((3, 0))
        y_n_ray = np.empty((3, 0))
        for i in range(self.w_points_h.shape[1]):
            x_m = P1 @ self.w_points_h[:, i]
            xp = self.get_image_with_h(x_m, csi1)

            y_m = P2 @ self.w_points_h[:, i]
            yp = self.get_image_with_h(y_m, csi2)

            if True:
                # Debug
                print("Epipolar right after projection: ", y_m.T @ E @ x_m)

            # Save projected points
            xp_v = np.append(xp_v, np.array([xp]).T, axis=1)
            yp_v = np.append(yp_v, np.array([yp]).T, axis=1)

            # Save rays
            x_n_ray = np.append(x_n_ray, np.array([x_m]).T, axis=1)
            y_n_ray = np.append(y_n_ray, np.array([y_m]).T, axis=1)

        # Check epipolar constraint
        E_v = np.empty((1, 0))
        x_n_reconstructed_ray = np.empty((3, 0))
        y_n_reconstructed_ray = np.empty((3, 0))
        for i in range(self.w_points_h.shape[1]):
            x = self.h_inv(Hc_inv @ xp_v[:, i], csi1)
            y = self.h_inv(Hc_inv @ yp_v[:, i], csi2)

            # x = x / np.linalg.norm(x)
            # y = y / np.linalg.norm(y)

            # x = x / x[2]
            # y = y / y[2]

            epi_val = y.T @ E @ x

            # Save epipolar condition value and reconstructed ray
            E_v = np.append(E_v, np.array([[epi_val]]), axis=1)
            x_n_reconstructed_ray = np.append(
                x_n_reconstructed_ray, np.array([x]).T, axis=1
            )
            y_n_reconstructed_ray = np.append(
                y_n_reconstructed_ray, np.array([y]).T, axis=1
            )

        # Check lifted epipolar constraint
        # Get veronese points
        x_lifted = self.lift_image_points(xp_v)
        y_lifted = self.lift_image_points(yp_v)

        H_lifted = self.lift_mat_to_veronese(Hc)
        H_inv_lifted = np.linalg.inv(H_lifted)
        Theta, _ = self.get_theta_and_psi_matrices(csi1, csi2)

        E_v_lifted = np.empty((1, 0))
        x_n_reconstructed_lifted_ray = np.empty((3, 0))
        y_n_reconstructed_lifted_ray = np.empty((3, 0))
        for i in range(self.w_points_h.shape[1]):
            # X is leader: perspective
            # Y is follower: parabolic
            x = xp_v[:, i]
            y = y_lifted[:, i].T @ Theta

            epi_val = y @ E @ x

            # Save lifted epipolar value and ray
            E_v_lifted = np.append(E_v_lifted, np.array([[epi_val]]), axis=1)
            x_n_reconstructed_lifted_ray = np.append(
                x_n_reconstructed_lifted_ray, np.array([x]).T, axis=1
            )
            y_n_reconstructed_lifted_ray = np.append(
                y_n_reconstructed_lifted_ray, np.array([y]).T, axis=1
            )

        # Check lifted matrices with augmentation
        Theta_aug, _ = self.get_theta_and_psi_aug_matrices(csi1, csi2)
        # Augment E
        V_E = np.zeros((6, 6))
        V_E[3:, 3:] = E

        # Augment K
        V_K = np.zeros((6, 6))
        V_K[3:, 3:] = K
        V_K_inv = np.zeros((6, 6))
        V_K_inv[3:, 3:] = K_inv

        E_v_lifted_aug = np.empty((1, 0))
        for i in range(self.w_points_h.shape[1]):
            x = x_lifted[:, i]
            y = y_lifted[:, i].T @ Theta_aug

            epi_val = y @ V_E @ x

            # Save lifted epipolar value and ray
            E_v_lifted_aug = np.append(E_v_lifted_aug, np.array([[epi_val]]), axis=1)

        # Check if epipolar constraint is met
        print("Epipolar constraint: ", np.round(E_v, 15))
        print("Lifted Epipolar constraint: ", np.round(E_v_lifted, 15))
        print("Lifted Epipolar constraint augmented: ", np.round(E_v_lifted_aug, 15))
        print(
            "[LEADER] Reconstruction error: ",
            np.linalg.norm(x_n_ray - x_n_reconstructed_ray),
        )
        print(
            "[LEADER] Reconstruction error lifted: ",
            np.linalg.norm(x_n_ray - x_n_reconstructed_lifted_ray),
        )
        print(
            "[F1] Reconstruction error: ",
            np.linalg.norm(y_n_ray - y_n_reconstructed_ray),
        )
        print(
            "[F1] Reconstruction error lifted: ",
            np.linalg.norm(
                y_n_ray - y_n_reconstructed_lifted_ray
            ),
        )

    @unittest.skip("Successful.")
    def test_lifted_epipolar_condition_keeping_curves(self):
        K = np.eye(3)
        K[0, 0] = 758.595136
        K[0, 2] = 305.937566
        K[1, 1] = 762.985127
        K[1, 2] = 220.271156
        K = np.eye(3)
        K_inv = np.linalg.inv(K)

        if False:
            leader = np.array([0.0, 0, 0, 0, 0, 0, 1])
            f1 = np.array([0.2, 0, 0, 0, 0, 0, 1])
        else:
            # Generate random pose
            leader_p = np.random.randn(3, 1)
            leader_q = np.random.randn(4, 1)
            leader_q = leader_q / np.linalg.norm(leader_q)

            f1_p = np.random.randn(3, 1)
            f1_q = np.random.randn(4, 1)
            f1_q = f1_q / np.linalg.norm(f1_q)

            leader = np.concatenate((leader_p, leader_q), axis=0).reshape((7, ))
            f1 = np.concatenate((f1_p, f1_q), axis=0).reshape((7, ))

        # Get relative pose
        R, t = get_relative_pose(leader, f1)

        # According to wikipedia this is correct
        # According to his paper, it is not (probably different frames)
        E = R @ skew(t)

        # Project points to each camera
        # Leader: parabolic
        csi1 = 1.0
        psi = 1.0 + 2 * 0.05
        Rc = Rotations.from_rotvec(np.pi * np.array([0, 1, 0])).as_matrix()
        # Rc = np.eye(3)
        Hc = K @ Rc @ np.diag([psi - csi1, psi - csi1, 1])
        Hc_inv = np.linalg.inv(Hc)
        Hc = np.eye(3)
        Hc_inv = np.eye(3)
        # Print

        # Follower: radial distortion
        csi2 = -0.2

        # Get projection matrices
        P1 = r_mat(leader[3:]).T @ np.concatenate((np.eye(3), -leader[0:3].reshape((3, 1))), axis=1)
        P2 = r_mat(f1[3:]).T @ np.concatenate((np.eye(3), -f1[0:3].reshape((3, 1))), axis=1)

        # Get points in leader camera
        x = self.w_points_h[:, :]
        xp_v = np.empty((3, 0))
        yp_v = np.empty((3, 0))

        # Save rays
        x_n_ray_parabolic = np.empty((3, 0))
        y_n_ray_distortion = np.empty((3, 0))
        for i in range(self.w_points_h.shape[1]):
            x_m = P1 @ self.w_points_h[:, i]
            x_m = x_m / np.linalg.norm(x_m)
            xp = Hc @ self.h(x_m, csi1)

            y_m = P2 @ self.w_points_h[:, i]
            y_m = y_m / np.linalg.norm(y_m)
            yp = self.delta(K @ y_m, csi2)

            if True:
                # Debug
                print("Epipolar right after projection: ", y_m.T @ E @ x_m)

            # Save projected points
            xp_v = np.append(xp_v, np.array([xp]).T, axis=1)
            yp_v = np.append(yp_v, np.array([yp]).T, axis=1)

            # Save rays
            x_n_ray_parabolic = np.append(x_n_ray_parabolic, np.array([x_m]).T, axis=1)
            y_n_ray_distortion = np.append(y_n_ray_distortion, np.array([y_m]).T, axis=1)

        # Check epipolar constraint
        E_v = np.empty((1, 0))
        x_n_reconstructed_ray_parabolic = np.empty((3, 0))
        y_n_reconstructed_ray_distortion = np.empty((3, 0))
        for i in range(self.w_points_h.shape[1]):
            x = self.h_inv(Hc_inv @ xp_v[:, i], csi1)
            y = K_inv @ self.delta_inv(yp_v[:, i], csi2)

            x = x / np.linalg.norm(x)
            y = y / np.linalg.norm(y)

            # x = x / x[2]
            # y = y / y[2]

            epi_val = y.T @ E @ x

            # Save epipolar condition value and reconstructed ray
            E_v = np.append(E_v, np.array([[epi_val]]), axis=1)
            x_n_reconstructed_ray_parabolic = np.append(x_n_reconstructed_ray_parabolic, np.array([x]).T, axis=1)
            y_n_reconstructed_ray_distortion = np.append(y_n_reconstructed_ray_distortion, np.array([y]).T, axis=1)

        # Check lifted epipolar constraint
        # Get veronese points
        x_lifted = self.lift_image_points(xp_v)
        y_lifted = self.lift_image_points(yp_v)

        H_lifted = self.lift_mat_to_veronese(Hc)
        H_inv_lifted = np.linalg.inv(H_lifted)
        Theta, Psi = self.get_theta_and_psi_matrices(csi1, csi2)

        E_v_lifted = np.empty((1, 0))
        x_n_reconstructed_lifted_ray_parabolic = np.empty((3, 0))
        y_n_reconstructed_lifted_ray_distortion = np.empty((3, 0))
        for i in range(self.w_points_h.shape[1]):
            x = Theta.T @ H_inv_lifted @ x_lifted[:, i]
            y = y_lifted[:, i].T @ Psi @ K_inv.T

            # x = x / np.linalg.norm(x)
            # y = y / np.linalg.norm(y)

            x = x / x[2]
            y = y / y[2]

            epi_val = y @ E @ x

            # Save lifted epipolar value and ray
            E_v_lifted = np.append(E_v_lifted, np.array([[epi_val]]), axis=1)
            x_n_reconstructed_lifted_ray_parabolic = np.append(
                x_n_reconstructed_lifted_ray_parabolic, np.array([x]).T, axis=1)
            y_n_reconstructed_lifted_ray_distortion = np.append(
                y_n_reconstructed_lifted_ray_distortion, np.array([y]).T, axis=1)

        # Check lifted matrices with augmentation
        Theta_aug, Psi_aug = self.get_theta_and_psi_aug_matrices(csi1, csi2)
        # Augment E
        V_E = np.zeros((6, 6))
        V_E[3:, 3:] = E

        # Augment K
        V_K = np.zeros((6, 6))
        V_K[3:, 3:] = K
        V_K_inv = np.zeros((6, 6))
        V_K_inv[3:, 3:] = K_inv

        E_v_lifted_aug = np.empty((1, 0))
        for i in range(self.w_points_h.shape[1]):
            x = Theta_aug.T @ H_inv_lifted @ x_lifted[:, i]
            y = y_lifted[:, i].T @ Psi_aug @ V_K_inv.T

            x = x / np.linalg.norm(x)
            y = y / np.linalg.norm(y)

            x = x / x[-1]
            y = y / y[-1]

            epi_val = y @ V_E @ x

            # Save lifted epipolar value and ray
            E_v_lifted_aug = np.append(E_v_lifted_aug, np.array([[epi_val]]), axis=1)

        # Check if epipolar constraint is met
        print("Epipolar constraint: ", np.round(E_v, 15))
        print("Lifted Epipolar constraint: ", np.round(E_v_lifted, 15))
        print("Lifted Epipolar constraint augmented: ", np.round(E_v_lifted_aug, 15))
        print("[LEADER] Reconstruction error: ", np.linalg.norm(x_n_ray_parabolic - x_n_reconstructed_ray_parabolic))
        print("[LEADER] Reconstruction error lifted: ",
              np.linalg.norm(x_n_ray_parabolic - x_n_reconstructed_lifted_ray_parabolic))
        print("[F1] Reconstruction error: ", np.linalg.norm(y_n_ray_distortion - y_n_reconstructed_ray_distortion))
        print("[F1] Reconstruction error lifted: ",
              np.linalg.norm(y_n_ray_distortion - y_n_reconstructed_lifted_ray_distortion))

    def get_theta_and_psi_matrices(self, csi1, csi2):

        d = np.diag([2, 2, 1])

        theta = np.zeros((3, 6))
        theta[2, 0] = -1
        theta[2, 2] = -1
        theta[:, 3:] = np.eye(3)

        Theta = d @ theta

        psi = np.zeros((3, 6))
        psi[2, 0] = csi2
        psi[2, 2] = csi2
        psi[0, 3] = 0.5
        psi[1, 4] = 0.5
        psi[2, 5] = 1

        Psi = d @ psi

        return Theta.T, Psi.T

    def get_theta_and_psi_aug_matrices(self, csi1, csi2):

        d = np.diag([2, 2, 1])

        theta = np.zeros((3, 6))
        theta[2, 0] = -1
        theta[2, 2] = -1
        theta[:, 3:] = np.eye(3)

        Theta = d @ theta
        Theta = np.concatenate((np.zeros((3, 6)), Theta), axis=0)

        psi = np.zeros((3, 6))
        psi[2, 0] = csi2
        psi[2, 2] = csi2
        psi[0, 3] = 0.5
        psi[1, 4] = 0.5
        psi[2, 5] = 1

        Psi = d @ psi
        Psi = np.concatenate((np.zeros((3, 6)), Psi), axis=0)

        return Theta.T, Psi.T

    def gamma_lifting_operator(self, c1, c2):
        """
        Gamma operator for lifted coordinates.

        :param c1: vector 1
        :type c1: np.ndarray
        :param c2: vector 2
        :type c2: np.ndarray
        :return: veronese transformed vector
        :rtype: np.ndarray
        """
        xi = c1[0]
        xj = c2[0]

        yi = c1[1]
        yj = c2[1]

        zi = c1[2]
        zj = c2[2]

        veronese_coords = np.array([[xi * xj],
                                    [(xi * yj + yi * xj) / 2.0],
                                    [yi * yj],
                                    [(xi * zj + zi * xj) / 2.0],  # xi * zj +
                                    [(yi * zj + zi * yj) / 2.0],
                                    [zi * zj]]).reshape(6, 1)
        return veronese_coords

    def lift_vec_to_veronese(self, vec):
        """
        Lifts vector to veronese coordinates

        :param mat: 3x1 vector in cartesian coordinates
        :type mat: np.ndarray
        :return: 6x1 lifted vector
        :rtype: np.ndarray
        """
        return self.gamma_lifting_operator(vec, vec)

    def lift_image_points(self, points):
        """
        Parallelize lifting points to veronese maps.
        Implemented the relation of Table 2 in
        "Epipolar Geometry of Central Projection
        Systems Using Veronese Maps" by J. P. Barreto
        and K. Daniilidis.

        :param points: image points, 3 x n
        :type points: np.ndarray
        :return: veronese points, 6 x n
        :rtype: np.ndarray
        """

        v_points = np.empty((6, 0))
        for i in range(points.shape[1]):
            v_points = np.append(v_points, self.lift_vec_to_veronese(points[:, i]), axis=1)

        return v_points

    def lift_mat_to_veronese(self, mat):
        """
        Lifts matrix to veronese coordinates

        :param mat: 3x3 matrix in cartesian coordinates
        :type mat: np.ndarray
        :return: 6x6 lifted matrix
        :rtype: np.ndarray
        """
        V_D = np.diag(np.array([1, 2, 1, 2, 2, 1]))

        gamma_11 = self.gamma_lifting_operator(mat[:, [0]], mat[:, [0]])
        gamma_12 = self.gamma_lifting_operator(mat[:, [0]], mat[:, [1]])
        gamma_13 = self.gamma_lifting_operator(mat[:, [0]], mat[:, [2]])
        gamma_22 = self.gamma_lifting_operator(mat[:, [1]], mat[:, [1]])
        gamma_23 = self.gamma_lifting_operator(mat[:, [1]], mat[:, [2]])
        gamma_33 = self.gamma_lifting_operator(mat[:, [2]], mat[:, [2]])

        gamma_mat = np.concatenate((gamma_11, gamma_12, gamma_22, gamma_13, gamma_23, gamma_33), axis=1)
        return np.dot(gamma_mat, V_D)


if __name__ == '__main__':
    unittest.main()
