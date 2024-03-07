import unittest

from epic.utils.geometry import get_relative_pose, skew, r_mat
from epic.utils.veronese import lift_image_points, lift_mat_to_veronese, lift_vec_to_veronese
from epic.model.generalized import GeneralizedCamera
from random import randint
from math import sqrt
from scipy.spatial.transform import Rotation as Rotations

import numpy as np

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

        # Generate random pose
        leader_p = np.random.randn(3, 1)
        leader_q = np.random.randn(4, 1)
        leader_q = leader_q / np.linalg.norm(leader_q)

        f1_p = np.random.randn(3, 1)
        f1_q = np.random.randn(4, 1)
        f1_q = f1_q / np.linalg.norm(f1_q)

        leader = np.concatenate((leader_p, leader_q), axis=0).reshape((7,))
        f1 = np.concatenate((f1_p, f1_q), axis=0).reshape((7,))
        self.leader = GeneralizedCamera("leader", state=leader)
        self.f1 = GeneralizedCamera("follower", model="parabolic", state=f1)

        self.leader.set_formation_pose(self.leader.get_state())
        self.f1.set_formation_pose(self.f1.get_state())
        self.f1.set_controller("ibrc", [self.leader], config="f1_mpc.json")
        self.f1.controller.set_ematrices_with_neighbors()
        pass

    def test_lifted_epipolar_condition_perspective_parabolic(self):
        K = np.eye(3)
        K_inv = np.eye(3)

        leader = self.leader.get_state()
        f1 = self.f1.get_state()
        # Get relative pose
        R, t = get_relative_pose(leader, f1)

        # According to wikipedia this is correct
        # According to his paper, it is not (probably different frames)
        E = R @ skew(t)

        # Compare with the one from the controller
        E_controller = self.f1.controller.E

        assert np.allclose(E, E_controller), "E matrices are not the same"

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
        E_v = np.empty((1, 0))
        for i in range(self.w_points_h.shape[1]):
            x_m = P1 @ self.w_points_h[:, i]
            xp = self.get_image_with_h(x_m, csi1)

            y_m = P2 @ self.w_points_h[:, i]
            yp = self.get_image_with_h(y_m, csi2)

            Ei_v = y_m.T @ E @ x_m
            E_v = np.append(E_v, np.array([[Ei_v]]), axis=1)
            assert np.allclose(
                Ei_v, y_m.T @ self.f1.controller.E @ x_m
            ), "Lifted epipolar constraint is not met"

            # Save projected points
            xp_v = np.append(xp_v, np.array([xp]).T, axis=1)
            yp_v = np.append(yp_v, np.array([yp]).T, axis=1)

            # Save rays
            x_n_ray = np.append(x_n_ray, np.array([x_m]).T, axis=1)
            y_n_ray = np.append(y_n_ray, np.array([y_m]).T, axis=1)

        # Obtain image points and rays with framework
        x_px_points_h, xp_v_framework, (x_depth, x_delta) = self.leader.get_image_points(
            self.w_points_h
        )
        y_px_points_h, yp_v_framework, (y_depth, y_delta) = self.f1.get_image_points(
            self.w_points_h
        )

        # Compare calibrated image points
        assert np.allclose(xp_v, xp_v_framework), "Leader image points are not the same"
        assert np.allclose(yp_v, yp_v_framework), "Follower image points are not the same"

        # Check lifted epipolar constraint
        # Get veronese points
        x_lifted = lift_image_points(xp_v)
        y_lifted = lift_image_points(yp_v)

        H_lifted = lift_mat_to_veronese(Hc)
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
            # Ensure that controller E mat produces same result
            assert np.allclose(
                epi_val, y @ self.f1.controller.E @ x
            ), "Lifted epipolar constraint is not met"

            # Save lifted epipolar value and ray
            E_v_lifted = np.append(E_v_lifted, np.array([[epi_val]]), axis=1)
            x_n_reconstructed_lifted_ray = np.append(
                x_n_reconstructed_lifted_ray, np.array([x]).T, axis=1
            )
            y_n_reconstructed_lifted_ray = np.append(
                y_n_reconstructed_lifted_ray, np.array([y]).T, axis=1
            )

        # Check lifted matrices with augmentation
        # Theta_aug, _ = self.get_theta_and_psi_aug_matrices(csi1, csi2)
        Theta_aug = self.f1.Theta

        # Augment E
        V_E = np.zeros((6, 6))
        V_E[3:, 3:] = E
        V_E = Theta_aug @ V_E
        V_E_framework = self.f1.controller.V_E1
        assert np.allclose(V_E, V_E_framework), "V_E matrix is not the same"

        # Augment K
        V_K = np.zeros((6, 6))
        V_K[3:, 3:] = K
        V_K_inv = np.zeros((6, 6))
        V_K_inv[3:, 3:] = K_inv

        E_v_lifted_aug = np.empty((1, 0))
        for i in range(self.w_points_h.shape[1]):
            x = x_lifted[:, i]
            y = y_lifted[:, i].T

            epi_val = y @ V_E @ x

            # Save lifted epipolar value and ray
            E_v_lifted_aug = np.append(E_v_lifted_aug, np.array([[epi_val]]), axis=1)

        # Check if epipolar constraint is met
        print("Epipolar constraint: ", np.round(E_v, 15))
        print("Lifted Epipolar constraint: ", np.round(E_v_lifted, 15))
        print("Lifted Epipolar constraint augmented: ", np.round(E_v_lifted_aug, 15))

    def test_parabolas(self):
        K = np.eye(3)
        K_inv = np.eye(3)

        leader = self.leader.get_state()
        f1 = self.f1.get_state()
        # Get relative pose
        R, t = get_relative_pose(leader, f1)

        # According to wikipedia this is correct
        # According to his paper, it is not (probably different frames)
        E = R @ skew(t)

        # Compare with the one from the controller
        E_controller = self.f1.controller.E

        assert np.allclose(E, E_controller), "E matrices are not the same"

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
        E_v = np.empty((1, 0))
        for i in range(self.w_points_h.shape[1]):
            x_m = P1 @ self.w_points_h[:, i]
            xp = self.get_image_with_h(x_m, csi1)

            y_m = P2 @ self.w_points_h[:, i]
            yp = self.get_image_with_h(y_m, csi2)

            Ei_v = y_m.T @ E @ x_m
            E_v = np.append(E_v, np.array([[Ei_v]]), axis=1)
            assert np.allclose(
                Ei_v, y_m.T @ self.f1.controller.E @ x_m
            ), "Lifted epipolar constraint is not met"

            # Save projected points
            xp_v = np.append(xp_v, np.array([xp]).T, axis=1)
            yp_v = np.append(yp_v, np.array([yp]).T, axis=1)

            # Save rays
            x_n_ray = np.append(x_n_ray, np.array([x_m]).T, axis=1)
            y_n_ray = np.append(y_n_ray, np.array([y_m]).T, axis=1)

        # Obtain image points and rays with framework
        x_px_points_h, xp_v_framework, (x_depth, x_delta) = self.leader.get_image_points(
            self.w_points_h
        )
        y_px_points_h, yp_v_framework, (y_depth, y_delta) = self.f1.get_image_points(
            self.w_points_h
        )

        # Compare calibrated image points
        assert np.allclose(xp_v, xp_v_framework), "Leader image points are not the same"
        assert np.allclose(yp_v, yp_v_framework), "Follower image points are not the same"

        # Check lifted epipolar constraint
        # Get veronese points
        x_lifted = lift_image_points(xp_v)
        y_lifted = lift_image_points(yp_v)

        H_lifted = lift_mat_to_veronese(Hc)
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
            # Ensure that controller E mat produces same result
            assert np.allclose(
                epi_val, y @ self.f1.controller.E @ x
            ), "Lifted epipolar constraint is not met"

            # Save lifted epipolar value and ray
            E_v_lifted = np.append(E_v_lifted, np.array([[epi_val]]), axis=1)
            x_n_reconstructed_lifted_ray = np.append(
                x_n_reconstructed_lifted_ray, np.array([x]).T, axis=1
            )
            y_n_reconstructed_lifted_ray = np.append(
                y_n_reconstructed_lifted_ray, np.array([y]).T, axis=1
            )

        # Check lifted matrices with augmentation
        # Theta_aug, _ = self.get_theta_and_psi_aug_matrices(csi1, csi2)
        Theta_aug = self.f1.Theta

        # Augment E
        V_E = np.zeros((6, 6))
        V_E[3:, 3:] = E
        V_E = Theta_aug @ V_E
        V_E_framework = self.f1.controller.V_E1
        assert np.allclose(V_E, V_E_framework), "V_E matrix is not the same"

        # Augment K
        V_K = np.zeros((6, 6))
        V_K[3:, 3:] = K
        V_K_inv = np.zeros((6, 6))
        V_K_inv[3:, 3:] = K_inv

        E_v_lifted_aug = np.empty((1, 0))
        for i in range(self.w_points_h.shape[1]):
            x = x_lifted[:, i]
            y = y_lifted[:, i].T

            epi_val = y @ V_E @ x

            # Save lifted epipolar value and ray
            E_v_lifted_aug = np.append(E_v_lifted_aug, np.array([[epi_val]]), axis=1)

        # Check if epipolar constraint is met
        print("Epipolar constraint: ", np.round(E_v, 15))
        print("Lifted Epipolar constraint: ", np.round(E_v_lifted, 15))
        print("Lifted Epipolar constraint augmented: ", np.round(E_v_lifted_aug, 15))
    """
    Extra includes for testing.
    """
    def h(self, x, csi, rho):
        return np.array([x[0], x[1], x[2] + csi * rho])

    def get_image_with_h(self, x, csi):
        rho = np.linalg.norm(x)
        xp = self.h(x, csi, rho)
        xp = xp / xp[2]
        return xp

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

if __name__ == "__main__":
    unittest.main()
