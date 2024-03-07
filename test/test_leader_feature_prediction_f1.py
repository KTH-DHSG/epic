import numpy as np
import unittest

from epic.model.generalized import GeneralizedCamera
from epic.utils.geometry import get_matched_features, get_relative_pose, get_random_quat_around, curve_from_vec_to_mat
from epic.utils.veronese import lift_image_points
from epic.plotting.plotter import Plotter
from random import randint

unittest.TestLoader.sortTestMethodsUsing = lambda _, x, y: randint(-1, 1)
# np.set_printoptions(precision=5, suppress=True)


class TestMPRVS(unittest.TestCase):
    def setUp(self):
        # Set world features
        self.w_points_h = np.array(
            [
                [0, 0.5, -0.5, 0, 0.5],
                [0, -0.5, 0.5, 0.5, 0.1],
                [3, 3.5, 2.7, 3.1, 4],
                [1, 1, 1, 1, 1]
            ]
        )

        self.dt = 1
        self.leader = GeneralizedCamera(
            h=self.dt, name="Leader", model="perspective"
        )

        self.f1 = GeneralizedCamera(
            h=self.dt, name="Test Follower 1", model="hyperbolic"
        )

        q1 = get_random_quat_around(np.array([0, 0, 0, 1]), 20)
        self.leader.set_formation_pose(np.concatenate((np.array([0, 0, 0]), q1)))
        self.leader.set_state(np.concatenate((np.array([0, 0, 0]), q1)))
        self.leader.set_controller("basic", [], config="")

        q2 = get_random_quat_around(np.array([0, 0, 0, 1]), 20)
        self.f1.set_formation_pose(np.concatenate((np.array([0.2, 0.1, 0.1]), q2)))
        self.f1.set_state(np.concatenate((np.array([0.2, 0.1, 0.1]), q2)))

        # Set controllers
        self.f1.set_controller("ibrc", [self.leader], config="f1_mpc_test.json")
        self.f1.controller.set_ematrices_with_neighbors()

        self.sim_time_steps = 100  # 50 seconds at dt of 0.1
        self.leader_u = np.array([[0, 0, 0, 0, 0, 0]]).T  # Leader's control input

        self.plotter = Plotter([self.leader, self.f1], features=self.w_points_h)

        # Add starting error:
        # q2 = get_random_quat_around(np.array([0, 0, 0, 1]), 20)
        self.f1.set_state(np.concatenate((np.array([0.2, 0.1, 0.1]), q2)))

        self.start_state_l = self.leader.get_state()
        self.start_state_f1 = self.f1.get_state()

        self.target_state_l = self.leader.get_formation_pose()
        self.target_state_f1 = self.f1.get_formation_pose()

        # Add leader motion
        # self.leader_u = np.array([[0.00, 0.001, 0.0, 0, 0, 0.0]]).T  # Leader's control input

    def test_moving_leaders_setpoint_setting(self):
        for i in range(self.sim_time_steps):
            self.leader.control(
                u=self.leader_u,
                my_points=None,
                my_points_z=None,
                matched_points=None,
            )

            mf = get_matched_features(
                self.w_points_h,
                my_cam=self.f1,
                cam1=self.f1.neighbors[0],
            )
            received_data_cam1 = {
                "csi": self.f1.neighbors[0].csi,
                "u": self.f1.neighbors[0].controller.control_trajectory,
                "Z0": mf["cam1_viz_z"],
                "dt": self.f1.neighbors[0].dt,
            }

            # Check epipolar constraint
            epi_v_lifted = np.empty((1, 0))
            epi_m_condition = np.empty((1, 0))
            x_lifted = lift_image_points(mf["neighbors_matched"])
            y_lifted = lift_image_points(mf["my_cam_u_h"])

            # Get essential matrix
            E = self.f1.controller.V_E1
            for j in range(self.w_points_h.shape[1]):
                # Check epipolar constraint
                epi_val = y_lifted[:, j].T @ E @ x_lifted[:, j]

                # Append to vector
                epi_v_lifted = np.append(epi_v_lifted, np.array([[epi_val]]), axis=1)

                # Get error given lines in matrix form
                line = E @ x_lifted[:, j]
                M = curve_from_vec_to_mat(line)
                y = mf["my_cam_u_h"][:, j]
                epi_m_condition = np.append(epi_m_condition, np.array([[y.T @ M @ y]]), axis=1)

            # Get control input for F1
            self.f1.control(
                leader=self.leader,
                my_points=mf["my_cam_u_h"],
                my_points_z=mf["my_cam_viz_z"],
                matched_points=mf["neighbors_matched"],
                neighbor_info=received_data_cam1,
            )

            if i % 1 == 0:
                self.plotter.plot_formation_error()
                self.plotter.plot_image_planes()
                self.plotter.plot_curves([self.f1])
                if i < 1:
                    self.plotter.plot_propagated_image_features(step=0)
                else:
                    self.plotter.plot_propagated_image_features(step=i - 1)
                input()

            R, t = get_relative_pose(self.leader.get_state(), self.f1.get_state())
            Rf, tf = get_relative_pose(self.leader.get_formation_pose(), self.f1.get_formation_pose())
            print("\n")
            print("Epipolar condition value in Agent 1: \n", epi_v_lifted)
            # print("Errors: R: ", np.linalg.norm(R - Rf), "   |   t: ", np.linalg.norm(t - tf))
            # print("Epipolar condition value in Agent 1: \n", epi_v_lifted, "\n   i) M condition: ",
            # epi_m_condition, "\n  ii) Range: ", np.linalg.norm(t), " / ", np.linalg.norm(tf))
            # print("Epipolar condition value in Agent 2: \n", epi1_v_lifted, "\n", epi2_v_lifted)

        self.plotter.plot_image_planes()
        self.plotter.plot_curves([self.f1])
        self.plotter.plot_costs()
        self.plotter.plot_formation_error()
        self.plotter.plot_predicted_costs()
        self.plotter.plot_mpc_costs()
        self.plotter.plot_block()


if __name__.__contains__("__main__"):
    unittest.main()
