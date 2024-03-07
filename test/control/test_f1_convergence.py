import numpy as np
import unittest

from epic.model.generalized import GeneralizedCamera
from epic.utils.geometry import (
    get_matched_features,
    get_relative_pose,
    get_random_quat_around,
    curve_from_vec_to_mat,
)
from epic.utils.veronese import lift_image_points
from epic.plotting.plotter import Plotter
from random import randint

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
                [1, 1, 1, 1, 1],
            ]
        )

        self.dt = 0.1

        # Generate states
        q = get_random_quat_around(np.array([0, 0, 0, 1]), 0)
        leader = np.array([0.2, 0.5, -0.3, q[0], q[1], q[2], q[3]])
        q = get_random_quat_around(np.array([0, 0, 0, 1]), 10)
        f1 = np.array([0.5, -0.3, 0.3, q[0], q[1], q[2], q[3]])
        # Instatiate cameras
        self.leader = GeneralizedCamera("Leader", state=leader)
        self.f1 = GeneralizedCamera("follower", model="parabolic", state=f1)

        self.leader.set_formation_pose(self.leader.get_state())
        self.leader.set_controller("basic", [], config="")
        self.f1.set_formation_pose(self.f1.get_state())
        self.f1.set_controller("ibrc", [self.leader], config="f1_mpc.json")
        self.f1.controller.set_ematrices_with_neighbors()

        # Instatiate plotter
        self.plotter = Plotter(
            [self.leader, self.f1], features=self.w_points_h
        )

        # Simulation steps
        self.sim_time_steps = 500  # 2 seconds at dt of 0.1

        # Ensure zero control inputs to leader and follower - can be adjusted
        self.leader_u = np.array([[0.001, 0.00, 0.00, 0, 0, 0.001]]).T

        # Set initial error
        f1s = self.f1.get_state()
        f1s = f1s + 0.1 * np.random.rand(
            7,
        )
        f1s[3:] = f1s[3:] / np.linalg.norm(f1s[3:])
        self.f1.set_state(f1s)

    def test_moving_leaders_setpoint_setting(self):
        for i in range(self.sim_time_steps):
            print("Step: ", i, "/", self.sim_time_steps, end="")
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
                epi_m_condition = np.append(
                    epi_m_condition, np.array([[y.T @ M @ y]]), axis=1
                )

            print("\nEpipolar condition value in Agent 1: \n", epi_m_condition)

            # Get control input for F1
            self.f1.control(
                leader=self.leader,
                my_points=mf["my_cam_u_h"],
                my_points_z=mf["my_cam_viz_z"],
                matched_points=mf["neighbors_matched"],
                neighbor_info=received_data_cam1,
                # skip_input=False,
            )

            if False:
                self.plotter.plot_image_planes()
                self.plotter.plot_curves([self.f1])
                self.plotter.plot_mpc_predicted_feature_trajectories()
                self.plotter.plot_predicted_costs()
                exit()

        # self.plotter.plot_environment()
        self.plotter.plot_image_planes()
        self.plotter.plot_selected_image_planes([self.f1])
        self.plotter.plot_curves([self.f1])
        self.plotter.plot_feature_errors()
        self.plotter.plot_formation_error()
        self.plotter.plot_block()


if __name__.__contains__("__main__"):
    unittest.main()
