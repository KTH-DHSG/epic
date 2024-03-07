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
# np.set_printoptions(precision=5, suppress=True)


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
        q = get_random_quat_around(np.array([0, 0, 0, 1]), 5)
        f1 = np.array([0.5, -0.3, 0.3, q[0], q[1], q[2], q[3]])
        q = get_random_quat_around(np.array([0, 0, 0, 1]), 5)
        f2 = np.array([0.7, 0.2, -0.1, q[0], q[1], q[2], q[3]])

        # Instatiate cameras
        self.leader = GeneralizedCamera(h=self.dt, name="Leader", model="perspective", state=leader)
        self.f1 = GeneralizedCamera(h=self.dt, name="Test Follower 1", model="perspective", state=f1)
        self.f2 = GeneralizedCamera(h=self.dt, name="Test Follower 2", model="parabolic", state=f2)

        # Set formation pose
        self.leader.set_formation_pose(self.leader.get_state())
        self.f1.set_formation_pose(self.f1.get_state())
        self.f2.set_formation_pose(self.f2.get_state())

        # Set controllers
        self.leader.set_controller("basic", [], config="")
        self.f1.set_controller("ibrc", [self.leader], config="f1_mpc.json")
        self.f1.controller.set_ematrices_with_neighbors()
        self.f2.set_controller(
            "ibfc", [self.leader, self.f1], config="f2_mpc.json", method="mpibvs"
        )
        self.f2.controller.set_ematrices_with_neighbors()

        # Instatiate plotter
        self.plotter = Plotter(
            [self.leader, self.f1, self.f2], features=self.w_points_h
        )

        # Simulation steps
        self.sim_time_steps = 300  # 2 seconds at dt of 0.1

        # Ensure zero control inputs to leader and follower - can be adjusted
        self.leader_u = np.array([[0.01, 0.00, 0.0, 0, 0, 0.0]]).T
        self.f1_u = np.array([[0.00, 0.00, 0.0, 0, 0, 0.0]]).T

        # Set F2 offset:
        f2s = self.f2.get_state()
        f2s = f2s + 1*np.random.rand(7,)
        f2s[3:] = f2s[3:] / np.linalg.norm(f2s[3:])
        self.f2.set_state(f2s)

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
                epi_m_condition = np.append(
                    epi_m_condition, np.array([[y.T @ M @ y]]), axis=1
                )

            print("Epipolar condition value in Agent 1: \n", epi_m_condition)

            # Get control input for F1
            self.f1.control(
                leader=self.leader,
                my_points=mf["my_cam_u_h"],
                my_points_z=mf["my_cam_viz_z"],
                matched_points=mf["neighbors_matched"],
                neighbor_info=received_data_cam1,
                #skip_input=True,
            )

            # Get control input for camera 2
            mf = get_matched_features(
                self.w_points_h,
                my_cam=self.f2,
                cam1=self.f2.neighbors[0],
                cam2=self.f2.neighbors[1],
            )
            received_data_cam1 = {
                "csi": self.f2.neighbors[0].csi,
                "u": self.f2.neighbors[0].controller.control_trajectory,
                "Z0": mf["cam1_viz_z"],
                "dt": self.f2.neighbors[0].dt,
            }
            received_data_cam2 = {
                "csi": self.f2.neighbors[1].csi,
                "u": self.f2.neighbors[1].controller.control_trajectory,
                "Z0": mf["cam2_viz_z"],
                "dt": self.f2.neighbors[1].dt,
            }

            # Get control input for F2
            self.f2.control(
                my_points=mf["my_cam_u_h"],
                my_points_z=mf["my_cam_viz_z"],
                matched_points=mf["neighbors_matched"],
                neighbor_info_1=received_data_cam1,
                neighbor_info_2=received_data_cam2
            )

            # Check epipolar constraint
            epi1_v_lifted = np.empty((1, 0))
            epi2_v_lifted = np.empty((1, 0))
            for j in range(self.w_points_h.shape[1]):
                x1_lifted = lift_image_points(mf["neighbors_matched"][:, :, 0])
                x2_lifted = lift_image_points(mf["neighbors_matched"][:, :, 1])
                y_lifted = lift_image_points(mf["my_cam_u_h"])

                # Get essential matrix
                E1 = self.f2.controller.V_E1
                E2 = self.f2.controller.V_E2

                # Check epipolar constraint
                epi1_val = y_lifted[:, j].T @ E1 @ x1_lifted[:, j]
                epi2_val = y_lifted[:, j].T @ E2 @ x2_lifted[:, j]

                # Append to vector
                epi1_v_lifted = np.append(epi1_v_lifted, np.array([[epi1_val]]), axis=1)
                epi2_v_lifted = np.append(epi2_v_lifted, np.array([[epi2_val]]), axis=1)

            # Print it:
            print(
                "Epipolar condition value in Agent 2: \n",
                epi1_v_lifted,
                "\n",
                epi2_v_lifted,
            )

            if (i == 0 or i % 1 == 0) and False:
                self.plotter.plot_image_planes()
                self.plotter.plot_curves([self.f1, self.f2])
                self.plotter.plot_mpc_predicted_feature_trajectories()
                # self.plotter.plot_predicted_intersections(step=0)
                # self.plotter.plot_mpc_predicted_feature_trajectories(step=0)
                # self.plotter.plot_predicted_costs()
                # self.plotter.plot_mpc_costs()
                # self.plotter.plot_costs()
                # self.plotter.plot_formation_error()
                # input()

            # R, t = get_relative_pose(self.leader.get_state(), self.f1.get_state())
            # Rf, tf = get_relative_pose(self.leader.get_formation_pose(), self.f1.get_formation_pose())
            # print("\n")
            # print("Errors: R: ", np.linalg.norm(R - Rf), "   |   t: ", np.linalg.norm(t - tf))
            # input()
            # print("Epipolar condition value in Agent 1: \n", epi_v_lifted, "\n   i) M condition: ",
            # epi_m_condition, "\n  ii) Range: ", np.linalg.norm(t), " / ", np.linalg.norm(tf))
            # print("Epipolar condition value in Agent 2: \n", epi1_v_lifted, "\n", epi2_v_lifted)

        # self.plotter.plot_environment()
        self.plotter.plot_image_planes()
        self.plotter.plot_selected_image_planes([self.f1, self.f2])
        self.plotter.plot_curves([self.f1, self.f2])
        self.plotter.plot_feature_errors()
        # self.plotter.plot_predicted_intersections(step=0)
        self.plotter.plot_costs()
        self.plotter.plot_formation_error()
        # self.plotter.plot_predicted_costs()
        # self.plotter.plot_mpc_costs()
        self.plotter.plot_block()
        print(self.f2.get_formation_pose())

if __name__.__contains__("__main__"):
    unittest.main()
