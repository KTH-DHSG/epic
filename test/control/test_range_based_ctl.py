import numpy as np
import unittest

from epic.model.generalized import GeneralizedCamera
from epic.utils.geometry import get_matched_features, get_relative_pose, get_random_quat_around
from epic.utils.veronese import lift_image_points
from epic.plotting.plotter import Plotter
from random import randint

unittest.TestLoader.sortTestMethodsUsing = lambda _, x, y: randint(-1, 1)


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

        self.dt = 0.1
        self.camera_1 = GeneralizedCamera(
            h=self.dt, name="Leader", model="perspective"
        )

        self.camera_2 = GeneralizedCamera(
            h=self.dt, name="Test Follower 1", model="parabolic"
        )
        q = get_random_quat_around(np.array([0, 0, 0, 1]), 20)
        q = np.array([0, 0, 0, 1])
        self.camera_1.set_formation_pose(np.concatenate((np.array([0, 0, 0]), q)))
        self.camera_1.set_state(np.concatenate((np.array([0, 0, 0]), q)))
        self.camera_1.set_controller("basic", [], config="")

        q = get_random_quat_around(np.array([0, 0, 0, 1]), 20)
        q = np.array([0, 0, 0, 1])

        self.camera_2.set_formation_pose(np.concatenate((np.array([0.2, 0.1, 0.1]), q)))
        self.camera_2.set_state(np.concatenate((np.array([0.2, 0.1, 0.1]), q)))
        self.camera_2.set_controller("ibrc", [self.camera_1], config="f1_mpc_test.json")
        self.camera_2.controller.set_ematrices_with_neighbors()

        self.sim_time_steps = 100
        self.leader_u = np.array([[0, 0, 0, 0, 0, 0]]).T  # Leader's control input

        self.plotter = Plotter([self.camera_1, self.camera_2], features=self.w_points_h)

        # self.leader_u = np.array([[0.001, 0, 0.00, 0, 0, 0.0001]]).T  # Leader's control input
        self.camera_2.set_state(np.array([0.2, 0.2, 0.1, 0, 0, 0, 1]))

    # @unittest.SkipTest
    def test_moving_leader_setpoint_setting(self):
        # self.leader_u = np.array([[0.001, 0, 0.00, 0, 0, 0.0001]]).T  # Leader's control input
        # self.camera_2.set_state(np.array([0.2, 0.1, -0.1, 0, 0, 0, 1]))
        for i in range(self.sim_time_steps):
            print("Time step: ", i, "/", self.sim_time_steps)
            self.camera_1.control(
                u=self.leader_u,
                my_points=None,
                my_points_z=None,
                matched_points=None,
            )

            mf = get_matched_features(
                self.w_points_h,
                my_cam=self.camera_2,
                cam1=self.camera_2.neighbors[0],
            )
            received_data_cam1 = {
                "csi": self.camera_2.neighbors[0].csi,
                "u": self.camera_2.neighbors[0].controller.control_trajectory,
                "Z0": mf["cam1_viz_z"],
                "dt": self.camera_2.neighbors[0].dt,
            }

            # Get control input for F1
            self.camera_2.control(
                leader=self.camera_1,
                my_points=mf["my_cam_u_h"],
                my_points_z=mf["my_cam_viz_z"],
                matched_points=mf["neighbors_matched"],
                neighbor_info=received_data_cam1,
            )

            # Check epipolar constraint
            epi_v_lifted = np.empty((1, 0))
            for i in range(self.w_points_h.shape[1]):
                x_lifted = lift_image_points(mf["neighbors_matched"])
                y_lifted = lift_image_points(mf["my_cam_u_h"])

                # Get essential matrix
                E = self.camera_2.controller.V_E1

                # Check epipolar constraint
                epi_val = y_lifted[:, i].T @ E @ x_lifted[:, i]

                # Append to vector
                epi_v_lifted = np.append(epi_v_lifted, np.array([[epi_val]]), axis=1)

            # print("\nEpipolar constraint: ", epi_v_lifted)

            if i % 10 == 0:
                # self.plotter.plot_environment()
                self.plotter.plot_image_planes()
                self.plotter.plot_curves([self.camera_2])
                # self.plotter.plot_propagated_image_features()
                # self.plotter.plot_costs()
                # input()

            R, t = get_relative_pose(self.camera_1.get_state(), self.camera_2.get_state())
            Rf, tf = get_relative_pose(self.camera_1.get_formation_pose(), self.camera_2.get_formation_pose())
            # Get predicted relative position of leader
            N = self.camera_2.controller.predictive_controller.Nt
            t_pred = np.asarray(self.camera_2.controller.predictive_controller.x_pred).T.reshape((-1, N))
            # print("Prediction of Leader: \n", np.round(t_pred[0:3, :], 3))
            current_r = np.linalg.norm(t_pred[0:3, :], axis=0)
            r_d = self.camera_2.controller.rd
            error_r = np.linalg.norm(current_r - r_d)
            # print("Error prediction: ", error_r)
            # print("Errors: R: ", np.linalg.norm(R - Rf), "   |   t: ", np.linalg.norm(t - tf))
            R, t = get_relative_pose(self.camera_2.get_state(), self.camera_1.get_state())
            # print("Relative position vectors: ", t.T, " | ", self.camera_2.estimated_t.T)

        self.plotter.plot_environment()
        self.plotter.plot_image_planes()
        self.plotter.plot_curves([self.camera_2])
        self.plotter.plot_formation_error()
        self.plotter.plot_costs()
        self.plotter.plot_mpc_costs()
        self.plotter.plot_block()


if __name__.__contains__("__main__"):
    unittest.main()
