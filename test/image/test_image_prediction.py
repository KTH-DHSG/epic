import numpy as np
import unittest

from epic.model.generalized import GeneralizedCamera
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
        self.leader = GeneralizedCamera(
            h=self.dt, name="Leader", model="perspective"
        )

        self.f1 = GeneralizedCamera(
            h=self.dt, name="Test Follower 1", model="parabolic"
        )

        # self.f2 = GeneralizedCamera(
        #     h=self.dt, name="Test Follower 2", model="distortion"
        # )

        self.f3 = GeneralizedCamera(
            h=self.dt, name="Test Follower 3", model="parabolic"
        )

        self.leader.set_formation_pose(np.array([0, 0, 0, 0, 0, 0, 1]))
        self.leader.set_state(self.leader.get_formation_pose())
        self.leader.set_controller("basic", [], config="")

        self.f1.set_formation_pose(np.array([0, 0, 0, 0, 0, 0, 1]))
        self.f1.set_state(self.f1.get_formation_pose())
        self.f1.set_controller("basic", [], config="")

        # self.f2.set_formation_pose(np.array([0, 0, 0, 0, 0, 0, 1]))
        # self.f2.set_state(self.f2.get_formation_pose())
        # self.f2.set_controller("basic", [], config="")

        self.f3.set_formation_pose(np.array([0, 0, 0, 0, 0, 0, 1]))
        self.f3.set_state(self.f3.get_formation_pose())
        self.f3.set_controller("basic", [], config="")

        self.sim_time_steps = 5  # 50 seconds at dt of 0.1

        self.plotter = Plotter([self.leader, self.f1, self.f3], features=self.w_points_h)

        # Add leader motion
        self.leader_u = np.array([[0.01, 0.01, 0.0, 0.01, 0.01, 0.01]]).T  # Leader's control input
        self.f1_u = np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01]]).T  # F1's control input
        self.f2_u = np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01]]).T  # F2's control input
        self.f3_u = np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01]]).T  # F3's control input

    def test_moving_leaders_setpoint_setting(self):
        for i in range(self.sim_time_steps):
            _, l_u_points_h, l_viz_z = self.leader.get_image_points(
                self.w_points_h
            )
            self.leader.control(
                u=self.leader_u,
                my_points=l_u_points_h,
                my_points_z=l_viz_z,
                matched_points=None,
            )

            _, f1_u_points_h, f1_viz_z = self.f1.get_image_points(
                self.w_points_h
            )
            self.f1.control(
                u=self.f1_u,
                my_points=f1_u_points_h,
                my_points_z=f1_viz_z,
                matched_points=None,
            )

            # _, f2_u_points_h, f2_viz_z = self.f2.get_image_points(
            #     self.w_points_h
            # )
            # self.f2.control(
            #     u=self.f2_u,
            #     my_points=f2_u_points_h,
            #     my_points_z=f2_viz_z,
            #     matched_points=None,
            # )

            _, f3_u_points_h, f3_viz_z = self.f3.get_image_points(
                self.w_points_h
            )
            self.f3.control(
                u=self.f3_u,
                my_points=f3_u_points_h,
                my_points_z=f3_viz_z,
                matched_points=None,
            )

            if i == 0 or i % 1 == 0:
                self.plotter.plot_image_planes()
                self.plotter.plot_propagated_image_features(step=0)
                input()

        # self.plotter.plot_environment()
        self.plotter.plot_image_planes()
        self.plotter.plot_propagated_image_features(step=0)
        self.plotter.plot_block()


if __name__.__contains__("__main__"):
    unittest.main()
