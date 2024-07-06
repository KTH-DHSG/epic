# !/usr/bin/env python

import numpy as np
import json
import os

from epic.model.generalized import GeneralizedCamera
from epic.plotting.plotter import Plotter
from epic.utils.geometry import get_matched_features, get_random_quat_around

# -----------------------------------------------------------------------------
#                              SIMULATION SETUP
# -----------------------------------------------------------------------------
# Plot type: 'interactive' or 'final'
PLOT = "final"
INTERACTIVE_PLOT_TIME_STEP = 1  # in seconds
TEST = "random"  # generate random initial conditions
followers_cameras = ["perspective", "parabolic", "hyperbolic", "parabolic", "perspective"]
camera_leader = "perspective"
trigger_local_minima = False
change_features = False
# -----------------------------------------------------------------------------
#                            END SIMULATION SETUP
# -----------------------------------------------------------------------------


# Fix printing too many zeros
np.set_printoptions(formatter={"float": "{: 0.3f}".format})
STATIC_TEST = False


class SimulationThreeAgents(object):
    def __init__(self):
        # Set number of runs
        self.sim_time = 15
        self.dt = 0.1
        self.sim_time_steps = int(self.sim_time / self.dt)
        self.method = "mpibvs"

        self.static_leader = False

        # Load formation geometry
        path = (
            os.path.dirname(__file__)
            + "/6_agent_geometry.json"
        )
        f = open(path, "r")
        self.formation_geometry = json.loads(f.read())

        # Generate world points
        self.w_points = np.array(
            [
                [0, 0.5, -0.5, 0, 0.5, 0.25],
                [0, -0.5, 0.5, 0.5, 0.1, 0.35],
                [5, 5, 5, 5, 5, 5],
            ]
        )

        self.w_points_h = np.concatenate((self.w_points, np.ones((1, self.w_points.shape[1]))), axis=0)

    def run(self):
        """
        Run simulation.
        """

        # Parse camera types and settings
        leader = GeneralizedCamera(
            h=self.dt,
            name="Leader",
            model=camera_leader,
        )

        if not self.static_leader:
            leader_u = np.array([[0.01, 0, 0, 0, 0, 0.00]]).T
        else:
            leader_u = np.array([[0, 0, 0, 0, 0, 0]]).T

        leader.set_formation_pose(np.array(self.formation_geometry["leader"]))
        leader.set_state(np.array([0.2, 0.5, -0.3, 0, 0, 0, 1]))
        leader.set_controller("basic", [], config="")

        # Instantiate all agents
        followers = {}
        poses_dict = {}
        for i, camera in enumerate(followers_cameras):
            # Create cameras
            followers[i] = GeneralizedCamera(
                h=self.dt,
                name="Follower {}".format(i + 1),
                model=camera,
            )

            # Instantiate controllers
            if i == 0:
                followers[i].set_controller("ibrc", [leader], config="f1_mpc.json")
            else:
                followers[i].set_controller(
                    "ibfc", [leader, followers[0]], config="f2_mpc.json", method=self.method
                )

            # Set test type
            if TEST == "random":
                # Set start pose from previously random generated data
                pose_from_json = np.array(self.formation_geometry["follower_{}".format(i + 1)])
                position = pose_from_json[:3]
                q = get_random_quat_around(pose_from_json[3:], 5)
                desired_state = np.hstack((position, q))

                followers[i].set_formation_pose(
                    desired_state
                )
                if trigger_local_minima:
                    pos_error = 3.0
                    quat_error = 0.5
                else:
                    pos_error = 0.5
                    quat_error = 0.2
                start_state = desired_state + np.hstack((pos_error * np.random.rand(3,), quat_error * np.random.rand(4,)))
                start_state[3:] = start_state[3:] / np.linalg.norm(start_state[3:])
                followers[i].set_state(
                    start_state
                )
            elif TEST == "keep":
                # Set random start pose
                pose_from_json = np.array(self.formation_geometry["follower_{}".format(i + 1)])
                followers[i].set_state(
                    pose_from_json
                )
                followers[i].set_formation_pose(
                    pose_from_json
                )

            followers[i].controller.set_ematrices_with_neighbors()
            poses_dict["follower_{}_x0_".format(i)] = followers[i].get_state()
            poses_dict["follower_{}_formation_".format(i)] = followers[i].get_formation_pose()
            print("Follower {}:".format(i), followers[i].get_state())
            print("Follower {} formation:".format(i), followers[i].get_formation_pose())
        # Create plotter
        self.plotter = Plotter([leader, *followers.values()], features=self.w_points_h)

        for t in range(self.sim_time_steps):
            # Main loop for each agent
            for i, camera in enumerate(followers_cameras):
                # Get matched features
                if i == 0:
                    print("Moving Leader... ", end="")
                    # Use simple control input for leader
                    leader.control(
                        u=leader_u,
                        my_points=None,
                        my_points_z=None,
                        matched_points=None,
                    )

                    print("\nSolving ", str(followers[i]), ": ", end="")
                    mf = get_matched_features(
                        self.w_points_h,
                        my_cam=followers[i],
                        cam1=followers[i].neighbors[0],
                        change_features=change_features
                    )
                    received_data_cam1 = {
                        "csi": followers[i].neighbors[0].csi,
                        "u": followers[i].neighbors[0].controller.control_trajectory,
                        "Z0": mf["cam1_viz_z"],
                        "dt": followers[i].neighbors[0].dt,
                    }

                    # Get control input for F1
                    followers[i].control(
                        leader=leader,
                        my_points=mf["my_cam_u_h"],
                        my_points_z=mf["my_cam_viz_z"],
                        matched_points=mf["neighbors_matched"],
                        neighbor_info=received_data_cam1,
                    )

                else:
                    mf = get_matched_features(
                        self.w_points_h,
                        my_cam=followers[i],
                        cam1=followers[i].neighbors[0],
                        cam2=followers[i].neighbors[1],
                        change_features=change_features
                    )
                    received_data_cam1 = {
                        "csi": followers[i].neighbors[0].csi,
                        "u": followers[i].neighbors[0].controller.control_trajectory,
                        "Z0": mf["cam1_viz_z"],
                        "dt": followers[i].neighbors[0].dt,
                    }
                    received_data_cam2 = {
                        "csi": followers[i].neighbors[1].csi,
                        "u": followers[i].neighbors[1].controller.control_trajectory,
                        "Z0": mf["cam2_viz_z"],
                        "dt": followers[i].neighbors[1].dt,
                    }

                    print("\nSolving ", str(followers[i]), ": ", end="")
                    # Get control input for F2
                    followers[i].control(
                        my_points=mf["my_cam_u_h"],
                        my_points_z=mf["my_cam_viz_z"],
                        matched_points=mf["neighbors_matched"],
                        neighbor_info_1=received_data_cam1,
                        neighbor_info_2=received_data_cam2,
                    )

            # Sleep
            print(" Steps: {}/{}".format(t + 1, self.sim_time_steps))
            if (
                PLOT == "interactive" and t % float(INTERACTIVE_PLOT_TIME_STEP / self.dt) == 0
            ):
                self.plotter.plot_formation_error()
                self.plotter.plot_environment()
                self.plotter.plot_image_planes()
                self.plotter.plot_curves([*followers.values()])
                self.plotter.plot_costs()
                self.plotter.plot_cpu_times()

            if STATIC_TEST:
                break

        # Save data
        # for i, camera in enumerate(followers_cameras):
        #     followers[i].save_log()
        self.plotter.plot_formation_error()
        self.plotter.plot_selected_image_planes([followers[0], followers[1]])
        self.plotter.plot_velocity_input(velocity_type="linear")
        self.plotter.plot_velocity_input(velocity_type="angular")
        self.plotter.plot_cpu_times()
        input("click Enter for animation...")
        self.plotter.create_animation(wpoints=self.w_points, save_video=False, show_trajectory=True,
                                      folder=os.path.dirname(__file__) + "/output/")
        self.plotter.plot_block()


if __name__ == "__main__":
    sim_env = SimulationThreeAgents()
    sim_env.run()
