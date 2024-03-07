import numpy as np
import json
import os

import casadi as ca

from epic.control.mprvs import MPRVS
from epic.utils.geometry import curve_from_vec_to_mat, skew, get_relative_pose
from epic.utils.veronese import set_emat_matrix_with_emat, lift_image_points


class IBRC(object):
    def __init__(self, my_cam, agent1, config_file):
        """
        Image Based Ranging Controller Class.

        :param my_cam: Camera system to apply control to
        :type my_cam: GeneralizedCamera
        :param agent1: Camera system acting as agent 1
        :type agent1: GeneralizedCamera
        :param mpc_args: Arguments for MPC controller
        :type mpc_args: dictionary
        """

        # Check camera pairs validity
        if (agent1.type == "hyperbolic" and my_cam.type != "perspective") or (
            my_cam.type == "hyperbolic" and (agent1.type != "perspective")
        ):
            # Wrong camera pair
            raise ValueError(
                "Hyperbolic cameras can only be paired with perspective ones"
            )

        self.cam1 = agent1
        self.my_cam = my_cam
        self.Rd = None
        self.td = None

        self.empty_control = dict({"u": np.zeros((6, 1)), "-x-": [], "error": np.inf})

        self.E1 = None
        self.V_E1 = None

        self.V_D = np.diag(np.array([1, 2, 1, 2, 2, 1]))

        # Load MPC args
        if type(config_file) is not dict:
            self.config_file = config_file
            mpc_args = self.parse_mpc_args()
        else:
            # Expects a dictionary with the same structure as MPC args
            mpc_args = config_file
        self.predictive_controller = MPRVS(mpc_args)

    def __str__(self):
        return "IBRC"

    def parse_mpc_args(self) -> dict:
        """
        Parse MPC arguments from config file.

        :return: dictionary of MPC parameters
        :rtype: dict
        """
        import epic
        path = os.path.dirname(epic.__file__) + "/config/" + self.config_file
        with open(path, "r") as f:
            mpc_args = json.loads(f.read())
            mpc_args["model"] = self.my_cam
            mpc_args["dynamics"] = self.my_cam.euler_rel_nav

            mpc_args["Q"] = np.diag([mpc_args["Q"][0], *[mpc_args["Q"][1]]*5])
            mpc_args["P"] = np.diag([mpc_args["P"][0], *[mpc_args["P"][1]]*5])
            mpc_args["R"] = np.diag(mpc_args["R"])
            mpc_args["ulb"] = np.array(mpc_args["ulb"])
            mpc_args["uub"] = np.array(mpc_args["uub"])
            mpc_args["xlb"] = np.array(mpc_args["xlb"])
            mpc_args["xub"] = np.array(mpc_args["xub"])
            mpc_args["xf"] = None
        return mpc_args

    def set_ematrices_with_neighbors(self):
        """
        Set essential matrix for the camera from known neighbors.
        """
        # Matrices for F1
        R_l_to_f1, t_l_to_f1 = get_relative_pose(
            pose_from=self.cam1.get_formation_pose(), pose_to=self.my_cam.get_formation_pose()
        )

        # Get control input rotation
        self.u_rot = R_l_to_f1

        self.set_emat_matrix_with_rmat_tvec(R_l_to_f1, t_l_to_f1)
        self.set_target_range(np.linalg.norm(t_l_to_f1))

    def set_emat_matrix_with_rmat_tvec(self, Rmat, tvec):
        """
        Set essential matrices.

        :param Rmats: rotation matrices
        :type Rmats: np.ndarray
        :param tvecs: relative translation vectors
        :type tvecs: np.ndarray
        :raises ValueError: error if the shapes are not what is expected
        """

        # Set target relative pose
        self.set_target_rotation(Rmat.T)
        self.set_target_td(-(tvec / np.linalg.norm(tvec)))

        if len(Rmat.shape) != 2 or len(tvec.shape) != 2:
            raise ValueError(
                "Expected `R` and `t` to have shape (3, 3, 1) and (3, 1, 1) "
                "got {} and {}".format(Rmat.shape, tvec.shape)
            )

        self.E = np.dot(Rmat.reshape((3, 3)), skew(tvec.reshape((3, 1))))
        self.V_E1 = set_emat_matrix_with_emat(self.E, self.my_cam, self.cam1)
        return



    def set_emat_matrix(self, Emat):

        E_mat = Emat.reshape((3, 3))
        self.V_E1 = set_emat_matrix_with_emat(E_mat, self.my_cam, self.cam1)
        return

    def set_RLF(self, RLF):
        self.RLF = RLF

    def set_target_range(self, rd):
        self.rd = rd

    def set_target_rotation(self, Rd):
        self.Rd = Rd

    def set_target_td(self, td):
        self.td = td

    def set_u_rot(self, u_rot):
        self.u_rot = u_rot

    def get_range_to(self, neighbor):
        """
        Get range between current agent and desired neighbor.

        :param neighbor: neighbor agent
        :type neighbor: GenericCam
        """

        my_pos = self.my_cam.get_position()
        n_pos = neighbor.get_position()
        return np.linalg.norm(my_pos - n_pos, ord=2)

    def adjust_neighbor_data_length(self, data):
        """
        Adjust the length of the neighbor data to match the control prediction horizon.

        :param data: received data
        :type data: np.ndarray
        :return: adjusted data
        :rtype: np.ndarray
        """

        if data.shape[1] < self.predictive_controller.Nt - 1:
            # If received data is smaller than our control prediction horizon, we repeat the last point
            repeated_last = np.repeat(data[:, [-1]], self.predictive_controller.Nt - data.shape[1] - 1, axis=1)
            return np.concatenate((data, repeated_last), axis=1)
        elif data.shape[1] > self.predictive_controller.Nt - 1:
            # If the received data is larger than our control prediction horizon, we truncate it
            return data[:, 0:self.predictive_controller.Nt - 1]
        else:
            return data

    def get_trajectory_reference(self, matched_points, neighbor_info):
        """
        Get trajectory reference for the current agent.

        :param matched_points: matched neighbor points
        :type matched_points: np.ndarray
        :param u: neighbor control inputs
        :type u: np.ndarray
        :param cam_params: camera parameters for trajectory prediction
        :type cam_params: dict
        :return: predicted range and trajectory for curves
        :rtype: np.ndarray
        """
        propagate_neighbor = self.my_cam.get_generic_camera_integrator(neighbor_info["csi"], neighbor_info["dt"])
        u = self.adjust_neighbor_data_length(data=neighbor_info["u"])
        Z0 = neighbor_info["Z0"]
        xn_traj = matched_points.reshape((-1, 1))

        # Initialize trajectory data
        u_ag1_lines = np.empty((6, 5, 0))
        xr = np.empty((46, 0))
        for ui in range(u.shape[1]):
            v_matched_points_ag1 = lift_image_points(xn_traj[:, -1].reshape((3, -1)))
            u_ag1_lines = np.append(u_ag1_lines, np.dot(self.V_E1, v_matched_points_ag1).reshape(6, 5, 1), axis=2)
            # Log first set of epipolar curves for the first
            if ui == 0:
                logged_ug1_line = u_ag1_lines[:, :, 0]

            # Add target range
            xr_i = np.array([[self.rd]])
            for i in range(self.my_cam.f_n):
                lines = curve_from_vec_to_mat(
                    u_ag1_lines[:, i, ui]).flatten(order="F").reshape((9, 1))
                xr_i = np.append(xr_i, lines, axis=0)

            # Update xr
            xr = np.append(xr, xr_i, axis=1)

            # Update next set of points
            features = xn_traj[:, -1].reshape((3, -1))
            fn = features[0:2, :]
            fn_plus_1_cd = propagate_neighbor(x0=fn.reshape((-1, 1), order="F"), p=ca.vertcat(u[:, ui].reshape((-1, 1)), Z0[0].T))["xf"]
            xn = np.vstack((np.asarray(fn_plus_1_cd).reshape((2, -1), order="F"), np.ones((1, np.asarray(fn_plus_1_cd).reshape((2, -1)).shape[1]))))
            xn_traj = np.append(xn_traj, xn.reshape((-1, 1)), axis=1)

        # Do last iteration
        v_matched_points_ag1 = lift_image_points(xn_traj[:, -1].reshape((3, -1)))
        u_ag1_lines = np.append(u_ag1_lines, np.dot(self.V_E1, v_matched_points_ag1).reshape(6, 5, 1), axis=2)
        xr_i = np.array([[self.rd]])
        for i in range(self.my_cam.f_n):
            lines = curve_from_vec_to_mat(
                u_ag1_lines[:, i, -1]).flatten(order="F").reshape((9, 1))
            xr_i = np.append(xr_i, lines, axis=0)

        # Update xr
        xr = np.append(xr, xr_i, axis=1)

        # Broadcast xr to 1D vector
        xr = xr.reshape(-1, 1, order="F")

        return xr, u, logged_ug1_line

    def tranform_u_to_my_frame(self, u):
        """
        Transform control input from neighbor frame to my frame.

        :param u: control input from neighbor
        :type u: np.ndarray
        :return: control input in my frame
        :rtype: np.ndarray
        """
        u = u.reshape((6, -1))

        # Rotate each of the inputs to our reference frame
        vL = u[0:3, :]
        wL = u[3:, :]

        rel_vec = self.td * self.rd  # goes from follower to leader
        VF_in_L = vL[0:3, :] + np.cross(wL.T, -rel_vec.T).T

        # Get desired linear and angular velocity in my frame

        u_rot = np.concatenate((self.u_rot @ VF_in_L.reshape(3, -1),
                                self.u_rot @ wL), axis=0)
        return u_rot

    def control(self, t12, my_points, my_points_z, matched_points, **kwargs):
        """
        Control law for image-based only coordination.

        :param my_points: current camera observations, image points
        :type my_point: numpy array, 4xN
        :param matched_points: matched image points for remaining agents
        :type matched_points: 3d numpy array, 4 x N x 2
        :return: control input
        :rtype: numpy array 6D
        """

        # Extract non-homogenous component w/o depth (grab x,y)
        xr, ur, u_ag1_lines = self.get_trajectory_reference(matched_points, kwargs["neighbor_info"])
        u_sp = self.tranform_u_to_my_frame(ur)
        self.predictive_controller.set_reference(xr, u_sp=u_sp, test_RLF=self.RLF)

        # Control observations towards intersected points
        self.z0 = my_points_z[0][:, :].reshape(5, 1)
        x = np.hstack(
            (np.asarray(t12).flatten(order="F"), my_points[0:2, :].flatten(order="F"))
        )

        u = self.predictive_controller.control(x, z0=self.z0)
        predicted_costs = self.predictive_controller.get_predicted_costs()
        if "skip_input" in kwargs.keys():
            u = np.zeros((6, ))
        predicted_features = self.predictive_controller.get_predicted_features()

        # Create dictionary to plot outside
        mpc_error = self.predictive_controller.get_error_vector(x, xr)
        veronese_error = self.predictive_controller.get_veronese_error_vector(
            x, xr, u_ag1_lines
        )

        # Controller trajectory
        self.control_trajectory = np.asarray(self.predictive_controller.u_pred).reshape(self.predictive_controller.Nt - 1, -1).T

        self.predictive_controller.compare_errors(x, xr, u_ag1_lines)

        data_struct = dict(
            {
                "u": np.asarray(u),
                "l": u_ag1_lines,
                "error": np.asarray(mpc_error),
                "verror": np.asarray(veronese_error),
                "x": x,
                "xr": xr,
                "J": self.predictive_controller.get_cost(),
                "ct": self.predictive_controller.get_last_solve_time(),
                "predicted_costs": predicted_costs,
                "propagated_features": predicted_features
            }
        )
        return data_struct

    def empty_log(self) -> dict:
        """
        Empty dictionary for logged variables.

        :return: empty dictionary with same footprint as control
        :rtype: dict
        """
        xr = np.empty((46 * self.predictive_controller.Nt, 0))
        return dict({"u": np.empty((self.my_cam.m, 0)),
                     "l": np.empty((6, self.my_cam.f_n, 0)),
                     "error": np.empty((6, 0)),
                     "verror": np.empty((6, 0)),
                     "x": np.empty((13, 0)),
                     "xr": xr,
                     "J": np.empty((1, 0)),
                     "ct": np.empty((1, 0)),
                     "predicted_costs": {"x": np.empty((1, 0)), "u": np.empty((1, 0))},
                     "propagated_features": np.empty((self.my_cam.f_n * 2, self.predictive_controller.Nt, 0))
                     })
