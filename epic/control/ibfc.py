import numpy as np

import os
import json

import casadi as ca

from epic.control.mpibvs import MPIBVS
from epic.control.ibvs import IBVS
from epic.utils.veronese import set_emat_matrix_with_emat, lift_image_points
from epic.utils.geometry import get_parabola_intersections, get_lines_intersection, is_line, skew, get_relative_pose, get_closest_essential_matrix


class IBFC(object):
    def __init__(self, my_cam, agent1, agent2, config_file, **kwargs):
        """
        Image Based Formation Controller Class.

        :param my_cam: Camera system to apply control to
        :type my_cam: GeneralizedCamera
        :param agent1: Camera system acting as agent 1
        :type agent1: GeneralizedCamera
        :param agent2: Camera system acting as agent 2
        :type agent2: GeneralizedCamera
        :param kwargs['method']: Control method (MPIBVS or IBVS)
        :type kwargs['method']: string
        """

        # Check camera pairs validity
        if (
            (agent1.type == "hyperbolic" and my_cam.type != "perspective")
            or (agent2.type == "hyperbolic" and my_cam.type != "perspective")
            or (
                my_cam.type == "hyperbolic"
                and (agent1.type != "perspective" or agent2.type != "perspective")
            )
        ):
            # Wrong camera pair
            raise ValueError(
                "Hyperbolic cameras can only be paired with perspective cameras."
            )

        self.cam1 = agent1
        self.cam2 = agent2
        self.my_cam = my_cam
        self.config_file = config_file

        self.empty_control = dict({"u": np.zeros((6, 1)), "-x-": np.empty((2, 5, 1)), "error": np.inf * np.ones((2, 5, 1))})

        self.E1 = None
        self.E2 = None

        self.V_E1 = None
        self.V_E2 = None

        self.V_D = np.diag(np.array([1, 2, 1, 2, 2, 1]))

        # Load MPC args
        if type(self.config_file) is not dict:
            self.config_file = config_file
            self.mpc_args = self.parse_mpc_args()
        else:
            # Expects a dictionary with the same structure as MPC args
            self.mpc_args = config_file

        self.method = kwargs.get("method", "mpibvs")
        if self.method == "mpibvs":
            self.predictive_controller = MPIBVS(self.mpc_args)
            self.control_trajectory = np.zeros((self.predictive_controller.Nu, self.predictive_controller.Nt - 1)).reshape((self.predictive_controller.Nt - 1, -1)).T
        elif self.method == "ibvs":
            self.controller = IBVS(cam=self.my_cam)

        self.z0 = None

        # convergence method:

    def __str__(self):
        return "IBFC"

    def parse_mpc_args(self) -> dict:
        """
        Prepare MPC weights from loaded config file.

        :return: ready mpc args for controller
        :rtype: dict
        """
        import epic
        path = os.path.dirname(epic.__file__) + "/config/" + self.config_file

        with open(path, "r") as f:
            mpc_args = json.loads(f.read())
            mpc_args["model"] = self.my_cam
            mpc_args["dynamics"] = self.my_cam.d_feature_model

            mpc_args["Q"] = np.eye(self.my_cam.f_n * 2) * mpc_args["Q"]
            mpc_args["P"] = np.eye(self.my_cam.f_n * 2) * mpc_args["P"]
            mpc_args["R"] = np.eye(self.my_cam.m) * mpc_args["R"]
            mpc_args["ulb"] = np.array(mpc_args["ulb"])
            mpc_args["uub"] = np.array(mpc_args["uub"])
            mpc_args["xlb"] = np.array([-1, -1] * 5)
            mpc_args["xub"] = np.array([1, 1] * 5)
            mpc_args["xf"] = None
        return mpc_args

    def set_u_rot(self, u_rot):
        self.u_rot = u_rot

    def set_target_td(self, td):
        """
        Target relative position. Needed for correct velocity transformation.

        Here it is not a unit vector.
        """
        self.td = td

    def set_ematrices_with_neighbors(self):
        """
        Set essential matrix for the camera from known neighbors.
        """

        # Create relative poses
        # Matrices for F2
        R_l_to_f2, t_l_to_f2 = get_relative_pose(
            pose_from=self.cam1.get_formation_pose(), pose_to=self.my_cam.get_formation_pose()
        )
        R_f1_to_f2, t_f1_to_f2 = get_relative_pose(
            pose_from=self.cam2.get_formation_pose(), pose_to=self.my_cam.get_formation_pose()
        )
        R_cat = np.dstack((R_l_to_f2, R_f1_to_f2))
        t_cat = np.dstack((t_l_to_f2, t_f1_to_f2))

        # Get control input rotation (we track camera 1)
        self.u_rot = R_l_to_f2
        self.set_target_td(-t_l_to_f2)
        self.set_emat_matrices_with_rmat_tvec(R_cat, t_cat)

    def set_emat_matrices_with_emat(self, E):
        """
        Set Essential matrices.

        :param E: Essential matrices tensor
        :type E: 3d numpy array
        :raises ValueError: wrong shape of E
        """

        if len(E.shape) != 3:
            raise ValueError(
                "Expected `E` to have shape (3, 3, 2), " "got {}".format(E.shape)
            )

        # Set essential matrices
        E1 = E[:, :, 0]
        E2 = E[:, :, 1]

        # Save E1 and E2
        self.E1 = E1
        self.E2 = E2

        # Create V_Ei depending on which camera types we have
        self.V_E1 = set_emat_matrix_with_emat(self.E1, self.my_cam, self.cam1)
        self.V_E2 = set_emat_matrix_with_emat(self.E2, self.my_cam, self.cam2)
        return

    def set_emat_matrices_with_rmat_tvec(self, Rmats, tvecs):
        """
        Set essential matrices.

        :param Rmats: rotation matrices
        :type Rmats: np.ndarray
        :param tvecs: relative translation vectors
        :type tvecs: np.ndarray
        :raises ValueError: error if the shapes are not what is expected
        """

        if len(Rmats.shape) != 3 or len(tvecs.shape) != 3:
            raise ValueError(
                "Expected `R` and `t` to have shape (3, 3, 2) and (3, 1, 2) "
                "got {} and {}".format(Rmats.shape, tvecs.shape)
            )

        E1 = np.dot(Rmats[:, :, 0].reshape(3, 3), skew(tvecs[:, :, 0].reshape(3, 1)))  # ! I changed this order!
        E2 = np.dot(Rmats[:, :, 1].reshape(3, 3), skew(tvecs[:, :, 1].reshape(3, 1)))
        E_mat = np.dstack((E1, E2))
        self.set_emat_matrices_with_emat(E_mat)
        return

    def get_closest(self, point_list, obs_point):
        """
        Get closest list point to the observed point.

        :param point_list: list of intersection points, 2xn
        :type point_list: np.ndarray
        :param obs_point: observed point
        :type obs_point: np.ndarray
        :return: closest point
        :rtype: np.ndarray
        """
        point_list = point_list.reshape((2, -1))
        obs_point = obs_point.reshape((2, 1))
        dists = np.linalg.norm(point_list - obs_point, axis=0)
        return point_list[:, np.argmin(dists)].reshape((2, 1))

    def adjust_neighbor_data_length(self, data):
        """
        Adjust the length of the neighbor data to match the control prediction horizon.

        :param data: received data
        :type data: np.ndarray
        :return: adjusted data
        :rtype: np.ndarray
        """
        if self.method == "mpibvs":
            if data.shape[1] < self.predictive_controller.Nt - 1:
                # If received data is smaller than our control prediction horizon, we repeat the last point
                repeated_last = np.repeat(data[:, [-1]], self.predictive_controller.Nt - data.shape[1] - 1, axis=1)
                return np.concatenate((data, repeated_last), axis=1)
            elif data.shape[1] > self.predictive_controller.Nt - 1:
                # If the received data is larger than our control prediction horizon, we truncate it
                return data[:, 0:self.predictive_controller.Nt - 1]
            else:
                return data
        elif self.method == "ibvs":
            # Only first data point matters
            return data[:, [0]]

    def get_dynamic_intersection_points(self, matched_points_ag1, matched_points_ag2, my_points, neighbor_info_1, neighbor_info_2):
        """
        Calculate intersection points for the two curves, consider static neighbors.

        :param matched_points_ag1: _description_
        :type matched_points_ag1: _type_
        :param matched_points_ag2: _description_
        :type matched_points_ag2: _type_
        :param my_points: _description_
        :type my_points: _type_
        :return: _description_
        :rtype: _type_
        """
        my_matched_points = my_points[0:2, :]

        un_1 = self.adjust_neighbor_data_length(neighbor_info_1["u"])
        un_2 = self.adjust_neighbor_data_length(neighbor_info_2["u"])

        # Initialize data - neighbor
        xn1_dynamics = self.my_cam.get_generic_camera_integrator(neighbor_info_1["csi"], neighbor_info_1["dt"])
        Z0n_1 = neighbor_info_1["Z0"]
        xn_1_traj = matched_points_ag1.reshape((-1, 1))

        xn2_dynamics = self.my_cam.get_generic_camera_integrator(neighbor_info_2["csi"], neighbor_info_2["dt"])
        Z0n_2 = neighbor_info_2["Z0"]
        xn_2_traj = matched_points_ag2.reshape((-1, 1))

        # Log intersection indicese
        log_intrsct_pts_idx = []
        if un_1.shape[1] != un_2.shape[1]:
            print("Warning: Unmatched number of points in the control trajectories: ", un_1.shape[1], " / ", un_2.shape[1])

        intrsct_pts = np.empty((10, 0))
        all_intersect_points = {}
        for ui in range(un_1.shape[1] + 1):
            # Lift points to Veronese coordinates
            v_matched_points_ag1 = lift_image_points(xn_1_traj[:, -1].reshape((3, -1)))
            v_matched_points_ag2 = lift_image_points(xn_2_traj[:, -1].reshape((3, -1)))
            u_ag1_lines = np.dot(self.V_E1, v_matched_points_ag1)
            u_ag2_lines = np.dot(self.V_E2, v_matched_points_ag2)

            # Log the initial set of lines
            if ui == 0:
                log_u_ag1_lines = u_ag1_lines
                log_u_ag2_lines = u_ag2_lines

            # Get intersections
            step_intrsct_pts = np.empty((2, 0))
            for i in range(u_ag1_lines.shape[1]):

                if is_line(u_ag1_lines[:, i]) and is_line(u_ag2_lines[:, i]):
                    # Intersection of lines
                    intersection_points = get_lines_intersection(
                        u_ag1_lines[:, i], u_ag2_lines[:, i]
                    )
                else:
                    intersection_points = get_parabola_intersections(
                        u_ag1_lines[:, i],
                        u_ag2_lines[:, i],
                    )

                # Here we check if the intersection of two lines is valid
                if intersection_points is None:
                    print("No intersection point found...")
                    continue
                # Log this
                all_intersect_points[str(i)] = {'x': intersection_points, 'pt': my_matched_points[:, i], 'pt_1': matched_points_ag1[0:2, i], 'pt_2': matched_points_ag2[0:2, i]}

                # Get real points
                intersection_points = np.real(intersection_points)

                # ... and if so, we get the closest point to the current point (fair IBVS assumption)
                intersection_points = np.real(intersection_points)
                intersection = self.get_closest(
                    intersection_points, my_matched_points[:, i]
                )

                # Continue if no closest point was found...
                if intersection is None:
                    print("No closest point found...")
                    continue

                # Log first intersection
                if ui == 0:
                    log_intrsct_pts_idx.append(i)

                # Append intersection points that were found
                step_intrsct_pts = np.append(step_intrsct_pts, intersection, axis=1)

            # If we have less than 5 intersections, then we keep the previous intersection points for feasibility
            if step_intrsct_pts.shape[1] < 5:
                print("Not enough points found for intersection, repeating previous points...")
                step_intrsct_pts = intrsct_pts[:, -1]

            # Append step intersection
            intrsct_pts = np.append(intrsct_pts, step_intrsct_pts.reshape((10, -1), order="F"), axis=1)

            # Update next set of points - Neighbor 1
            if ui < un_1.shape[1]:
                features = xn_1_traj[:, -1].reshape((3, -1))
                fn = features[0:2, :]
                fn_plus_1 = xn1_dynamics(x0=fn.reshape((-1, 1), order="F"), p=ca.vertcat(un_1[:, ui].reshape((-1, 1)), Z0n_1[0].T))["xf"]
                xn = np.vstack((np.asarray(fn_plus_1).reshape((2, -1), order="F"), np.ones((1, np.asarray(fn_plus_1).reshape((2, -1)).shape[1])))).reshape((-1, 1))
                xn_1_traj = np.append(xn_1_traj, xn, axis=1)

                # Update next set of points - Neighbor 2
                features = xn_2_traj[:, -1].reshape((3, -1))
                fn = features[0:2, :]
                fn_plus_1 = xn2_dynamics(x0=fn.reshape((-1, 1), order="F"), p=ca.vertcat(un_2[:, ui].reshape((-1, 1)), Z0n_2[0].T))["xf"]
                xn = np.vstack((np.asarray(fn_plus_1).reshape((2, -1), order="F"), np.ones((1, np.asarray(fn_plus_1).reshape((2, -1)).shape[1])))).reshape((-1, 1))
                xn_2_traj = np.append(xn_2_traj, xn, axis=1)

        return un_1, intrsct_pts, log_intrsct_pts_idx, log_u_ag1_lines, log_u_ag2_lines, all_intersect_points

    def tranform_u_to_my_frame(self, u):
        """
        Transform control input from neighbor frame to my frame.

        :param u: control input from neighbor
        :type u: np.ndarray
        :return: control input in my frame
        :rtype: np.ndarray
        """
        u = u.reshape((6, -1))
        # Angular is fine, but linear needs to be changed, as it depends on td as well!
        # https://www.physics.usu.edu/Wheeler/ClassicalMechanics/CMRigidBodyDynamics.pdf, 2.2, add linear velocity to with cross product

        # Rotate each of the inputs to our reference frame
        vL = u[0:3, :]
        wL = u[3:, :]

        rel_vec = self.td
        VF_in_L = vL[0:3, :] + np.cross(wL.T, -rel_vec.T).T

        # Get desired linear and angular velocity in my frame

        u_rot = np.concatenate((self.u_rot @ VF_in_L.reshape(3, -1),
                                self.u_rot @ wL), axis=0)
        return u_rot

    def control(self, my_points, my_points_z, matched_points, **kwargs):
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
        matched_points_ag1 = matched_points[:, :, 0]
        matched_points_ag2 = matched_points[:, :, 1]
        my_matched_points = my_points[0:2, :]

        # Get reference for predictive controller
        un_1, intrsct_pts, intrsct_pts_idx, u_ag1_lines, u_ag2_lines, all_intersects = self.get_dynamic_intersection_points(
            matched_points_ag1, matched_points_ag2, my_points, kwargs["neighbor_info_1"], kwargs["neighbor_info_2"]
        )

        if self.method == "mpibvs":
            # We select neighbor 1 for control reference
            u_sp = self.tranform_u_to_my_frame(un_1)

            # Not predict, use just first set of intersct points
            self.predictive_controller.set_reference(intrsct_pts, u_sp=u_sp)

            # Control observations towards intersected points
            self.z0 = my_points_z[0][:, intrsct_pts_idx[0:5]].reshape(5, 1)

            u = self.predictive_controller.control(
                my_matched_points[:, intrsct_pts_idx[0:5]].reshape(10, 1, order="F"),
                z0=self.z0,
            )
            predicted_costs = self.predictive_controller.get_predicted_costs()
            if "skip_input" in kwargs.keys():
                u = np.zeros((6,))

            # Calculate error
            error = my_matched_points[:, intrsct_pts_idx[0:5]].reshape(10, 1, order="F") - intrsct_pts[:, [0]]

            # Controller trajectory
            self.control_trajectory = np.asarray(self.predictive_controller.u_pred).reshape(self.predictive_controller.Nt - 1, -1).T
            predicted_features = np.asarray(self.predictive_controller.x_pred).reshape(self.predictive_controller.Nt, -1).T
            predicted_features = predicted_features.reshape(2 * self.my_cam.f_n, self.predictive_controller.Nt, 1)

            # Update data struct
            data_struct = dict(
                {"u": u, "l1": u_ag1_lines, "l2": u_ag2_lines, "-x-": intrsct_pts}
            )
            data_struct.update(
                {
                    "x": my_matched_points[:, intrsct_pts_idx[0:5]].reshape(
                        10, 1, order="F"
                    ),
                    "ct": self.predictive_controller.get_last_solve_time(),
                    "J": self.predictive_controller.get_cost(),
                }
            )
            data_struct.update({"predicted_costs": predicted_costs})
            data_struct.update({"predicted_features": predicted_features})
            data_struct.update({"error": error})
        else:
            # Since we dont use predictive control, slice intersection points
            ref = intrsct_pts[0:self.my_cam.f_n * 2, [0]]
            # Use interaction matrix for controlling points towards intersections
            self.controller.set_reference(ref)

            # Control observations towards intersected points
            self.z0 = my_points_z[0][:, intrsct_pts_idx[0:5]].reshape(5, 1)

            # Get control inpuit
            u, data = self.controller.control(
                my_matched_points[:, intrsct_pts_idx[0:5]].reshape(10, 1, order="F"),
                self.z0,
            )
            self.control_trajectory = u.reshape((-1, 1))

            # Clip input
            u = np.clip(np.asarray(u).reshape((6,)), self.mpc_args["ulb"], self.mpc_args["uub"])

            if "skip_input" in kwargs.keys():
                u = np.zeros((6,))

            error = self.controller.error

            data_struct = dict(
                {"u": u, "l1": u_ag1_lines, "l2": u_ag2_lines, "-x-": ref.reshape((-1, 1, 1)), "error": error}
            )
            data_struct.update({"x": my_matched_points[:, intrsct_pts_idx[0:5]].reshape(10, 1, order="F")})
            data_struct.update(data)

        # Create dictionary to plot outside
        data_struct.update({'all_intersections': all_intersects})
        return data_struct

    def empty_log(self) -> dict:
        """
        Empty dictionary for logged variables.

        :return: empty dictionary with same footprint as control
        :rtype: dict
        """
        empty_log = {
                        "u": np.empty((self.my_cam.m, 0)),
                        "l1": np.empty((6, self.my_cam.f_n, 0)),
                        "l2": np.empty((6, self.my_cam.f_n, 0)),
                        "error": np.empty((2 * self.my_cam.f_n, 0)),
                        "x": np.empty((self.my_cam.f_n * 2, 0)),
                        "xr": np.empty((self.my_cam.f_n * 2)),
                        "J": np.empty((1, 0)),
                        "ct": np.empty((1, 0)),
                        "all_intersections": {},
                        "predicted_costs": {"x": np.empty((1, 0)), "u": np.empty((1, 0))},
                    }
        if self.method == "mpibvs":
            empty_log.update(
                {
                    "predicted_features": np.empty(
                        (self.my_cam.f_n * 2, self.predictive_controller.Nt, 0)
                    ),
                    "-x-": np.empty(
                        (2 * self.my_cam.f_n, self.predictive_controller.Nt, 0)
                    ),
                }
            )
        else:
            empty_log.update(
                {
                    "-x-": np.empty(
                        (2 * self.my_cam.f_n, 1, 0)
                    ),
                }
            )
        return empty_log
