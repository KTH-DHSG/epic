import casadi as ca
import numpy as np

from epic.utils.geometry import get_closest_relative_pose
from scipy.io import savemat

# Import mixins
import epic.model._cartesian as _cartesian
import epic.model._imaging as _imaging

DEBUG = False


class GeneralizedCamera(_cartesian.CartesianSpace, _imaging.Imaging):
    def __init__(
        self,
        name,
        model="perspective",
        tracked_features=5,
        h=0.01,
        state=None,
        K=None,
        **kwargs
    ):
        """
        6DoF Camera Kinematic class.

        :param iface: select interface for casadi or acados control integration
        :type iface: str
        :param h: sampling time of the discrete system, defaults to 0.01
        :type h: float, optional
        """

        self.name = name
        self.type = model
        self.dt = h

        # State and camera parameters
        self.n = 7  # State is: position, quaternion [px, py, pz, qx, qy, qz, qw]
        self.m = 6  # Input is: linear and angular velocity [vx, vy, vz, wx, wy, wz]
        self.res_x = None  # Camera X-axis resolution, in pixels
        self.res_y = None  # Camera Y-axis resolution, in pixels
        self.f_n = tracked_features  # Number of tracked features - defined dimension of feature model
        self.x_d = np.zeros((7, 1))

        # Controller
        self.controller = None
        self.neighbors = None
        self.integrator = "cvodes"

        # Logger
        self.log = None

        if "d" in kwargs and "p" in kwargs:
            self.cam_p = kwargs["p"]
            self.cam_d = kwargs["d"]
        else:
            # Set defaults for each camera model
            if self.type == "perspective":
                self.cam_p = 0
                self.cam_d = 0
            elif self.type == "hyperbolic":
                self.cam_p = 0.3
                self.cam_d = 0.9
            elif self.type == "parabolic":
                self.cam_p = 0.05
            elif self.type == "distortion":
                self.cam_p = 0.05
                self.cam_d = 0.8

        if state is None:
            cam_origin = np.array([0, 0, 0, 0, 0, 0, 1])
            self.set_state(cam_origin)
        else:
            self.set_state(state)

        self.set_intrinsics(K)

        self.model = self.camera_kinematics
        self.feature_model = self.feature_kinematics
        self.set_discrete_kinematics()

    def __str__(self):
        return self.name

    def set_reference(self, x_d):
        """
        Set camera reference state

        :param x_d: reference state
        :type x_d: np.ndarray
        """
        self.x_d = x_d

    def get_reference(self) -> np.ndarray:
        """
        Get reference state

        :return: reference state
        :rtype: _type_
        """
        return self.x_d

    def set_formation_pose(self, x_d: np.ndarray):
        """
        Set formation position in the world frame. Will be used
        to calculate the desired epipolar matrices.

        :param x_d: _description_
        :type x_d: np.ndarray
        """
        self.formation_x_d = x_d

    def get_formation_pose(self) -> np.ndarray:
        """
        Get reference formation pose

        :return: reference state
        :rtype: np.ndarray
        """
        return self.formation_x_d

    def get_generic_camera_integrator(self, csi, dt):
        """
        Provide integrator for generic camera given csi parameter

        :param csi: distortion coefficient
        :type csi: float
        :param dt: sampling time
        :type dt: float
        :return: integrator for generic camera
        :rtype: ca.integrator
        """
        f0 = ca.MX.sym("f0", self.f_n * 2, 1)  # [ux, uy]
        Z0 = ca.MX.sym("Z0", self.f_n, 1)  # [Z]
        u = ca.MX.sym("u", self.m, 1)
        dae = {
            "x": f0,
            "ode": self.feature_model(f0, u, Z0, csi),
            "p": ca.vertcat(u, Z0),
        }
        options = {
            "abstol": 1e-5,
            "reltol": 1e-9,
            "max_num_steps": 100,
            "tf": dt,
        }
        return ca.integrator(
            "integrator", "cvodes", dae, options
        )

    def set_discrete_kinematics(self, integrator="cvodes"):
        """
        Helper function to populate a 6DoF Camera Kinematics.
        """
        # Integrator options:
        integrator_options = {
            "abstol": 1e-5,
            "reltol": 1e-9,
            "max_num_steps": 100,
            "tf": self.dt,
        }
        # Camera kinematics
        u = ca.MX.sym("u", self.m, 1)
        x0 = ca.MX.sym("x0", self.n, 1)

        # Currently we are using a simple RK4 integrator, compare with cvodes
        if integrator == "rk4":
            self.d_model = self.rk4_camera_integrator(self.model, x0, u)
        elif integrator == "cvodes":
            dae = {"x": x0, "ode": self.model(x0, u), "p": u}
            self.d_model = ca.integrator(
                "integrator", "cvodes", dae, integrator_options
            )
        else:
            raise ValueError("Integrator not implemented")

        # Feature model
        f0 = ca.MX.sym("f0", self.f_n * 2, 1)  # [ux, uy]
        Z0 = ca.MX.sym("Z0", self.f_n, 1)  # [Z]
        csi = ca.MX.sym("csi", 1, 1)
        u = ca.MX.sym("u", self.m, 1)
        # self.d_feature_model = self.rk4_feature_integrator(
        #     self.feature_model, f0, u, Z0, csi
        # )
        # feature_model_euler = f0 + self.dt * self.feature_kinematics(f0, u, Z0, self.csi)
        self.d_feature_model = ca.Function(
            "Fkinematics",
            [f0, u, Z0],
            [f0 + self.dt * self.get_interaction_matrix(f0, Z0) @ u],
        )
        dae = {
            "x": f0,
            "ode": self.feature_model(f0, u, Z0),
            "p": ca.vertcat(u, Z0),
        }
        self.d_feature_model_int = ca.integrator(
            "integrator", "cvodes", dae, integrator_options
        )
        self.c_feature_model = ca.Function(
            "Fkinematics", [f0, u, Z0], [self.feature_model(f0, u, Z0)]
        )

        # IBRC model
        self.relative_navigation_model()

        return

    def get_interaction_matrix(self, features, features_depth, csi=None):
        """
        Get camera interaction matrix wrt the provided features.

        :param features: image features
        :type features: np.ndarray
        :param features_depth: depth of each feature, ordered
        :type features_depth: np.ndarray
        :param csi: distortion parameter, defaults to None
        :type csi: float, optional
        :return: interaction matrix
        :rtype: np.ndarray
        """
        if type(features) is ca.casadi.MX:
            L = ca.MX(self.f_n * 2, 6)
        else:
            L = np.zeros((self.f_n * 2, 6))
        for i in range(self.f_n):
            x = features[i * 2: i * 2 + 2]
            xs = x[0]
            ys = x[1]
            r = features_depth[i]

            if csi is None:
                csi = self.csi

            if type(features) is ca.casadi.MX:
                Li = ca.MX(2, 6)
                ups = ca.sqrt(1 + (1 - csi**2) * (xs**2 + ys**2))
            else:
                Li = np.zeros((2, 6))
                ups = np.sqrt(1 + (1 - csi**2) * (xs**2 + ys**2))

            # Matrix entries
            l11 = -(1 + xs**2 * (1 - csi * (ups + csi)) + ys**2) / (r * (ups + csi))
            l12 = xs * ys * csi / r
            l13 = xs * ups / r
            l14 = xs * ys
            l15 = -((1 + xs**2) * ups - ys**2 * csi) / (ups + csi)
            l16 = ys

            l21 = xs * ys * csi / r
            l22 = -(1 + xs**2 + ys**2 * (1 - csi * (ups + csi))) / (r * (ups + csi))
            l23 = ys * ups / r
            l24 = ((1 + ys**2) * ups - xs**2 * csi) / (ups + csi)
            l25 = -xs * ys
            l26 = -xs

            if type(features) is ca.casadi.MX:
                Li = ca.vertcat(ca.horzcat(l11, l12, l13, l14, l15, l16), ca.horzcat(l21, l22, l23, l24, l25, l26))
            else:
                Li = np.array([[l11, l12, l13, l14, l15, l16], [l21, l22, l23, l24, l25, l26]]).reshape((2, 6))

            L[i * 2: i * 2 + 2, :] = Li

        return L

    def feature_kinematics(self, features, u, features_depth, csi=None):
        """
        Feature movement in the camera frame.

        :param features: state [ux, uy]*f_n
        :type features: ca.MX or ca.DM
        :param u: camera body frame velocity
        :type u: ca.MX or ca.DM
        :param featured_depth: depth for each feature
        :type featured_depth: ca.MX or ca.DM, [Z]*f_n
        :param csi: distortion model parameter csi
        :type csi: float
        :return: time derivative of feature position
        :rtype: ca.MX or ca.DM
        """

        L = self.get_interaction_matrix(features, features_depth, csi)
        # NOTE: propagation of Z is not done (assumed to be measured outside)
        fdot = ca.mtimes(L, u)

        return fdot

    def rk4_feature_integrator(self, kinematics, x0, u, p, csi):
        """
        Runge-Kutta 4th Order discretization.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state at next step
        :rtype: ca.MX
        """

        x = x0
        csi = self.csi
        k1 = kinematics(x, u, p, csi)
        k2 = kinematics(x + self.dt / 2 * k1, u, p, csi)
        k3 = kinematics(x + self.dt / 2 * k2, u, p, csi)
        k4 = kinematics(x + self.dt * k3, u, p, csi)
        xdot = x0 + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Set function options
        fun_options = {"jit": False, "jit_options": {"flags": ["-O2"]}}
        rk4 = ca.Function("RK4", [x0, u, p], [xdot], fun_options)

        return rk4

    def set_controller(self, ctl_type: str, neighbors: list, config: str, tracking: bool = False, **kwargs):
        """
        Set controller.

        :param controller: controller
        :type controller: Controller
        """
        self.neighbors = neighbors
        self.ctl_type = ctl_type.lower()
        if self.ctl_type == "basic":
            import epic.control.basic as ctl
            self.controller = ctl.BasicTemplate(self, **kwargs)
        elif self.ctl_type == "ibrc":
            import epic.control.ibrc as ctl
            self.controller = ctl.IBRC(self, neighbors[0], config_file=config, **kwargs)
        elif self.ctl_type == "ibfc":
            import epic.control.ibfc as ctl
            self.controller = ctl.IBFC(self, neighbors[0], neighbors[1], config_file=config, **kwargs)
        elif self.ctl_type == "bearing_rigidity":
            raise NotImplementedError("Bearing Rigidity controller not publicly available. \
                                       Please contact the repository maintainer.")
        else:
            raise ValueError("Unknown controller type on 'set_controller' method.")
        print("Controller set for ", self)

    def control(self, **kwargs) -> None:
        """
        Control step.
        """

        if self.controller is None:
            raise ValueError("Controller not set!")

        if self.ctl_type == "ibrc":
            assert "my_points" in kwargs, "Features not provided to IBRC ctl of " + str(self)
            assert "my_points_z" in kwargs, "Feature Depth not provided to IBRC ctl of " + str(self)
            assert "matched_points" in kwargs, "Neighbor not provided to IBRC ctl of " + str(self)
            assert "leader" in kwargs, "Leader not provided to IBRC ctl of " + str(self)
            kwargs['my_points'] = kwargs['my_points'] / kwargs['my_points'][2, :]
            kwargs['matched_points'] = kwargs['matched_points'] / kwargs['matched_points'][2, :]
            if "t1_2" not in kwargs:
                # Calculate estimate of the relative position
                rng = self.controller.get_range_to(kwargs['leader'])
                p1_ray = self.get_image_ray_from_calibrated_points(kwargs['my_points'], kwargs['my_points_z'])
                p2_ray = self.get_image_ray_from_calibrated_points(kwargs['matched_points'], kwargs['neighbor_info']['Z0'], self.neighbors[0])
                Rlf, t = get_closest_relative_pose(p2_ray[:2, :].T, p1_ray[:2, :].T, self.controller.Rd, self.controller.td)
                t12 = t * rng
                self.estimated_t = t12
                self.estimated_R = Rlf
                self.controller.set_RLF(Rlf.T)
            else:
                t12 = kwargs['t1_2']
            ctl_dict = self.controller.control(t12=t12, **kwargs)
        elif self.ctl_type == "ibfc" or self.ctl_type == "basic" or self.ctl_type == "ibvs":
            ctl_dict = self.controller.control(**kwargs)
        elif self.ctl_type == "bearing_rigidity":
            ctl_dict = self.controller.control(**kwargs)

        else:
            raise ValueError("Controller type not implemented in generalized camera class.")

        # Apply control input
        self.apply_control(ctl_dict["u"])

        # Log variables
        self.log_variables(ctl_dict)

        # return control dictionary if needed
        return ctl_dict

    def apply_control(self, u):
        """
        Update camera state given a control input.

        :param u: control input
        :type u: numpy array, 6x1
        """

        if self.integrator == "rk4":
            self.state = np.asarray(self.d_model(self.state, u)).reshape(
                7,
            )
        elif self.integrator == "cvodes":
            self.state = np.asarray(self.d_model(x0=self.state, p=u)["xf"]).reshape(
                7,
            )
        # Re-normalize quaternion
        self.state[3:7] = self.state[3:7] / np.linalg.norm(self.state[3:7])

        return self.state

    def set_log_variables(self) -> None:
        """
        Generate empty log.
        """
        self.log = dict()
        self.log['state'] = self.state.reshape((7, 1))
        self.log['pos_error'] = np.empty((3, 0))
        ctl_empty_log = self.controller.empty_log()
        for key, value in ctl_empty_log.items():
            self.log[key] = value
        return

    def log_variables(self, ctl_variables: dict) -> None:
        """
        Log variables.

        :param ctl_dict: dictionary of variables to log
        :type ctl_dict: dict
        """
        if self.log is None:
            self.set_log_variables()
        self.log['state'] = np.append(self.log['state'], self.state.reshape((7, 1)), axis=1)
        self.log['pos_error'] = np.append(self.log['pos_error'], self.get_cartesian_error().reshape((3, 1)), axis=1)
        for key, value in ctl_variables.items():
            if key == 'all_intersections':
                self.log.update({'all_intersections': ctl_variables['all_intersections']})
                continue
            if key == 'predicted_costs':
                self.log.update({'predicted_costs': ctl_variables['predicted_costs']})
                continue

            if key in self.log:
                if len(self.log[key].shape) == 3:
                    # Log matrices
                    self.log[key] = np.append(self.log[key], value.reshape(value.shape[0], value.shape[1], 1), axis=2)
                else:
                    # Log vectors
                    self.log[key] = np.append(self.log[key], np.asarray(value).reshape(-1, 1), axis=1)

    def save_log(self, base_path='data/') -> None:
        """
        Save log to .mat file
        """
        savemat(base_path + str(self) + '_log.mat', self.log)
