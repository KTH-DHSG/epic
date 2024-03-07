import numpy as np
import casadi as ca

class BasicTemplate(object):

    def __init__(self, my_cam):
        self.control_trajectory = None
        self.Nt = 5
        self.my_cam = my_cam
        self.predictive_dynamics = self.my_cam.get_generic_camera_integrator(self.my_cam.csi, self.my_cam.dt)

    def control(self, **kwargs):

        ctl_input = kwargs["u"]
        ctl_dict = {"u": ctl_input}

        # Set control trajectory
        if ctl_input.shape[1] == 1:
            ctl_traj = np.repeat(ctl_input, self.Nt - 1, axis=1)
            self.control_trajectory = ctl_traj

        # Predict feature motion
        if kwargs["my_points"] is not None:
            propagated_features = self.propagate_feature_motion(self.control_trajectory, kwargs["my_points"], kwargs["my_points_z"])
            ctl_dict.update({"propagated_features": propagated_features})
        return ctl_dict

    def propagate_feature_motion(self, u, my_points, my_points_z):

        Z = my_points_z
        xn_traj = my_points.reshape((-1, 1))

        for ui in range(u.shape[1]):
            features = xn_traj[:, -1].reshape((3, -1))
            fn = features[0:2, :]
            # Features are propagated with same depth, so it makes sense that they are not perfectly aligned with prediction
            fn_plus_1 = self.predictive_dynamics(x0=fn.reshape((-1, 1), order="F"), p=ca.vertcat(u[:, ui].reshape((-1, 1)), Z[0].T))["xf"]
            xn = np.vstack((np.asarray(fn_plus_1).reshape((2, -1), order="F"), np.ones((1, np.asarray(fn_plus_1).reshape((2, -1)).shape[1])))).reshape((-1, 1))
            xn_traj = np.append(xn_traj, xn, axis=1)

        return xn_traj[:self.my_cam.f_n * 2, :]

    def empty_log(self):
        return {"u": np.empty((6, 0)),
                "propagated_features": np.empty((self.my_cam.f_n * 2, self.Nt, 0))
                }
