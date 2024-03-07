import casadi as ca
import numpy as np
import pyquaternion as pq

from epic.utils.geometry import r_mat, skew, xi_mat


class CartesianSpace:

    def get_cartesian_error(self):
        """
        Calculate state error
        :return: pose error
        :rtype: np.ndarray
        """
        try:
            p_error = self.get_position() - self.x_d[0:3].reshape((3,))
        except Exception:
            p_error = np.ones(3) * np.inf
        return p_error

    def set_position(self, position):
        """
        Set Camera position.

        :param position: position in world coordinates (X Y Z)
        :type position: numpy array or list
        """

        self.state[0] = position[0]
        self.state[1] = position[1]
        self.state[2] = position[2]

    def get_position(self):
        """
        Get camera position

        :return: camera position
        :rtype: numpy array
        """
        return self.state[0:3]

    def set_attitude_quat(self, quat):
        """
        Set attitude quaternion. Format: x y z w

        :param quat: attitude quaternion
        :type quat: numpy array
        """

        self.state[3] = quat[0]
        self.state[4] = quat[1]
        self.state[5] = quat[2]
        self.state[6] = quat[3]

    def get_attitude_quat(self):
        """
        Get attitude quaternion

        :return: [description]
        :rtype: [type]
        """
        return self.state[3:]

    def get_attitude_rmat(self):
        """
        Returns the attitude in a Rotation matrix format

        :return: attitude rotation matrix
        :rtype: numpy array
        """
        q = self.get_attitude_quat()
        rmat = r_mat(q)

        return rmat

    def get_attitude_rpy(self):
        """
        Get the attitude in roll-pitch-yaw radians

        :return: euler angles in xyz format
        :rtype: numpy array
        """
        rmat = self.get_attitude_rmat()
        euler = rmat.as_euler("xyz")

        return euler

    def set_state(self, state):
        """
        Set camera state. [p, q]

        :param state: state (position and att quaternion)
        :type state: numpy array, 4x1, quat is scalar-last
        """
        self.state = state

    def get_state(self):
        """
        Get Camera state (state), [p, q].

        :return: state (position and att quaternion)
        :rtype: numpy array, 4x1, quat is scalar-last
        """
        return self.state

    def camera_kinematics(self, x, u):
        """
        Camera 6dof kinematics.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state time derivative
        :rtype: ca.MX
        """

        # State extraction
        q = x[3:] / ca.norm_2(x[3:])

        # 3D Force
        v = u[0:3]

        # 3D Torque
        w = u[3:]

        # Model
        pdot = ca.mtimes(r_mat(q), v)
        qdot = ca.mtimes(xi_mat(q), w) / 2

        dxdt = [pdot, qdot]

        return ca.vertcat(*dxdt)

    def rk4_camera_integrator(self, kinematics, x0, u):
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

        k1 = kinematics(x, u)
        k2 = kinematics(x + self.dt / 2 * k1, u)
        k3 = kinematics(x + self.dt / 2 * k2, u)
        k4 = kinematics(x + self.dt * k3, u)
        xdot = x0 + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Normalize quaternion: TODO(Pedro-Roque): check how to normalize cleanly
        xdot[3:] = xdot[3:] / ca.norm_2(xdot[3:])

        # Set function options
        fun_options = {"jit": False, "jit_options": {"flags": ["-O2"]}}
        rk4 = ca.Function("RK4", [x0, u], [xdot], fun_options)

        return rk4

    def get_range_step(self, r0):
        """
        Get range step.

        :param r0: range
        :type r0: ca.MX
        :return: range step
        :rtype: ca.MX
        """

        self.r_step = ca.horzcat(-ca.DM.eye(3), skew(r0))
        return self.r_step

    def relative_navigation_model(self):
        """
        Relative navigation model considering range and features.
        """

        # IBVS model
        f0 = ca.MX.sym("f0", self.f_n * 2, 1)  # [ux, uy]
        Z0 = ca.MX.sym("Z0", self.f_n, 1)  # [Z]
        t0 = ca.MX.sym("rng", 3, 1)
        x0 = ca.vertcat(t0, f0)
        u = ca.MX.sym("u", self.m, 1)

        # Leader data
        uL = ca.MX.sym("u", self.m, 1)
        RLF = ca.MX.sym("R_L->F", 3, 3)

        L_tilde = ca.vertcat(
            self.get_range_step(t0), self.get_interaction_matrix(f0, Z0)
        )

        # Leader contribution
        L_contrib = ca.horzcat(RLF, ca.DM.zeros((3, 3))) @ uL
        L_contrib = ca.vertcat(L_contrib, ca.DM.zeros(self.f_n * 2, 1))
        self.euler_rel_nav = ca.Function(
            "RNav", [x0, u, Z0, uL, RLF], [ca.vertcat(t0, f0) + (L_tilde @ u + L_contrib) * self.dt]
        )

    def get_random_state(self, lp, up):
        """
        Get a random camera state given position lower and upper bounds

        :param lp: lower bound on camera position
        :type lp: numpy array 3x1
        :param up: upper bound on camera position
        :type up: numpy array 3x1
        :return: a random camera state (position+quaternion)
        :rtype: numpy array, 7x1
        """

        x = np.random.uniform(low=lp[0], high=up[0])
        y = np.random.uniform(low=lp[1], high=up[1])
        z = np.random.uniform(low=lp[2], high=up[2])

        q = pq.Quaternion.random()
        state = np.array([x, y, z, q[1], q[2], q[3], q[0]])

        return state
