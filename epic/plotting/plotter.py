import numpy as np
import matplotlib.pyplot as plt

from epic.utils.geometry import r_mat, get_relative_pose


class Plotter(object):
    """
    Environment plotting class

    :param object: _description_
    :type object: _type_
    """
    def __init__(self, cameras: list, features: np.array) -> None:
        self.camera_list = cameras
        self.world_points = features

        # Create environment figure
        plt.ion()

        # Check how many subplots are needed
        if len(self.camera_list) > 3:
            self.subplot_rows = 3
            self.subplot_cols = 4
        else:
            self.subplot_rows = 1
            self.subplot_cols = len(self.camera_list)
        plt.show(block=False)

    def plot_environment(self):
        """
        Plot environment with camera frames and image planes.

        :return: matplotlib figure with environment plot
        :rtype: Figure
        """
        # Plot world points
        if not hasattr(self, "env_ax"):
            fig = plt.figure()
            self.env_ax = fig.add_subplot(111, projection="3d")

        self.env_ax.cla()
        self.env_ax.scatter(
            self.world_points[0, :],
            self.world_points[1, :],
            self.world_points[2, :],
            color="k",
            s=2,
        )

        # Plot camera frames
        for camera in self.camera_list:
            self.plot_camera(camera, self.env_ax)
        plt.show(block=False)
        plt.pause(0.001)

    def plot_camera(self, camera, ax):
        """
        Plot camera coordinate frame on axis ax.

        :param ax: matplotlib axis
        :type ax: Axes3d
        :return: updated matplotlib axis with camera frame
        :rtype: Axes3d
        """

        rmat = camera.get_attitude_rmat()
        pos = camera.get_position()

        ax.plot(
            [pos[0], pos[0] + rmat[0, 0]],
            [pos[1], pos[1] + rmat[1, 0]],
            zs=[pos[2], pos[2] + rmat[2, 0]],
            color="r",
        )
        ax.plot(
            [pos[0], pos[0] + rmat[0, 1]],
            [pos[1], pos[1] + rmat[1, 1]],
            zs=[pos[2], pos[2] + rmat[2, 1]],
            color="g",
        )
        ax.plot(
            [pos[0], pos[0] + rmat[0, 2]],
            [pos[1], pos[1] + rmat[1, 2]],
            zs=[pos[2], pos[2] + rmat[2, 2]],
            color="b",
        )

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)

        return ax

    def plot_image_planes(self, features=None, pt_type="normalized", lb=-.5, ub=.5):
        """
        Plot image-plane for specified camera.

        :param ax: matplotlib axis
        :type ax: Axis
        :param w_points_h: world points in homogeneous coordinates
        :type w_points_h: np.ndarray
        :param pt_type: point units (normalized or pixel), defaults to "normalized"
        :type pt_type: str, optional
        :raises TypeError: wrong point type
        :return: updated matplotlib axis with camera frame
        :rtype: Axes3d
        """

        if not hasattr(self, "image_plane_ax"):
            fig = plt.figure()
            self.image_plane_ax = {}
            for i, camera in enumerate(self.camera_list):
                self.image_plane_ax[camera.name] = fig.add_subplot(self.subplot_rows, self.subplot_cols, i + 1)

        for i, camera in enumerate(self.camera_list):
            # Clear axis
            self.image_plane_ax[camera.name].cla()
            if pt_type == "normalized":
                _, f, _ = camera.get_visible_points(self.world_points, camera.state)
                if features is not None:
                    _, t_f, _ = camera.get_visible_points(features, camera.state)
            else:
                f, _, _ = camera.get_visible_points(self.world_points, camera.state)
                if features is not None:
                    t_f, _, _ = camera.get_visible_points(features, camera.state)

            if pt_type == "normalized":
                x = f[0, :]
                y = f[1, :]
                self.image_plane_ax[camera.name].scatter(x, y, color="k")
                if features is not None:
                    x_f = t_f[0, :]
                    y_f = t_f[1, :]
                    self.image_plane_ax[camera.name].scatter(x_f, y_f, color="r")
                self.image_plane_ax[camera.name].set_xlim(lb, ub)
                self.image_plane_ax[camera.name].set_ylim(lb, ub)
                self.image_plane_ax[camera.name].grid()
                self.image_plane_ax[camera.name].set_aspect("equal")
            self.image_plane_ax[camera.name].set_xlabel(camera.name + " image-plane", loc="center")
            self.invert_camera_planes()
        plt.show(block=False)

    def plot_mpc_costs(self):
        """
        Plot image-plane for specified camera.

        :param ax: matplotlib axis
        :type ax: Axis
        :param w_points_h: world points in homogeneous coordinates
        :type w_points_h: np.ndarray
        :param pt_type: point units (normalized or pixel), defaults to "normalized"
        :type pt_type: str, optional
        :raises TypeError: wrong point type
        :return: updated matplotlib axis with camera frame
        :rtype: Axes3d
        """

        if not hasattr(self, "mpc_costs_ax"):
            fig = plt.figure()
            self.mpc_costs_ax = {}
            for i, camera in enumerate(self.camera_list):
                self.mpc_costs_ax[camera.name] = fig.add_subplot(self.subplot_rows, self.subplot_cols, i + 1)

        legend = []
        for i, camera in enumerate(self.camera_list):
            # Skip leader
            if "error" not in camera.log:
                continue

            # Append name to legend and plot it
            self.mpc_costs_ax[camera.name].cla()
            self.plot_mpc_cost(camera, self.mpc_costs_ax[camera.name], camera.log['error'])

            self.mpc_costs_ax[camera.name].set_xlabel("time [s]")
            self.mpc_costs_ax[camera.name].set_ylabel("MPC Cost")
            if camera.ctl_type == "ibfc":
                legend = ['ef']
            elif camera.ctl_type == "ibrc":
                legend = ['ep', 'ef']
            self.mpc_costs_ax[camera.name].legend(legend)
        plt.pause(0.001)

    def plot_mpc_cost(self, camera, ax, cost):
        """
        Plot cost

        :param camera: camera object
        :type camera: GeneralizedCamera
        :param ax: matplotlib axis handle
        :type ax: Axis
        :param cost: cost vector
        :type cost: np.ndarray
        """

        time = np.linspace(0, cost.shape[1] * camera.dt, cost.shape[1])
        if camera.ctl_type == "ibrc":
            e_p = cost[0, :].reshape(-1, time.shape[0])
            e_f = np.sum(cost[1:, :].reshape(-1, time.shape[0]), axis=0)
            ax.plot(time, e_p.reshape(time.shape), 'r',
                    time, e_f.reshape(time.shape), 'b')
        elif camera.ctl_type == "ibfc":
            e_f = np.sum(cost[0:, :].reshape(-1, time.shape[0]), axis=0)
            ax.plot(time, e_f.reshape(time.shape), 'b')

    def invert_camera_planes(self):
        """
        Invert camera planes to match observation.
        """

        for i, camera in enumerate(self.camera_list):
            self.image_plane_ax[camera.name].invert_yaxis()

    def plot_predicted_intersections(self, camera_list: list = None, step = None):
        """
        Plot predicted intersections of quadric curves

        :param camera_list: cameras to be plot, defaults to None
        :type camera_list: list, optional
        """

        if not hasattr(self, "image_plane_ax"):
            fig = plt.figure()
            self.image_plane_ax = {}
            for i, camera in enumerate(self.camera_list):
                self.image_plane_ax[camera.name] = fig.add_subplot(self.subplot_rows, self.subplot_cols, i + 1)

        for i, camera in enumerate(self.camera_list):
            # Skip leader
            if "-x-" not in camera.log:
                continue

            # Append name to legend and plot it
            # self.image_plane_ax[camera.name].cla()
            if step is None:
                self.plot_predicted_intersection_trajectories(self.image_plane_ax[camera.name], camera.log['-x-'][:, :, -1])
            else:
                self.plot_predicted_intersection_trajectories(self.image_plane_ax[camera.name], camera.log['-x-'][:, :, step])

            # self.predicted_intersections_ax[camera.name].set_xlabel("Y axis [u]")
            # self.predicted_intersections_ax[camera.name].set_ylabel("X axis [u]")
        plt.pause(0.001)

    def plot_predicted_intersection_trajectories(self, ax, intersections_traj):
        """
        Plot predicted trajectories

        :param camera: Generalized Camera
        :type camera: GeneralizedCamera
        :param ax: _description_
        :type ax: _type_
        :param x_traj: _description_
        :type x_traj: _type_
        """
        colors = ['r', 'g', 'b', 'c', 'm']
        for i in range(intersections_traj.shape[1]):
            step_intersection = intersections_traj[:, i].reshape(2, -1, order="F")
            for j in range(step_intersection.shape[1]):
                ax.scatter(step_intersection[0, j], step_intersection[1, j], color=colors[j], marker="x")

    def plot_mpc_predicted_feature_trajectories(self, camera_list: list = None, step = None):
        """
        Plot feature trajectories predicted by the MPC controller.

        :param camera_list: cameras to be plot, defaults to None
        :type camera_list: list, optional
        """

        if not hasattr(self, "image_plane_ax"):
            fig = plt.figure()
            self.image_plane_ax = {}
            for i, camera in enumerate(self.camera_list):
                self.image_plane_ax[camera.name] = fig.add_subplot(self.subplot_rows, self.subplot_cols, i + 1)

        for i, camera in enumerate(self.camera_list):
            # Skip leader
            if "predicted_features" not in camera.log:
                continue

            # Append name to legend and plot it
            # self.image_plane_ax[camera.name].cla()
            if step is None:
                self.plot_mpc_predicted_features(self.image_plane_ax[camera.name], camera.log['predicted_features'][:, :, -1])
            else:
                self.plot_mpc_predicted_features(self.image_plane_ax[camera.name], camera.log['predicted_features'][:, :, step])

            # self.predicted_intersections_ax[camera.name].set_xlabel("Y axis [u]")
            # self.predicted_intersections_ax[camera.name].set_ylabel("X axis [u]")
        plt.pause(0.001)

    def plot_mpc_predicted_features(self, ax, features):
        """
        Plot predicted trajectories

        :param camera: Generalized Camera
        :type camera: GeneralizedCamera
        :param ax: _description_
        :type ax: _type_
        :param x_traj: _description_
        :type x_traj: _type_
        """
        colors = ['r', 'g', 'b', 'c', 'm']
        for i in range(features.shape[1]):
            step_intersection = features[:, i].reshape(2, -1, order="F")
            for j in range(step_intersection.shape[1]):
                ax.scatter(step_intersection[0, j], step_intersection[1, j], color=colors[j], marker="o")

    def plot_selected_image_planes(self, camera_list, lb=-.5, ub=.5):
        """
        Plot image-plane for specified cameras.

        :param camera_list: list of cameras, maximum 3
        :type camera_list: list
        """
        if not hasattr(self, "selected_image_plane"):
            fig_first_run = True
            fig = plt.figure()
            self.selected_image_plane = {}
            # for i, camera in enumerate(camera_list):
            #     self.selected_image_plane[camera.name] = fig.add_subplot(1, len(camera_list), i + 1)
            fig.tight_layout()
        else:
            fig_first_run = False

        if len(camera_list) > 3:
            raise ValueError("Maximum 3 image planes can be plotted at once.")

        for i, camera in enumerate(camera_list):
            # Create image plane
            if fig_first_run:
                self.selected_image_plane[camera.name] = fig.add_subplot(1, len(camera_list), i + 1)

            # Plot feature trajectories
            if camera.ctl_type == "ibrc":
                self.plot_feature_trajectories(self.selected_image_plane[camera.name], camera.log['x'][3:, :])
            else:
                self.plot_feature_trajectories(self.selected_image_plane[camera.name], camera.log['x'])

            # Plot image plane at the end of the trajectory
            _, f, _ = camera.get_visible_points(self.world_points, camera.state)
            x = f[0, :]
            y = f[1, :]
            self.selected_image_plane[camera.name].scatter(x, y, color="k")
            self.selected_image_plane[camera.name].set_xlim(lb, ub)
            self.selected_image_plane[camera.name].set_ylim(lb, ub)

            # Plot curves at last state
            if camera.ctl_type == "ibrc":
                self.plot_quadrics(self.selected_image_plane[camera.name], camera.log['l'][:, :, -1])
            else:
                self.plot_quadrics_and_intersection(self.selected_image_plane[camera.name], camera.log['l1'][:, :, -1], camera.log['l2'][:, :, -1], camera.log['-x-'][:, 0, -1].reshape(2, 5, order='F'), camera.log['all_intersections'])

            self.selected_image_plane[camera.name].set_aspect("equal")
            self.selected_image_plane[camera.name].grid()
            self.selected_image_plane[camera.name].set_title(camera.name)
        plt.show(block=False)

    def plot_curves(self, camera_list: list, new_image=False):
        """
        Interface for curve plotting.

        :param camera_id: list of cameras to plot
        :type camera_id: list
        :param new_image: new image or current image_plane images, defaults to False
        :type new_image: bool, optional
        """
        if new_image:
            fig = plt.figure()
            image_plane_ax = {}
            for i, camera in enumerate(camera_list):
                image_plane_ax[camera.name] = fig.add_subplot(self.subplot_rows, self.subplot_cols, i + 1)
            img_dict = image_plane_ax
        else:
            img_dict = self.image_plane_ax

        for i, camera in enumerate(camera_list):
            if camera.ctl_type == "ibrc":
                self.plot_quadrics(img_dict[camera.name], camera.log['l'][:, :, -1])
            elif camera.ctl_type == "ibfc":
                self.plot_quadrics_and_intersection(img_dict[camera.name], camera.log['l1'][:, :, -1], camera.log['l2'][:, :, -1], camera.log['-x-'][:, 0, -1].reshape(2, 5, order='F'), camera.log['all_intersections'])
            else:
                continue
        self.invert_camera_planes()
        plt.show(block=False)
        plt.pause(0.001)

    def plot_quadrics_and_intersection(self, ax, stack_l1, stack_l2, intersection, all_intersections, lb=-.5, ub=.5):
        """
        Plot intersection of lines helper function

        :param ax: axis handle
        :type ax: [type]
        :param stack_l1: stack of lines l1
        :type stack_l1: np.ndarray
        :param stack_l2: stack of lines l2
        :type stack_l2: np.ndarray
        :param intersection: intersection point
        :type intersection: np.ndarray
        :return: updated axis handle
        :rtype: [type]
        """

        for i in range(stack_l1.shape[1]):
            l1 = stack_l1[:, i]
            l2 = stack_l2[:, i]

            # l1 and l2 are quartics, so we need to plot with:
            delta = 0.01
            xrange = np.arange(-1, 1, delta)
            yrange = np.arange(-1, 1, delta)
            X, Y = np.meshgrid(xrange, yrange)

            # Do: x**2
            l1eq = (
                l1[0] * X**2
                + l1[1] * X * Y
                + l1[2] * Y**2
                + l1[3] * X
                + l1[4] * Y
                + l1[5]
            )
            l2eq = (
                l2[0] * X**2
                + l2[1] * X * Y
                + l2[2] * Y**2
                + l2[3] * X
                + l2[4] * Y
                + l2[5]
            )

            ax.contour(X, Y, l1eq, [0], colors='blue')
            ax.contour(X, Y, l2eq, [0])

            ax.scatter(intersection[0, i], intersection[1, i], color='g', marker="x")
            ax.set_xlim(lb, ub)
            ax.set_ylim(lb, ub)

        return ax

    def plot_quadrics(self, ax, stack_l1, lb=-.5, ub=.5):
        """
        Plot intersection of lines helper function

        :param ax: axis handle
        :type ax: [type]
        :param stack_l1: stack of lines l1
        :type stack_l1: np.ndarray
        :param stack_l2: stack of lines l2
        :type stack_l2: np.ndarray
        :param intersection: intersection point
        :type intersection: np.ndarray
        :return: updated axis handle
        :rtype: [type]
        """
        for i in range(stack_l1.shape[1]):
            l1 = stack_l1[:, i]

            # l1 and l2 are quartics, so we need to plot with:
            delta = 0.01
            xrange = np.arange(-1, 1, delta)
            yrange = np.arange(-1, 1, delta)
            X, Y = np.meshgrid(xrange, yrange)

            # Do: x**2
            l1eq = (
                l1[0] * X**2
                + l1[1] * X * Y
                + l1[2] * Y**2
                + l1[3] * X
                + l1[4] * Y
                + l1[5]
            )

            ax.contour(X, Y, l1eq, [0], colors="blue")
            ax.set_xlim(lb, ub)
            ax.set_ylim(lb, ub)

        return ax

    def plot_costs(self, camera_list: list = None, block=False):
        """
        Interface for plotting costs

        :param camera_list: list of cameras to plot, defaults to None
        :type camera_list: list, optional
        """
        if not hasattr(self, "costs_ax"):
            fig2 = plt.figure()
            self.costs_ax = fig2.add_subplot(111)
        self.costs_ax.clear()
        if camera_list is None:
            camera_list = self.camera_list

        legend = []
        for i, camera in enumerate(camera_list):
            # Skip leader
            if "J" not in camera.log:
                continue

            # Append name to legend and plot it
            legend.append(camera.name)
            self.plot_cost(camera, self.costs_ax, camera.log['J'])

        self.costs_ax.set_xlabel("time [s]")
        self.costs_ax.set_ylabel("MPC Cost")
        self.costs_ax.legend(legend)
        plt.show(block=block)
        plt.pause(0.001)

    def plot_cost(self, camera, ax, cost):
        """
        Plot cost

        :param camera: camera object
        :type camera: GeneralizedCamera
        :param ax: matplotlib axis handle
        :type ax: Axis
        :param cost: cost vector
        :type cost: np.ndarray
        """

        time = np.linspace(0, cost.shape[1] * camera.dt, cost.shape[1])
        ax.plot(time, cost.reshape(time.shape))

    def plot_predicted_costs(self, camera_list: list = None):
        """
        Interface for plotting costs

        :param camera_list: list of cameras to plot, defaults to None
        :type camera_list: list, optional
        """
        if camera_list is None:
            camera_list = self.camera_list

        if not hasattr(self, "predicted_costs_ax"):
            fig = plt.figure()
            self.predicted_costs_ax = {}
            for i, camera in enumerate(camera_list):
                self.predicted_costs_ax[camera.name] = fig.add_subplot(self.subplot_rows, self.subplot_cols, i + 1)
            img_dict = self.predicted_costs_ax
        else:
            img_dict = self.predicted_costs_ax

        for i, camera in enumerate(camera_list):
            if camera.ctl_type == "basic":
                continue

            self.predicted_costs_ax[camera.name].cla()
            self.plot_predicted_costs_step(camera, img_dict[camera.name], camera.log['predicted_costs'])
        plt.show(block=False)
        plt.pause(0.001)

    def plot_predicted_costs_step(self, camera, ax, cost):
        """
        Plot cost

        :param camera: camera object
        :type camera: GeneralizedCamera
        :param ax: matplotlib axis handle
        :type ax: Axis
        :param cost: cost vector
        :type cost: np.ndarray
        """

        cost_state = cost['x']
        cost_ctl = cost['u']
        cost_ctl = np.append(cost_ctl, cost_ctl[:, [-1]], axis=1)
        time = np.linspace(0, cost_state.shape[1] * camera.dt, cost_state.shape[1])
        ax.plot(time, cost_state.reshape(time.shape))
        ax.plot(time, cost_ctl.reshape(time.shape))
        ax.legend(['state', 'control'])

    def plot_propagated_image_features(self, camera_list: list = None, step = None):
        """
        Plot feature trajectories predicted by the MPC controller.

        :param camera_list: cameras to be plot, defaults to None
        :type camera_list: list, optional
        """

        if not hasattr(self, "image_plane_ax"):
            fig = plt.figure()
            self.image_plane_ax = {}
            for i, camera in enumerate(self.camera_list):
                self.image_plane_ax[camera.name] = fig.add_subplot(self.subplot_rows, self.subplot_cols, i + 1)

        for i, camera in enumerate(self.camera_list):
            # Skip leader
            if "propagated_features" not in camera.log or camera.log["propagated_features"].size == 0:
                continue

            # Append name to legend and plot it
            # self.image_plane_ax[camera.name].cla()
            if step is None:
                self.plot_propagated_features(self.image_plane_ax[camera.name], camera.log['propagated_features'][:, :, -1])
            else:
                self.plot_propagated_features(self.image_plane_ax[camera.name], camera.log['propagated_features'][:, :, step])

        plt.pause(0.001)

    def plot_velocity_input(self, camera_list: list = None, velocity_type="linear"):
        """
        Plot control inputs provided by the MPC controller.

        :param camera_list: cameras to be plot, defaults to None
        :type camera_list: list, optional
        """

        if not hasattr(self, "linear_velocity_ax") and velocity_type == "linear":
            fig = plt.figure()
            self.linear_velocity_ax = {}
            for i, camera in enumerate(self.camera_list):
                self.linear_velocity_ax[camera.name] = fig.add_subplot(self.subplot_rows, self.subplot_cols, i + 1)
            fig.tight_layout()

        if not hasattr(self, "angular_velocity_ax") and velocity_type == "angular":
            fig = plt.figure()
            self.angular_velocity_ax = {}
            for i, camera in enumerate(self.camera_list):
                self.angular_velocity_ax[camera.name] = fig.add_subplot(self.subplot_rows, self.subplot_cols, i + 1)
            fig.tight_layout()

        for i, camera in enumerate(self.camera_list):
            # Skip leader
            if "u" not in camera.log or camera.log["u"].size == 0:
                continue

            # Append name to legend and plot it
            if velocity_type == "linear":
                self.linear_velocity_ax[camera.name].cla()
                self.plot_velocity(self.linear_velocity_ax[camera.name], camera.log['u'][:3, :])
                self.linear_velocity_ax[camera.name].set_ylabel("Linear velocity [m/s]")
                self.linear_velocity_ax[camera.name].grid()
                self.linear_velocity_ax[camera.name].set_xlabel("Time [s]")
                self.linear_velocity_ax[camera.name].legend(['x', 'y', 'z'])
                self.linear_velocity_ax[camera.name].set_title(camera.name)
            elif velocity_type == "angular":
                self.angular_velocity_ax[camera.name].cla()
                self.plot_velocity(self.angular_velocity_ax[camera.name], camera.log['u'][3:, :])
                self.angular_velocity_ax[camera.name].set_ylabel("Angular velocity [rad/s]")
                self.angular_velocity_ax[camera.name].grid()
                self.angular_velocity_ax[camera.name].set_xlabel("Time [s]")
                self.angular_velocity_ax[camera.name].legend(['x', 'y', 'z'])
                self.angular_velocity_ax[camera.name].set_title(camera.name)
        plt.pause(0.001)

    def plot_velocity(self, ax, u):
        """
        Plot feature error

        :param ax: axis handle
        :type ax: [type]
        :param err_vec: error vector
        :type err_vec: [type]
        :return: axis handle
        :rtype: [type]
        """

        t = np.linspace(0, u.shape[1] * self.camera_list[0].dt, u.shape[1])
        ax.plot(t, u.T)
        return ax

    def plot_cpu_times(self, camera_list: list = None, velocity_type="linear"):
        """
        Plot control inputs provided by the MPC controller.

        :param camera_list: cameras to be plot, defaults to None
        :type camera_list: list, optional
        """

        if not hasattr(self, "cpu_time_ax"):
            fig = plt.figure()
            self.cpu_time_ax = {}
            for i, camera in enumerate(self.camera_list):
                self.cpu_time_ax[camera.name] = fig.add_subplot(self.subplot_rows, self.subplot_cols, i + 1)
            fig.tight_layout()

        for i, camera in enumerate(self.camera_list):
            # Skip leader
            if "ct" not in camera.log or camera.log["ct"].size == 0 or camera.name == "Leader":
                continue

            self.cpu_time_ax[camera.name].cla()
            self.plot_cpu_time(self.cpu_time_ax[camera.name], camera.log['ct'])
            self.cpu_time_ax[camera.name].set_ylabel("CPU time [s]")
            self.cpu_time_ax[camera.name].grid()
            self.cpu_time_ax[camera.name].set_xlabel("Simulation Time [s]")
            self.cpu_time_ax[camera.name].legend(['instant', 'mean'])
            self.cpu_time_ax[camera.name].set_title(camera.name)

        plt.pause(0.001)

    def plot_cpu_time(self, ax, ct):
        """
        Plot CPU Time helper.

        :param ax: axis
        :type ax: _type_
        :param ct: cpu time vector
        :type ct: _type_
        """
        t = np.linspace(0, ct.shape[1] * self.camera_list[0].dt, ct.shape[1])
        ax.plot(t, ct.T)
        ax.plot(t, np.mean(ct) * np.ones(t.shape))
        return ax

    def plot_feature_errors(self, camera_list: list = None):
        """
        Plot feature trajectories predicted by the MPC controller.

        :param camera_list: cameras to be plot, defaults to None
        :type camera_list: list, optional
        """

        if not hasattr(self, "feature_errors_ax"):
            fig = plt.figure()
            self.feature_errors_ax = {}
            for i, camera in enumerate(self.camera_list):
                self.feature_errors_ax[camera.name] = fig.add_subplot(
                    self.subplot_rows, self.subplot_cols, i + 1
                )

        for i, camera in enumerate(self.camera_list):
            # Skip leader
            if (
                "error" not in camera.log
                or camera.log["error"].size == 0
            ):
                continue

            # Append name to legend and plot it
            # self.image_plane_ax[camera.name].cla()
            self.plot_error(
                self.feature_errors_ax[camera.name], camera,
                camera.log["error"],
            )
            self.feature_errors_ax[camera.name].set_ylabel("Feature error")
            self.feature_errors_ax[camera.name].set_xlabel("Time [s]")

        plt.pause(0.001)

    def plot_propagated_features(self, ax, features):
        """
        Plot predicted trajectories

        :param camera: Generalized Camera
        :type camera: GeneralizedCamera
        :param ax: _description_
        :type ax: _type_
        :param x_traj: _description_
        :type x_traj: _type_
        """
        colors = ['r', 'g', 'b', 'c', 'm']
        for i in range(features.shape[1]):
            step_points = features[:, i].reshape((2, -1))
            for j in range(step_points.shape[1]):
                ax.scatter(step_points[0, j], step_points[1, j], color=colors[j], marker="o")

    def plot_error(self, ax, camera, err_vec):
        """
        Plot feature error

        :param ax: axis handle
        :type ax: [type]
        :param err_vec: error vector
        :type err_vec: [type]
        :return: axis handle
        :rtype: [type]
        """

        t = np.linspace(0, err_vec.shape[1] * camera.dt, err_vec.shape[1])
        ax.plot(t, err_vec.T)
        return ax

    def plot_individual_errors(self, ax, err_vec, legend=None):
        """
        Plot feature error

        :param ax: axis handle
        :type ax: [type]
        :param err_vec: error vector
        :type err_vec: [type]
        :return: axis handle
        :rtype: [type]
        """

        t = np.linspace(0, err_vec.shape[1] * self.camera.dt, err_vec.shape[1])
        ax.plot(t, err_vec.T)
        if legend is not None:
            ax.legend(legend)

        return ax

    def plot_pos_error(self, ax, p_error):
        """
        Plot pose error ctl_variables.items()
        :param ax: axis handle
        :type ax: [type]
        :param x_d: desired pose p, q
        :type x_d: np.ndarray
        :return: axis handle
        :rtype: [type]
        """

        t = np.linspace(0, p_error.shape[1] * self.camera.dt, p_error.shape[1])
        ax.plot(t, p_error.T)

        return ax

    def plot_feature_trajectories(self, ax, x_traj, lb=-.5, ub=.5):
        """
        Plot image-plane for specified camera.

        :param ax: matplotlib axis
        :type ax: Axis
        :param w_points_h: world points in homogeneous coordinates
        :type w_points_h: np.ndarray
        :param pt_type: point units (normalized or pixel), defaults to "normalized"
        :type pt_type: str, optional
        :raises TypeError: wrong point type
        :return: updated matplotlib axis with camera frame
        :rtype: Axes3d
        """
        for i in range(5):
            x = x_traj[i * 2, :]
            y = x_traj[i * 2 + 1, :]
            ax.plot(x, y, "b--")

        # Plot starting positions
        for i in range(5):
            x = x_traj[i * 2, 0]
            y = x_traj[i * 2 + 1, 0]
            ax.plot(x, y, "g*")

        ax.set_xlim(lb, ub)
        ax.set_ylim(lb, ub)
        ax.xaxis.tick_top()

        return ax

    def plot_block(self):
        """
        Blocks plot until closed
        """
        plt.show(block=True)
        return

    def plot_cam_formation_error(self, leader, camera, ax, color):
        """
        Calculates and plots formation error

        :param leader: formatino leader
        :type leader: GeneralizedCamera
        :param camera: camera to calculate the error for
        :type camera: GeneralizedCamera
        :param ax: axis to plot on
        :type ax: matplotlib.Axes
        """
        error = np.zeros((camera.log['state'].shape[1], 1))
        distance_error = np.zeros((camera.log['state'].shape[1], 1))
        for i in range(camera.log['state'].shape[1]):
            leader_pose = leader.log['state'][:, i]
            camera_pose = camera.log['state'][:, i]
            R, t = get_relative_pose(leader_pose, camera_pose)
            Rf, tf = get_relative_pose(leader.get_formation_pose(), camera.get_formation_pose())
            # Calculate error
            p_error = np.linalg.norm(t - tf)
            error[i] = p_error
            distance_error[i] = np.linalg.norm(t) - np.linalg.norm(tf)

        t = np.linspace(0, error.shape[0] * camera.dt, error.shape[0])
        ax.plot(t, error, c=color)
        # ax.plot(t, distance_error)

    def plot_formation_error(self, camera_list: list = None):
        """
        Plot formation error
        """
        if not hasattr(self, "pose_error_ax"):
            fig3 = plt.figure()
            self.pose_error_ax = fig3.add_subplot(111)
        self.pose_error_ax.clear()
        if camera_list is None:
            camera_list = self.camera_list

        legend = []
        leader = None
        color = ["", "r", "g", "b", "k", "c"]
        for i, camera in enumerate(camera_list):
            # Skip leader
            if camera.name == "Leader":
                leader = camera
                continue

            # Append name to legend and plot it
            legend.append(camera.name)
            self.plot_cam_formation_error(leader, camera, self.pose_error_ax, color=color[i])

        self.pose_error_ax.set_xlabel("Time [s]")
        self.pose_error_ax.set_ylabel("Formation error [m]")
        self.pose_error_ax.legend(legend)
        self.pose_error_ax.grid()
        plt.show(block=False)
        plt.pause(0.001)

    def create_animation(self, wpoints, save_video=False, folder=None):
        """
        Create a 3D animation
        """
        import pytransform3d.transformations as pt
        import pytransform3d.trajectories as ptr
        import pytransform3d.rotations as pr
        import pytransform3d.camera as pc
        import os
        import subprocess
        import glob

        if save_video and folder is None:
            raise ValueError("Folder must be specified when saving video")

        plt.ion()
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")  # pt.plot_transform(s=0.3)

        # Calibrated camera symbol
        K = np.eye(3)
        K[0, 2] = 0.5
        K[1, 2] = 0.5

        # Offset test
        add = np.zeros((4, 4))
        add[0, 3] = 1
        add[1, 3] = 1
        add[2, 3] = 1

        # colors: leader and 5 colors
        c = ["y", "r", "g", "b", "k", "c"]

        for t in range(self.camera_list[0].log['state'].shape[1]):
            ax.clear()
            for i, cam in enumerate(self.camera_list):
                s1 = cam.log['state'][:, t]
                cam2world_transform = ptr.transforms_from_pqs(
                    np.array([s1[0], s1[1], s1[2], s1[6], s1[3], s1[4], s1[5]]), normalize_quaternions=False
                )
                pc.plot_camera(ax, K, cam2world_transform, sensor_size=np.array([1, 1]), virtual_image_distance=0.2, c=c[i])
            ax.scatter(wpoints[0, :], wpoints[1, :], wpoints[2, :], color="k", s=2)
            ax.set_xlim((-2, 2))
            ax.set_ylim((-2, 2))
            ax.set_zlim((-1, 6))
            ax.view_init(azim=110, elev=40)
            if save_video:
                plt.savefig(folder + "file%02d.png" % t)
            plt.pause(0.001)

        ax.view_init(azim=110, elev=40)
        plt.show()

        if save_video:
            os.chdir(folder)
            subprocess.call([
                'ffmpeg', '-y', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
                'simulated_animation.mp4'
            ])
            for file_name in glob.glob("*.png"):
                os.remove(file_name)
