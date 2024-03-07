import numpy as np
import matplotlib.pyplot as plt
import viscoor.util as vu

plt.rcParams['text.usetex'] = True


class Plotter(object):
    """
    Environment plotting class

    :param object: _description_
    :type object: _type_
    """
    def __init__(self, logs: dict) -> None:
        self.logs = logs

        # Create MPC Cost figure - overlay all agents
        fig2 = plt.figure()
        self.costs_ax = fig2.add_subplot(111)

        # Create image plane - separate each agent
        image_plane_figs = []
        self.image_plane_start_ax = []
        self.image_plane_end_ax = []
        for i in range(len(self.logs)):
            image_plane_figs.append(plt.figure())
            self.image_plane_start_ax.append(image_plane_figs[-1].add_subplot(111))

            image_plane_figs.append(plt.figure())
            self.image_plane_end_ax.append(image_plane_figs[-1].add_subplot(111))
           
            self.image_plane_start_ax[-1]
            self.image_plane_end_ax[-1]

            self.image_plane_start_ax[-1].set_xlabel(r'$u$ [u]')
            self.image_plane_start_ax[-1].set_ylabel(r'$v$ [u]')
            self.image_plane_start_ax[-1].xaxis.set_label_position('top')

            self.image_plane_end_ax[-1].set_xlabel(r'$u$ [u]')
            self.image_plane_end_ax[-1].set_ylabel(r'$v$ [u]')
            self.image_plane_end_ax[-1].xaxis.set_label_position('top')

        # Create control input figures
        control_input_figs = []
        self.ctl_input_ax = []
        for i in range(len(self.logs)):
            control_input_figs.append(plt.figure())
            self.ctl_input_ax.append(control_input_figs[-1].add_subplot(111))

        plt.show(block=False)

    def plot_image_planes(self, lb=-.25, ub=.25):
        """
        Plot image-plane for specified cameras.

        :param camera_list: list of cameras, maximum 3
        :type camera_list: list
        """
        for i in range(len(self.logs)):
            # Create image plane
            log = self.logs[i]

            # Plot feature trajectories and curves
            if '-x-' not in log:
                # IBRC type
                self.plot_feature_trajectories(self.image_plane_start_ax[i], log['x'][3:, :], mode="start")
                self.plot_feature_trajectories(self.image_plane_end_ax[i], log['x'][3:, :])
                
                self.plot_quadrics(self.image_plane_start_ax[i], log['l'][:, :, -1])
                self.plot_quadrics(self.image_plane_end_ax[i], log['l'][:, :, -1])
            else:
                # IBFC type
                self.plot_feature_trajectories(self.image_plane_start_ax[i], log['x'], mode="start")
                self.plot_feature_trajectories(self.image_plane_end_ax[i], log['x'])
                
                self.plot_quadrics_and_intersection(self.image_plane_start_ax[i], log['l1'][:, :, -1], log['l2'][:, :, -1], log['-x-'][:, 0, -1].reshape(2, 5, order='F'), log['all_intersections'])
                self.plot_quadrics_and_intersection(self.image_plane_end_ax[i], log['l1'][:, :, -1], log['l2'][:, :, -1], log['-x-'][:, 0, -1].reshape(2, 5, order='F'), log['all_intersections'])

            # Plot image plane at the end of the trajectory
            self.image_plane_start_ax[i].set_aspect("equal")
            self.image_plane_end_ax[i].set_aspect("equal")
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
            else:
                self.plot_quadrics_and_intersection(img_dict[camera.name], camera.log['l1'][:, :, -1], camera.log['l2'][:, :, -1], camera.log['-x-'][:, 0, -1].reshape(2, 5, order='F'), camera.log['all_intersections'])
        plt.show(block=False)
        plt.pause(0.001)

    def plot_quadrics_and_intersection(self, ax, stack_l1, stack_l2, intersection, all_intersections):
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
            ax.set_xlim(-.25, .25)
            ax.set_ylim(-.25, .25)

        return ax

    def plot_quadrics(self, ax, stack_l1):
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
            ax.set_xlim(-0.25, 0.25)
            ax.set_ylim(-0.25, 0.25)

        return ax

    def plot_costs(self, camera_list: list = None):
        """
        Interface for plotting costs

        :param camera_list: list of cameras to plot, defaults to None
        :type camera_list: list, optional
        """
        self.costs_ax.clear()
        if camera_list is None:
            camera_list = self.camera_list

        legend = []
        for i, camera in enumerate(camera_list):
            # Skip leader
            if camera.name == "Leader":
                continue

            # Append name to legend and plot it
            legend.append(camera.name)
            self.plot_cost(camera, self.costs_ax, camera.log['J'])

        self.costs_ax.set_xlabel("time [s]")
        self.costs_ax.set_ylabel("MPC Cost")
        self.costs_ax.legend(legend)
        plt.show(block=False)
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

    def plot_predictions(self, camera_list: list = None):
        """
        Interface for plotting costs

        :param camera_list: list of cameras to plot, defaults to None
        :type camera_list: list, optional
        """
        if camera_list is None:
            camera_list = self.camera_list

        if not hasattr(self, "costs_ax"):
            fig = plt.figure()
            self.costs_ax = {}
            for i, camera in enumerate(camera_list):
                self.costs_ax[camera.name] = fig.add_subplot(self.subplot_rows, self.subplot_cols, i + 1)
            img_dict = self.costs_ax
        else:
            img_dict = self.costs_ax

        for i, camera in enumerate(camera_list):
            if camera.name == "Leader":
                continue
            self.plot_cost(camera, img_dict[camera.name], camera.log['J'])
        plt.show(block=False)
        plt.pause(0.001)

    def plot_error(self, ax, err_vec):
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
        ax.plot(t, np.linalg.norm(err_vec.T, axis=1))
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
        Plot pose error

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

    def plot_feature_trajectories(self, ax, x_traj, mode: str = 'full'):
        """

        :param ax: axis
        :param x_traj: feature trajectories
        :param mode: 'full', 'start', 'end'
        :return: _description_
        """
        if mode == 'full' or mode == 'start':
            # Plot starting positions
            for i in range(5):
                x = x_traj[i * 2, 0]
                y = x_traj[i * 2 + 1, 0]
                ax.plot(x, y, "go")
        if mode == 'full':
            # Plot trajectory
            for i in range(5):
                x = x_traj[i * 2, 1:-1]
                y = x_traj[i * 2 + 1, 1:-1]
                ax.plot(x, y, "b--")

        if mode == 'full' or mode == 'end':
            # Plot trajectory
            for i in range(5):
                x = x_traj[i * 2, -1]
                y = x_traj[i * 2 + 1, -1]
                ax.plot(x, y, "ko")

        ax.grid()
        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(-0.25, 0.25)
        ax.xaxis.tick_top()

        return ax

    def plot_block(self):
        """
        Blocks plot until closed
        """
        plt.show(block=True)
        return

    def plot_cam_formation_error(self, leader, camera, ax):
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
        for i in range(camera.log['state'].shape[1]):
            leader_p = leader.log['state'][0:3, i]
            leader_q = leader.log['state'][3:7, i]
            leader_R = vu.r_mat(leader_q)

            camera_p = camera.log['state'][0:3, i]

            # Calculate error
            rel_p = leader_R @ (camera_p - leader_p)
            p_error = np.linalg.norm(camera.formation_x_d[0:3] - rel_p)
            error[i] = p_error

        t = np.linspace(0, error.shape[0] * camera.dt, error.shape[0])
        ax.plot(t, error)

    def plot_formation_error(self, camera_list: list = None):
        """
        Plot formation error
        """
        self.pose_error_ax.clear()
        if camera_list is None:
            camera_list = self.camera_list

        legend = []
        leader = None
        for i, camera in enumerate(camera_list):
            # Skip leader
            if camera.name == "Leader":
                leader = camera
                continue

            # Append name to legend and plot it
            legend.append(camera.name)
            self.plot_cam_formation_error(leader, camera, self.pose_error_ax)

        self.pose_error_ax.set_xlabel("Time [s]")
        self.pose_error_ax.set_ylabel("Formation error [m]")
        self.pose_error_ax.legend(legend)
        plt.show(block=False)
        plt.pause(0.001)
