import unittest
import numpy as np
import casadi as ca
import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr
from epic.utils.geometry import get_relative_pose, r_mat, skew
import matplotlib.pyplot as plt


class TestCalculateRelativePose(unittest.TestCase):

    def xi_mat(self, q):

        if type(q) is ca.casadi.DM:
            Xi = ca.DM(4, 3)
        elif type(q) is ca.casadi.MX:
            Xi = ca.MX(4, 3)
        elif type(q) is np.ndarray:
            Xi = np.zeros((4, 3))
        else:
            raise ValueError("Unknown type for q")
        # Extract states
        qx = q[0]
        qy = q[1]
        qz = q[2]
        qw = q[3]

        # Generate Xi matrix
        Xi[0, 0] = qw
        Xi[0, 1] = -qz
        Xi[0, 2] = qy

        Xi[1, 0] = qz
        Xi[1, 1] = qw
        Xi[1, 2] = -qx

        Xi[2, 0] = -qy
        Xi[2, 1] = qx
        Xi[2, 2] = qw

        Xi[3, 0] = -qx
        Xi[3, 1] = -qy
        Xi[3, 2] = -qz

        return Xi

    def kinematics(self, x, u):
        q = x[3:] / np.linalg.norm(x[3:])

        # 3D Force
        v = u[0:3]

        # 3D Torque
        w = u[3:]

        # Model
        pdot = r_mat(q) @ v
        qdot = self.xi_mat(q) @ w / 2
        return np.concatenate((pdot, qdot), axis=0).reshape((-1, 1))

    def get_range_step(self, r0):
        """
        Get range step.

        :param r0: range
        :type r0: ca.MX
        :return: range step
        :rtype: ca.MX
        """

        self.r_step = np.concatenate((-np.eye(3), skew(r0)), axis=1)

        return self.r_step

    def relative_navigation_model(self, t0, u, uL, RLF=np.eye(3)):
        """
        Relative navigation model considering range and features.
        """

        # Leader contribution
        L_contrib = np.hstack((RLF, np.zeros((3, 3))))

        relative_pos_dt = self.get_range_step(t0) @ u + L_contrib @ uL

        return relative_pos_dt

    def test_kinematics_2ag(self):
        u1 = np.array([[0.01, 0.0, 0.0, 0, 0, 0.1]]).T
        x1 = np.array([[0.0, 0.0, 0.0, 0, 0, 0, 1]]).T

        x2 = np.array([[1.0, 0.0, 0.0, 0, 0, 0, 1]]).T

        dt = 0.1

        vL = u1[0:3].reshape((3, 1))
        wL = u1[3:].reshape((3, 1))

        rel_vec = np.array([[1, 0, 0]]).T  # goes from follower to leader
        VF_in_L = vL[0:3] + np.cross(wL.T, rel_vec.T).T
        u_rot = np.eye(3)

        # Get desired linear and angular velocity in my frame
        u_rot = np.concatenate((u_rot @ VF_in_L.reshape(3, -1),
                                u_rot @ wL), axis=0)
        rel_vec_test = -rel_vec
        plot_3 = np.zeros((3, 200))
        plot_3[:, 0] = rel_vec_test.flatten()
        for i in range(200):
            # Propagate relative position only
            if i == 0:
                continue
            rldot = self.relative_navigation_model(-rel_vec, u_rot, u1)
            rel_vec_test = plot_3[:, [i - 1]] + rldot * dt
            plot_3[:, i] = rel_vec_test.flatten()

        for i in range(200):
            print("Step: ", i, "/200")
            rldot = self.relative_navigation_model(-rel_vec, u_rot, u1)
            xdot2 = self.kinematics(x2, u_rot)
            xdot1 = self.kinematics(x1, u1)
            x1 = x1 + xdot1 * dt
            x2 = x2 + xdot2 * dt
            rel_vec_test = rel_vec_test + rldot * dt
            # Re-normalize quaternion
            x1[3:] = x1[3:] / np.linalg.norm(x1[3:])
            x2[3:] = x2[3:] / np.linalg.norm(x2[3:])

            # Get relative pose
            R, t = get_relative_pose(x1, x2)
            print("R: ", R)
            print("t: ", t)

            # Plot
            q1_p3d = np.roll(x1[3:], 1)
            q2_p3d = np.roll(x2[3:], 1)
            if not hasattr(self, "ax"):
                self.ax = pytr.plot_transform(A2B=pytr.transform_from(
                    pyrot.matrix_from_quaternion(q1_p3d.flatten()), x1[0:3].flatten()))
                pytr.plot_transform(ax=self.ax, A2B=pytr.transform_from(
                    pyrot.matrix_from_quaternion(q2_p3d.flatten()), x2[0:3].flatten()))
                pytr.plot_transform(ax=self.ax, A2B=pytr.transform_from(
                    pyrot.matrix_from_quaternion(q2_p3d.flatten()), x2[0:3].flatten()))
            else:
                self.ax.cla()
                pytr.plot_transform(ax=self.ax, A2B=pytr.transform_from(
                    pyrot.matrix_from_quaternion(q1_p3d.flatten()), x1[0:3].flatten()))
                pytr.plot_transform(ax=self.ax, A2B=pytr.transform_from(
                    pyrot.matrix_from_quaternion(q2_p3d.flatten()), x2[0:3].flatten()))
                R = r_mat(x2[3:])
                plot_3_step = np.concatenate((x2[:3], x2[:3] + R @ plot_3[:, [i]]), axis=1)
                print("Plot: ", plot_3_step)
                self.ax.plot(plot_3_step[0, :], plot_3_step[1, :], plot_3_step[2, :], color="black")
                self.ax.set_xlim([-2, 2])
                self.ax.set_ylim([0, 4])
                self.ax.set_zlim([0, 4])
            plt.show(block=False)
            plt.pause(0.001)

        plt.show(block=True)


if __name__ == '__main__':
    unittest.main()
