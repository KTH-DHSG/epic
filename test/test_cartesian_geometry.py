import unittest
import numpy as np
import casadi as ca
import pyquaternion as pq
import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr
from epic.utils.geometry import get_relative_pose, r_mat
import matplotlib.pyplot as plt

PLOT = False


class TestCalculateRelativePose(unittest.TestCase):

    def test_relative_pose_basic(self):
        # Create two quaternions
        quat1 = pq.Quaternion(axis=[0, 0, 1], degrees=-90)  # Representing a 45-degree rotation around the z-axis
        quat2 = pq.Quaternion(axis=[0, 0, 1], degrees=-90)  # Representing a 30-degree rotation around the x-axis

        # Expected answer
        expected_rotation_matrix = np.eye(3)
        expected_translation = np.array([[1.0, 0.0, 0.0]]).T

        # My answer
        q1 = np.array(quat1.elements)
        q1 = np.roll(q1, -1)

        q2 = np.array(quat2.elements)
        q2 = np.roll(q2, -1)

        state1 = np.concatenate((np.array([0.0, -5.0, 0.0]), q1), axis=0)
        state2 = np.concatenate((np.array([0.0, -6.0, 0.0]), q2), axis=0)

        R, t = get_relative_pose(state1, state2)  # t in body frame of state1
        np.testing.assert_array_almost_equal(R, expected_rotation_matrix)
        np.testing.assert_array_almost_equal(t, expected_translation)

    def test_relative_pose_with_library(self):
        p = np.array([0.0, -5.0, 0.0])
        quat1 = pq.Quaternion(axis=[0, 0, 1], degrees=-90)
        quat1_np = np.array(quat1.elements)
        q1 = np.roll(quat1_np, -1)
        W2L = pytr.transform_from(pyrot.matrix_from_quaternion(quat1_np), p)

        p = np.array([0.0, -6.0, 0.0])
        quat2 = pq.Quaternion(axis=[0, 0, 1], degrees=-90)  # Quaternion 2,
        quat2_np = np.array(quat2.elements)
        q2 = np.roll(quat2_np, -1)
        W2F = pytr.transform_from(pyrot.matrix_from_quaternion(quat2_np), p)

        # L2F = pytr.concat(W2L, pytr.invert_transform(W2F))
        # F2L = pytr.concat(W2F, pytr.invert_transform(W2L))

        state1 = np.concatenate((np.array([0.0, -5.0, 0.0]), q1), axis=0)
        state2 = np.concatenate((np.array([0.0, -6.0, 0.0]), q2), axis=0)
        R, t = get_relative_pose(state1, state2)

        if PLOT:
            ax = pytr.plot_transform(A2B=W2L)
            pytr.plot_transform(ax, A2B=W2F)
            # ax.scatter(p[0], p[1], p[2])
            plt.show(block=False)

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

    def test_kinematics(self):
        u = np.array([0.0, 0.0, 0.0, 0, 0, 0.1])
        x = np.array([[0.0, 0.0, 0.0, 0, 0, 0, 1]]).T

        dt = 0.01

        for i in range(200):
            print("Step: ", i, "/200")
            xdot = self.kinematics(x, u)
            x = x + xdot * dt

            # Re-normalize quaternion
            x[3:] = x[3:] / np.linalg.norm(x[3:])

            # Plot
            q_p3d = np.roll(x[3:], 1)
            if not hasattr(self, "ax"):
                self.ax = pytr.plot_transform(A2B=pytr.transform_from(
                    pyrot.matrix_from_quaternion(q_p3d.flatten()), x[0:3].flatten()))
            else:
                self.ax.cla()
                pytr.plot_transform(ax=self.ax, A2B=pytr.transform_from(
                    pyrot.matrix_from_quaternion(q_p3d.flatten()), x[0:3].flatten()))
            if PLOT:
                plt.show(block=False)
                plt.pause(0.001)

    def test_kinematics_2ag(self):
        u1 = np.array([0.0, 0.0, 0.0, 0, 0, 0.1])
        x1 = np.array([[0.0, 0.0, 0.0, 0, 0, 0, 1]]).T

        x2 = np.array([[1.0, 0.0, 0.0, 0, 0, 0, 1]]).T

        dt = 0.1

        for i in range(200):
            print("Step: ", i, "/200")

            vL = u1[0:3].reshape((3, 1))
            wL = u1[3:].reshape((3, 1))

            rel_vec = np.array([[1, 0, 0]]).T  # goes from follower to leader
            VF_in_L = vL[0:3] + np.cross(wL.T, rel_vec.T).T
            u_rot = np.eye(3)
            # Get desired linear and angular velocity in my frame

            u_rot = np.concatenate((u_rot @ VF_in_L.reshape(3, -1),
                                    u_rot @ wL), axis=0)

            xdot2 = self.kinematics(x2, u_rot)
            xdot1 = self.kinematics(x1, u1)
            x1 = x1 + xdot1 * dt
            x2 = x2 + xdot2 * dt
            # Re-normalize quaternion
            x1[3:] = x1[3:] / np.linalg.norm(x1[3:])
            x2[3:] = x2[3:] / np.linalg.norm(x2[3:])

            # Get relative pose
            R, t = get_relative_pose(x1, x2)

            # Plot
            q1_p3d = np.roll(x1[3:], 1)
            q2_p3d = np.roll(x2[3:], 1)
            if not hasattr(self, "ax"):
                self.ax = pytr.plot_transform(A2B=pytr.transform_from(
                    pyrot.matrix_from_quaternion(q1_p3d.flatten()), x1[0:3].flatten()))
                pytr.plot_transform(ax=self.ax, A2B=pytr.transform_from(
                    pyrot.matrix_from_quaternion(q2_p3d.flatten()), x2[0:3].flatten()))
            else:
                self.ax.cla()
                pytr.plot_transform(ax=self.ax, A2B=pytr.transform_from(
                    pyrot.matrix_from_quaternion(q1_p3d.flatten()), x1[0:3].flatten()))
                pytr.plot_transform(ax=self.ax, A2B=pytr.transform_from(
                    pyrot.matrix_from_quaternion(q2_p3d.flatten()), x2[0:3].flatten()))

            if PLOT:
                plt.show(block=False)
                plt.pause(0.001)


if __name__ == '__main__':
    unittest.main()
