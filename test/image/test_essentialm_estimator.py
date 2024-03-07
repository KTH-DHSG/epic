import numpy as np
import cv2 as cv
import unittest

from epic.model.generalized import GeneralizedCamera
from epic.utils.geometry import get_range_to, get_matched_features
from random import randint

unittest.TestLoader.sortTestMethodsUsing = lambda _, x, y: randint(-1, 1)


class TestMPRVS(unittest.TestCase):

    def setUp(self):
        # Set world features
        self.w_points = np.array(
            [
                [0, 0.5, -0.5, 0, 0.5],
                [0, -0.5, 0.5, 0.5, 0.1],
                [5, 5, 5, 5, 5],
            ]
        )

        self.w_points_h = np.concatenate(
            (self.w_points, np.ones((1, 5))), axis=0
        )

        self.dt = 0.1
        self.camera_1 = GeneralizedCamera(
                            h=self.dt,
                            name="Test Camera",
                            model="perspective"
                        )

        self.camera_2 = GeneralizedCamera(
                            h=self.dt,
                            name="Test Camera",
                            model="perspective"
                        )

        self.camera_1.set_formation_pose(np.array([0, 0, 0, 0, 0, 0, 1]))
        self.camera_1.set_state(np.array([0.0, 0.5, 0.0, 0, 0, 0, 1]))
        self.camera_1.set_controller("basic", [], config="")

        self.camera_2.set_formation_pose(np.array([0, 0, 0, 0, 0, 0, 1]))
        self.camera_2.set_state(np.array([0.0, -0.5, 0.0, 0, 0, 0, 1]))
        self.camera_2.set_controller("basic", [], config="")

        self.Rd = np.eye(3)
        self.td = np.array([0, -1, 0]).reshape((3, 1))

    def test_epipolar_estimator(self):
        matched_points = get_matched_features(self.w_points_h, self.camera_1, self.camera_2)
        # INFO: These are switched! So I should double check this in my code as well!
        points2 = matched_points["my_cam_u_h"][:2, :].T
        points1 = matched_points["neighbors_matched"][:2, :].T

        [E, mask] = cv.findEssentialMat(points1, points2, method=cv.RANSAC, prob=0.999, threshold=0.001)

        max_pe = np.inf
        max_re = np.inf

        # Check closest pose
        pd = self.td
        rng = get_range_to(self.camera_1, self.camera_2)
        Rd = self.Rd
        sol = {'R': None, 't': None}

        for i in range(int(E.shape[0] / 3)):

            Ei = E[i * 3:i * 3 + 3, :]
            [_, R, t, _] = cv.recoverPose(Ei, points1, points2)

            # Select the closest pose
            r, _ = cv.Rodrigues(R.dot(Rd.T))
            re = np.linalg.norm(r)

            # Pose error
            pe = np.linalg.norm(pd - t * rng)

            if pe <= max_pe and re <= max_re:
                max_pe = pe
                max_re = re
                sol['R'] = R
                sol['t'] = t

        self.assertAlmostEqual(max_re, 0)
        self.assertAlmostEqual(max_pe, 0)


if __name__.__contains__("__main__"):
    unittest.main()
