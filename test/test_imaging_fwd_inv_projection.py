import numpy as np
import unittest

from epic.model.generalized import GeneralizedCamera
from epic.utils.geometry import get_random_quat_around
from random import randint

unittest.TestLoader.sortTestMethodsUsing = lambda _, x, y: randint(-1, 1)


class TestMPRVS(unittest.TestCase):

    def setUp(self):
        # Set print options
        np.set_printoptions(precision=10)
        # Set camera
        self.camera_perspective = GeneralizedCamera("test_camera_p", model="perspective")
        self.camera_hyperbolic = GeneralizedCamera("test_camera_h", model="hyperbolic")
        self.camera_parabolic = GeneralizedCamera("test_camera_par", model="parabolic")
        self.camera_distortion = GeneralizedCamera("test_camera_d", model="distortion")

        # Set world points
        self.world_points = np.array(
            [
                [0, 0.5, -0.5, 0, 0.5],
                [0, -0.5, 0.5, 0.5, 0.1],
                [10, 10, 10, 10, 10],
                [1, 1, 1, 1, 1]
            ]
        )
        pass

    def set_origin_state(self):
        camera_state = np.array([0, 0, 0, 0, 0, 0, 1])

        self.camera_perspective.set_state(camera_state)
        self.camera_hyperbolic.set_state(camera_state)
        self.camera_parabolic.set_state(camera_state)
        self.camera_distortion.set_state(camera_state)

    def set_known_state(self):
        camera_state = np.array([0.02, -0.3, -0.5, 0.089, 0.04, 0.089, 0.994])

        self.camera_perspective.set_state(camera_state)
        self.camera_hyperbolic.set_state(camera_state)
        self.camera_parabolic.set_state(camera_state)
        self.camera_distortion.set_state(camera_state)

    def set_random_position(self):
        camera_position_xy = (np.random.rand(1, 2) - 0.5) * 0.2
        camera_position_z = (np.random.rand(1, 1) - 0.5)
        camera_position = np.concatenate((camera_position_xy, camera_position_z), axis=1).flatten()
        camera_state = np.concatenate((camera_position, np.array([0, 0, 0, 1]))).flatten()

        self.camera_perspective.set_state(camera_state)
        self.camera_hyperbolic.set_state(camera_state)
        self.camera_parabolic.set_state(camera_state)
        self.camera_distortion.set_state(camera_state)

    def set_random_attitude(self):
        q = np.array([0, 0, 0, 1])
        q = get_random_quat_around(q, 7.5)
        camera_state = np.concatenate((np.array([0, 0, 0]), q)).flatten()

        self.camera_perspective.set_state(camera_state)
        self.camera_hyperbolic.set_state(camera_state)
        self.camera_parabolic.set_state(camera_state)
        self.camera_distortion.set_state(camera_state)

    def set_random_state(self):
        camera_position_xy = (np.random.rand(1, 2) - 0.5) * 0.2
        camera_position_z = (np.random.rand(1, 1) - 0.5)
        camera_position = np.concatenate((camera_position_xy, camera_position_z), axis=1).flatten()
        q = np.array([0, 0, 0, 1])
        q = get_random_quat_around(q, 7.5)
        camera_state = np.concatenate((camera_position, q)).flatten()

        self.camera_perspective.set_state(camera_state)
        self.camera_hyperbolic.set_state(camera_state)
        self.camera_parabolic.set_state(camera_state)
        self.camera_distortion.set_state(camera_state)

    def test_image_projection(self):
        self.set_origin_state()
        image_points_p, u_points_p, depth_p = self.camera_perspective.get_image_points(self.world_points)
        image_points_h, u_points_h, depth_h = self.camera_hyperbolic.get_image_points(self.world_points)
        image_points_par, u_points_par, depth_par = self.camera_parabolic.get_image_points(self.world_points)
        image_points_d, u_points_d, depth_d = self.camera_distortion.get_image_points(self.world_points)
        pass

    def test_projective_model(self):
        self.set_origin_state()
        image_points_p, u_points_p, depth_p = self.camera_perspective.get_image_points(self.world_points)
        wp_p, depth_p = self.camera_perspective.get_camera_frame_points(image_points_p, depth_p)
        equal_p = np.all(np.isclose(self.world_points[:3, :], wp_p) is True)
        self.assertTrue(equal_p)

    def test_projective_model_known(self):
        self.set_known_state()
        image_points_p, u_points_p, depth_p = self.camera_perspective.get_image_points(self.world_points)
        wp_p, depth_p = self.camera_perspective.get_camera_frame_points(image_points_p, depth_p)
        equal_p = np.all(np.isclose(self.world_points[:3, :], wp_p) is True)
        self.assertTrue(equal_p)

    def test_project_to_image_and_back(self):
        self.set_origin_state()

        image_points_p, u_points_p, depth_p = self.camera_perspective.get_image_points(self.world_points)
        wp_p, depth_p = self.camera_perspective.get_camera_frame_points(image_points_p, depth_p)
        equal_p = np.all(np.isclose(self.world_points[:3, :], wp_p) is True)
        self.assertTrue(equal_p)

        image_points_h, u_points_h, depth_h = self.camera_hyperbolic.get_image_points(self.world_points)
        wp_h, depth_h = self.camera_hyperbolic.get_camera_frame_points(image_points_h, depth_h)
        equal_h = np.all(np.isclose(self.world_points[:3, :], wp_h) is True)
        self.assertTrue(equal_h)

        image_points_par, u_points_par, depth_par = self.camera_parabolic.get_image_points(self.world_points)
        wp_par, depth_par = self.camera_parabolic.get_camera_frame_points(image_points_par, depth_par)
        equal_par = np.all(np.isclose(self.world_points[:3, :], wp_par) is True)
        self.assertTrue(equal_par)

        image_points_d, u_points_d, depth_d = self.camera_distortion.get_image_points(self.world_points)
        wp_d, depth_d = self.camera_distortion.get_camera_frame_points(image_points_d, depth_d)
        equal_d = np.all(np.isclose(self.world_points[:3, :], wp_d) is True)
        self.assertTrue(equal_d)
        pass

    def test_project_to_image_and_back_known(self):
        self.set_known_state()

        image_points_p, u_points_p, depth_p = self.camera_perspective.get_image_points(self.world_points)
        wp_p, depth_p = self.camera_perspective.get_camera_frame_points(image_points_p, depth_p)
        equal_p = np.all(np.isclose(self.world_points[:3, :], wp_p) is True)
        self.assertTrue(equal_p)

        image_points_h, u_points_h, depth_h = self.camera_hyperbolic.get_image_points(self.world_points)
        wp_h, depth_h = self.camera_hyperbolic.get_camera_frame_points(image_points_h, depth_h)
        equal_h = np.all(np.isclose(self.world_points[:3, :], wp_h) is True)
        self.assertTrue(equal_h)

        image_points_par, u_points_par, depth_par = self.camera_parabolic.get_image_points(self.world_points)
        wp_par, depth_par = self.camera_parabolic.get_camera_frame_points(image_points_par, depth_par)
        equal_par = np.all(np.isclose(self.world_points[:3, :], wp_par) is True)
        self.assertTrue(equal_par)

        image_points_d, u_points_d, depth_d = self.camera_distortion.get_image_points(self.world_points)
        wp_d, depth_d = self.camera_distortion.get_camera_frame_points(image_points_d, depth_d)
        equal_d = np.all(np.isclose(self.world_points[:3, :], wp_d) is True)
        self.assertTrue(equal_d)
        pass

    def test_project_to_image_and_back_random_position(self):
        self.set_random_position()

        image_points_p, u_points_p, depth_p = self.camera_perspective.get_image_points(self.world_points)
        wp_p, depth_p = self.camera_perspective.get_camera_frame_points(image_points_p, depth_p)
        equal_p = np.all(np.isclose(self.world_points[:3, :], wp_p) is True)
        self.assertTrue(equal_p)

        image_points_h, u_points_h, depth_h = self.camera_hyperbolic.get_image_points(self.world_points)
        wp_h, depth_h = self.camera_hyperbolic.get_camera_frame_points(image_points_h, depth_h)
        equal_h = np.all(np.isclose(self.world_points[:3, :], wp_h) is True)
        self.assertTrue(equal_h)

        image_points_par, u_points_par, depth_par = self.camera_parabolic.get_image_points(self.world_points)
        wp_par, depth_par = self.camera_parabolic.get_camera_frame_points(image_points_par, depth_par)
        equal_par = np.all(np.isclose(self.world_points[:3, :], wp_par) is True)
        self.assertTrue(equal_par)

        image_points_d, u_points_d, depth_d = self.camera_distortion.get_image_points(self.world_points)
        wp_d, depth_d = self.camera_distortion.get_camera_frame_points(image_points_d, depth_d)
        equal_d = np.all(np.isclose(self.world_points[:3, :], wp_d) is True)
        self.assertTrue(equal_d)

    def test_project_to_image_and_back_random_attitude(self):
        self.set_random_attitude()

        image_points_p, u_points_p, depth_p = self.camera_perspective.get_image_points(self.world_points)
        wp_p, depth_p = self.camera_perspective.get_camera_frame_points(image_points_p, depth_p)
        equal_p = np.all(np.isclose(self.world_points[:3, :], wp_p) is True)
        self.assertTrue(equal_p)

        image_points_h, u_points_h, depth_h = self.camera_hyperbolic.get_image_points(self.world_points)
        wp_h, depth_h = self.camera_hyperbolic.get_camera_frame_points(image_points_h, depth_h)
        equal_h = np.all(np.isclose(self.world_points[:3, :], wp_h) is True)
        self.assertTrue(equal_h)

        image_points_par, u_points_par, depth_par = self.camera_parabolic.get_image_points(self.world_points)
        wp_par, depth_par = self.camera_parabolic.get_camera_frame_points(image_points_par, depth_par)
        equal_par = np.all(np.isclose(self.world_points[:3, :], wp_par) is True)
        self.assertTrue(equal_par)

        image_points_d, u_points_d, depth_d = self.camera_distortion.get_image_points(self.world_points)
        wp_d, depth_d = self.camera_distortion.get_camera_frame_points(image_points_d, depth_d)
        equal_d = np.all(np.isclose(self.world_points[:3, :], wp_d) is True)
        self.assertTrue(equal_d)

    def test_project_to_image_and_back_random_state(self):
        self.set_random_state()

        image_points_p, u_points_p, depth_p = self.camera_perspective.get_image_points(self.world_points)
        wp_p, depth_p = self.camera_perspective.get_camera_frame_points(image_points_p, depth_p)
        equal_p = np.all(np.isclose(self.world_points[:3, :], wp_p) is True)
        self.assertTrue(equal_p)

        image_points_h, u_points_h, depth_h = self.camera_hyperbolic.get_image_points(self.world_points)
        wp_h, depth_h = self.camera_hyperbolic.get_camera_frame_points(image_points_h, depth_h)
        equal_h = np.all(np.isclose(self.world_points[:3, :], wp_h) is True)
        self.assertTrue(equal_h)

        image_points_par, u_points_par, depth_par = self.camera_parabolic.get_image_points(self.world_points)
        wp_par, depth_par = self.camera_parabolic.get_camera_frame_points(image_points_par, depth_par)
        equal_par = np.all(np.isclose(self.world_points[:3, :], wp_par) is True)
        self.assertTrue(equal_par)

        image_points_d, u_points_d, depth_d = self.camera_distortion.get_image_points(self.world_points)
        wp_d, depth_d = self.camera_distortion.get_camera_frame_points(image_points_d, depth_d)
        equal_d = np.all(np.isclose(self.world_points[:3, :], wp_d) is True)
        self.assertTrue(equal_d)

    def test_hyperbolic_model_known(self):
        self.set_known_state()
        image_points_h, u_points_h, depth_h = self.camera_hyperbolic.get_image_points(self.world_points)
        wp_h, depth_h = self.camera_hyperbolic.get_camera_frame_points(image_points_h, depth_h)
        equal_h = np.all(np.isclose(self.world_points[:3, :], wp_h) is True)
        self.assertTrue(equal_h)


if __name__ == '__main__':
    unittest.main()
