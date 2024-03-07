import casadi as ca
import numpy as np


class IBVS(object):
    def __init__(self, cam):
        self.cam = cam
        self.interaction_matrix = self.cam.get_interaction_matrix  # Function map
        self.csi = self.cam.csi
        self.Nx = self.cam.f_n * 2  # Tracked features number, ref Camera Class
        self.Nu = self.cam.m
        self.dt = self.cam.dt
        self.error = np.zeros((10, 1))
        self.gain = 10

    def set_reference(self, x_d):
        self.x_sp = x_d.reshape(self.cam.f_n * 2, 1, order="F")

    def control(self, x, Z):
        L = self.interaction_matrix(x, Z, self.csi)
        self.error = x - self.x_sp
        u = - self.gain * ca.pinv(L) @ self.error

        data_struct = dict(
            {"J": np.linalg.norm(self.x_sp - x), "ct": 0.0}
        )

        return u, data_struct

    def empty_log(self) -> dict:
        """
        Empty dictionary for logged variables.

        :return: empty dictionary with same footprint as control
        :rtype: dict
        """
        return dict(
            {
                "u": np.empty((self.my_cam.m, 0)),
                "J": np.empty((1, 0)),
                "ct": np.empty((1, 0)),
                "-x-": np.empty((self.my_cam.f_n * 2, 1, 0)),
            }
        )
