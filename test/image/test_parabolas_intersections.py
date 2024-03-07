import numpy as np
import unittest
import matplotlib.pyplot as plt

from epic.utils.geometry import get_parabola_intersections, get_lines_intersection
from random import randint

unittest.TestLoader.sortTestMethodsUsing = lambda _, x, y: randint(-1, 1)


class TestIntersections(unittest.TestCase):

    def setUp(self):
        """
        Set up test data.
        """
        # p = [a, b, c, d, e, f] ->  (a x**2 + b xy + c y**2 + d x + e y + f) = 0
        self.p11 = np.array([1, 0, 0, 0, -1, -2])
        self.p12 = np.array([-1, 0, 0, 6, -1, -6.5])
        self.p11 = self.p11 / self.p11[-1]
        self.p12 = self.p12 / self.p12[-1]
        self.i1 = np.array([1.5, 0.25])

        self.p21 = np.array([1, 0, 0, -2, -1, 2])
        self.p22 = np.array([-1, 0, 0, 8, -1, 1])
        self.p21 = self.p21 / self.p21[-1]
        self.p22 = self.p22 / self.p22[-1]
        self.i2 = np.array([[0.102, 1.806], [4.898, 16.194]])

        self.l1 = np.array([0, 0, 0, 1.5, -1, -2])
        self.l2 = np.array([0, 0, 0, -0.7, -1, 3])
        self.l1 = self.l1 / self.l1[-1]
        self.l2 = self.l2 / self.l2[-1]

    def test_quadrics_intersection(self):
        """
        Test intersecation of parabolics
        """
        # Parabolic test is failing
        test_i1 = get_parabola_intersections(self.p11, self.p12)
        test_i2 = get_parabola_intersections(self.p21, self.p22)

        self.assertAlmostEqual(test_i1[0][0], 1.5, places=5)
        self.assertAlmostEqual(test_i1[0][1], 0.25, places=5)
        self.assertAlmostEqual(test_i1[1][0], 1.5, places=5)
        self.assertAlmostEqual(test_i1[1][1], 0.25, places=5)

        if np.abs(test_i2[0][0] - 4.898) < np.abs(test_i2[1][0] - 4.898):
            self.assertAlmostEqual(test_i2[1][0], 0.102, places=3)
            self.assertAlmostEqual(test_i2[1][1], 1.806, places=3)
            self.assertAlmostEqual(test_i2[0][0], 4.898, places=3)
            self.assertAlmostEqual(test_i2[0][1], 16.194, places=3)
        else:
            self.assertAlmostEqual(test_i2[0][0], 0.102, places=3)
            self.assertAlmostEqual(test_i2[0][1], 1.806, places=3)
            self.assertAlmostEqual(test_i2[1][0], 4.898, places=3)
            self.assertAlmostEqual(test_i2[1][1], 16.194, places=3)

    def test_lines_intersection(self):
        """
        Test intersecation of lines
        """
        test_i = get_lines_intersection(self.l1, self.l2)
        self.assertAlmostEqual(test_i[0, 0], 2.2727272727272725)
        self.assertAlmostEqual(test_i[1, 0], 1.5 * 2.2727272727272725 - 2)

    def test_plot_parabolas(self):
        """
        Test plotting of parabolas
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        delta = 0.01
        xrange = np.arange(-20, 20, delta)
        yrange = np.arange(-20, 20, delta)
        X, Y = np.meshgrid(xrange, yrange)

        # Do the contour plot
        l1eq = (
            self.p11[0] * X**2
            + self.p11[1] * X * Y
            + self.p11[2] * Y**2
            + self.p11[3] * X
            + self.p11[4] * Y
            + self.p11[5]
        )
        l2eq = (
            self.p12[0] * X**2
            + self.p12[1] * X * Y
            + self.p12[2] * Y**2
            + self.p12[3] * X
            + self.p12[4] * Y
            + self.p12[5]
        )

        l3eq = (
            self.p21[0] * X**2
            + self.p21[1] * X * Y
            + self.p21[2] * Y**2
            + self.p21[3] * X
            + self.p21[4] * Y
            + self.p21[5]
        )

        l4eq = (
            self.p22[0] * X**2
            + self.p22[1] * X * Y
            + self.p22[2] * Y**2
            + self.p22[3] * X
            + self.p22[4] * Y
            + self.p22[5]
        )

        ax.contour(X, Y, l1eq, [0], colors='blue')
        ax.contour(X, Y, l2eq, [0], colors='blue')
        ax.contour(X, Y, l3eq, [0], colors='red')
        ax.contour(X, Y, l4eq, [0], colors='red')
        plt.show(block=False)

        # If it gets here there are no errors
        self.assertAlmostEqual(0, 0)


if __name__.__contains__("__main__"):
    unittest.main()
