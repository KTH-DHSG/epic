import numpy as np


def gamma_lifting_operator(c1, c2):
    """
    Gamma operator for lifted coordinates.

    :param c1: vector 1
    :type c1: np.ndarray
    :param c2: vector 2
    :type c2: np.ndarray
    :return: veronese transformed vector
    :rtype: np.ndarray
    """
    xi = c1[0]
    xj = c2[0]

    yi = c1[1]
    yj = c2[1]

    zi = c1[2]
    zj = c2[2]

    veronese_coords = np.array([[xi * xj],
                                [(xi * yj + yi * xj) / 2.0],
                                [yi * yj],
                                [(xi * zj + zi * xj) / 2.0],  # xi * zj +
                                [(yi * zj + zi * yj) / 2.0],
                                [zi * zj]]).reshape(6, 1)
    return veronese_coords


def lift_vec_to_veronese(vec):
    """
    Lifts vector to veronese coordinates

    :param mat: 3x1 vector in cartesian coordinates
    :type mat: np.ndarray
    :return: 6x1 lifted vector
    :rtype: np.ndarray
    """
    return gamma_lifting_operator(vec, vec)


def lift_image_points(points):
    """
    Parallelize lifting points to veronese maps.
    Implemented the relation of Table 2 in
    "Epipolar Geometry of Central Projection
    Systems Using Veronese Maps" by J. P. Barreto
    and K. Daniilidis.

    :param points: image points, 3 x n
    :type points: np.ndarray
    :return: veronese points, 6 x n
    :rtype: np.ndarray
    """

    v_points = np.empty((6, 0))
    for i in range(points.shape[1]):
        v_points = np.append(v_points, lift_vec_to_veronese(points[:, i]), axis=1)

    return v_points


def lift_mat_to_veronese(mat):
    """
    Lifts matrix to veronese coordinates

    :param mat: 3x3 matrix in cartesian coordinates
    :type mat: np.ndarray
    :return: 6x6 lifted matrix
    :rtype: np.ndarray
    """
    V_D = np.diag(np.array([1, 2, 1, 2, 2, 1]))

    gamma_11 = gamma_lifting_operator(mat[:, [0]], mat[:, [0]])
    gamma_12 = gamma_lifting_operator(mat[:, [0]], mat[:, [1]])
    gamma_13 = gamma_lifting_operator(mat[:, [0]], mat[:, [2]])
    gamma_22 = gamma_lifting_operator(mat[:, [1]], mat[:, [1]])
    gamma_23 = gamma_lifting_operator(mat[:, [1]], mat[:, [2]])
    gamma_33 = gamma_lifting_operator(mat[:, [2]], mat[:, [2]])

    gamma_mat = np.concatenate((gamma_11, gamma_12, gamma_22, gamma_13, gamma_23, gamma_33), axis=1)
    return np.dot(gamma_mat, V_D)


def set_emat_matrix_with_emat(E, my_cam, nh_cam, calibrated=True):
    """
    Set Essential matrices.

    :param E: Essential matrices tensor
    :type E: 3d numpy array
    :raises ValueError: wrong shape of E
    """

    if len(E.shape) != 2:
        raise ValueError("Expected `E` to have shape (3, 3), "
                         "got {}".format(E.shape))

    # Set D matrix
    V_D = np.diag(np.array([1, 2, 1, 2, 2, 1]))
    E_aug = np.zeros((6, 6))
    E_aug[3:, 3:] = E

    # Set matrix components to be multiplied by lifted points
    if calibrated:
        if my_cam.type == 'perspective':
            yT = np.eye(6)
        elif my_cam.type == 'hyperbolic':
            yT = my_cam.Hc_n_inv_lifted.T @ my_cam.Delta_c
        elif my_cam.type == 'parabolic':
            yT = my_cam.Hc_n_inv_lifted.T @ my_cam.Theta
        elif my_cam.type == 'distortion':
            yT = my_cam.Psi @ np.eye(6)

        if nh_cam.type == 'perspective':
            x = np.eye(6)
        elif nh_cam.type == 'hyperbolic':
            x = nh_cam.Delta_c.T @ nh_cam.Hc_n_inv_lifted
        elif nh_cam.type == 'parabolic':
            x = nh_cam.Theta.T @ nh_cam.Hc_n_inv_lifted
        elif nh_cam.type == 'distortion':
            x = np.eye(6) @ nh_cam.Psi.T
    else:
        if my_cam.type == 'perspective':
            yT = my_cam.K_inv_aug.T
        elif my_cam.type == 'hyperbolic':
            yT = my_cam.Hc_inv_lifted.T @ my_cam.Delta_c
        elif my_cam.type == 'parabolic':
            yT = my_cam.Hc_inv_lifted.T @ my_cam.Theta
        elif my_cam.type == 'distortion':
            yT = my_cam.Psi @ my_cam.K_inv_aug.T

        if nh_cam.type == 'perspective':
            x = nh_cam.K_inv_aug
        elif nh_cam.type == 'hyperbolic':
            x = nh_cam.Delta_c.T @ nh_cam.Hc_inv_lifted
        elif nh_cam.type == 'parabolic':
            x = nh_cam.Theta.T @ nh_cam.Hc_inv_lifted
        elif nh_cam.type == 'distortion':
            x = nh_cam.K_inv_aug @ nh_cam.Psi.T

    # Do the full operation to send the operational matrix
    if nh_cam.type == 'hyperbolic' or my_cam.type == 'hyperbolic':
        V_E1 = yT @ V_D @ lift_mat_to_veronese(E) @ x
    else:
        V_E1 = yT @ E_aug @ x
    return V_E1
