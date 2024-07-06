import numpy as np
import cv2 as cv
import casadi as ca
import pyquaternion as pq


def get_parabola_intersections(p1, p2):
    """
    Calculate intersection of two parabolas (p1 and p2 are in the lifted
    Veronese coordinates).

    :param p1: parabola 1
    :type p1: np.ndarray
    :param p2: parabola 2
    :type p2: np.ndarray
    :raises ValueError: no intersection was found
    :return: intersection point(s)
    :rtype: np.ndarray
    """

    # p = [a, b, c, d, e, f] ->  (a x**2 + b xy + c y**2 + d x + e y + f) = 0

    # Get coeffs
    k0 = p1[0]
    k1 = p1[1]
    k2 = p1[2]
    k3 = p1[3]
    k4 = p1[4]
    k5 = p1[5]

    m0 = p2[0]
    m1 = p2[1]
    m2 = p2[2]
    m3 = p2[3]
    m4 = p2[4]
    m5 = p2[5]

    c1 = (
        k0**2 * m2**2
        - k0 * k1 * m1 * m2
        - 2 * k0 * k2 * m0 * m2
        + k0 * k2 * m1**2
        + k1**2 * m0 * m2
        - k1 * k2 * m0 * m1
        + k2**2 * m0**2
    ) / m0**2
    c2 = (
        2 * k0**2 * m2 * m4
        - k0 * k1 * m1 * m4
        - k0 * k1 * m2 * m3
        - 2 * k0 * k2 * m0 * m4
        + 2 * k0 * k2 * m1 * m3
        - k0 * k3 * m1 * m2
        - 2 * k0 * k4 * m0 * m2
        + k0 * k4 * m1**2
        + k1**2 * m0 * m4
        - k1 * k2 * m0 * m3
        + 2 * k1 * k3 * m0 * m2
        - k1 * k4 * m0 * m1
        - k2 * k3 * m0 * m1
        + 2 * k2 * k4 * m0**2
    ) / m0**2
    c3 = (
        2 * k0**2 * m2 * m5
        + k0**2 * m4**2
        - k0 * k1 * m1 * m5
        - k0 * k1 * m3 * m4
        - 2 * k0 * k2 * m0 * m5
        + k0 * k2 * m3**2
        - k0 * k3 * m1 * m4
        - k0 * k3 * m2 * m3
        - 2 * k0 * k4 * m0 * m4
        + 2 * k0 * k4 * m1 * m3
        - 2 * k0 * k5 * m0 * m2
        + k0 * k5 * m1**2
        + k1**2 * m0 * m5
        + 2 * k1 * k3 * m0 * m4
        - k1 * k4 * m0 * m3
        - k1 * k5 * m0 * m1
        - k2 * k3 * m0 * m3
        + 2 * k2 * k5 * m0**2
        + k3**2 * m0 * m2
        - k3 * k4 * m0 * m1
        + k4**2 * m0**2
    ) / m0**2
    c4 = (
        2 * k0**2 * m4 * m5
        - k0 * k1 * m3 * m5
        - k0 * k3 * m1 * m5
        - k0 * k3 * m3 * m4
        - 2 * k0 * k4 * m0 * m5
        + k0 * k4 * m3**2
        - 2 * k0 * k5 * m0 * m4
        + 2 * k0 * k5 * m1 * m3
        + 2 * k1 * k3 * m0 * m5
        - k1 * k5 * m0 * m3
        + k3**2 * m0 * m4
        - k3 * k4 * m0 * m3
        - k3 * k5 * m0 * m1
        + 2 * k4 * k5 * m0**2
    ) / m0**2
    c5 = (
        k0**2 * m5**2
        - k0 * k3 * m3 * m5
        - 2 * k0 * k5 * m0 * m5
        + k0 * k5 * m3**2
        + k3**2 * m0 * m5
        - k3 * k5 * m0 * m3
        + k5**2 * m0**2
    ) / m0**2
    sol_y = np.roots([c1, c2, c3, c4, c5])
    sol_y = np.array(sol_y[np.isreal(sol_y)])
    # If we have no intersections, leave
    if sol_y.size == 0:
        return None

    # Otherwise, collect y solutions
    sol_list = []
    for i, x in enumerate(sol_y):
        y = sol_y[i]

        # Solution pair: (x1, y) and (x2, y)
        # try first solution of x
        x = (
            -m1 * y
            - m3
            - np.sqrt(
                -4 * m0 * m2 * y**2
                - 4 * m0 * m4 * y
                - 4 * m0 * m5
                + m1**2 * y**2
                + 2 * m1 * m3 * y
                + m3**2
            )
        ) / (2 * m0)
        p1 = m0 * x**2 + m1 * x * y + m2 * y**2 + m3 * x + m4 * y + m5
        p2 = k0 * x**2 + k1 * x * y + k2 * y**2 + k3 * x + k4 * y + k5

        if np.abs(p1) < 1e-9 and np.abs(p2) < 1e-9:
            sol_list.append(np.array([x, y]))
            continue

        # try second solution of x
        x = (
            -m1 * y
            - m3
            + np.sqrt(
                -4 * m0 * m2 * y**2
                - 4 * m0 * m4 * y
                - 4 * m0 * m5
                + m1**2 * y**2
                + 2 * m1 * m3 * y
                + m3**2
            )
        ) / (2 * m0)
        p1 = m0 * x**2 + m1 * x * y + m2 * y**2 + m3 * x + m4 * y + m5
        p2 = k0 * x**2 + k1 * x * y + k2 * y**2 + k3 * x + k4 * y + k5
        if np.abs(p1) < 1e-9 and np.abs(p2) < 1e-9:
            sol_list.append(np.array([x, y]))

    sol_array = np.concatenate([sol_list], axis=0).T
    return sol_array


def get_lines_intersection(l1, l2):
    """
    Get the intersection point between two lines if it exists.

    :param l1: a quadric line
    :type l1: np.ndarray(3, 1)
    :param l2: another quadric line
    :type l2: np.ndarray(3, 1)
    :return: intersection point
    :rtype: np.ndarray(3, 1)
    """
    # li = [a, b, c, d, e, f] ->  (a x**2 + b xy + c y**2 + d x + e y + f) = 0
    pt_int = np.cross(l1[3:], l2[3:])
    pt_int = pt_int / pt_int[2]
    return pt_int[0:2].reshape((2, 1))


def is_line(quadric: np.ndarray) -> bool:
    """
    Check if a line is valid.

    :param l: a quadric description of a parabola
    :type l: np.ndarray(6, 1)
    :return: True if the line is valid, False otherwise
    :rtype: bool
    """
    # quadric = [a, b, c, d, e, f] ->  (a x**2 + b xy + c y**2 + d x + e y + f) = 0
    return quadric[0] < 1e-9 and quadric[1] < 1e-9 and quadric[2] < 1e-9


def get_closest_relative_pose(p1, p2, Rd, td):
    """
    Get the closest relative pose between two cameras.

    :param p1: points in camera 1
    :type p1: np.ndarray
    :param p2: points in camera
    :type p2: np.ndarray
    :param Rd: reference rotation
    :type Rd: np.ndarray
    :param td: reference translation
    :type td: np.ndarray
    :return: relative pose R and t
    :rtype: np.ndarray, np.ndarray
    """
    [E, mask] = cv.findEssentialMat(
        p1, p2, method=cv.RANSAC, prob=0.999, threshold=0.001
    )

    max_pe = np.inf
    max_re = np.inf

    sol = {"R": None, "t": None}

    for i in range(int(E.shape[0] / 3)):
        Ei = E[i * 3: i * 3 + 3, :]
        [_, R, t, _] = cv.recoverPose(Ei, p1, p2)
        # Select the closest pose
        r, _ = cv.Rodrigues(R.dot(Rd.T))
        re = np.linalg.norm(r)
        # Pose error
        pe = np.linalg.norm(td - t)
        if pe <= max_pe and re <= max_re:
            max_pe = pe
            max_re = re
            sol["R"] = R
            sol["t"] = t

    return sol["R"].T, sol["t"]


def get_closest_essential_matrix(p1, p2, Ed):
    """
    Get the closest relative pose between two cameras.

    :param p1: points in camera 1
    :type p1: np.ndarray
    :param p2: points in camera
    :type p2: np.ndarray
    :param Ed: reference Emat
    :type Ed: np.ndarray
    :return: closest essential matrix
    :rtype: np.ndarray
    """
    [E, mask] = cv.findEssentialMat(
        p1, p2, method=cv.RANSAC, prob=0.999, threshold=0.001
    )

    max_pe = np.inf
    sol_E = np.zeros((3, 3))

    for i in range(int(E.shape[0] / 3)):
        Ei = E[i * 3 : i * 3 + 3, :]
        error = np.linalg.norm(Ei - Ed)
        if error < max_pe:
            max_pe = error
            sol_E = Ei

    return sol_E


def curve_from_vec_to_mat(vec):
    """
    Helper function to convert a vector to a matrix
    """
    if type(vec) == ca.casadi.MX:
        M = ca.MX.zeros(3, 3)
    elif type(vec) == ca.casadi.DM:
        M = ca.DM.zeros(3, 3)
    elif type(vec) == np.ndarray:
        M = np.zeros((3, 3))
    else:
        raise ValueError("Unknown type for vec")

    M[0, 0] = vec[0] * 2
    M[0, 1] = vec[1]
    M[0, 2] = vec[3]

    M[1, 0] = vec[1]
    M[1, 1] = vec[2] * 2
    M[1, 2] = vec[4]

    M[2, 0] = vec[3]
    M[2, 1] = vec[4]
    M[2, 2] = vec[5] * 2
    return M


def get_range_to(cam1, cam2):
    """
    Get range between current agent and desired neighbor.

    :param neighbor: neighbor agent
    :type neighbor: GenericCam
    """

    my_pos = cam1.get_position()
    n_pos = cam2.get_position()
    return np.linalg.norm(my_pos - n_pos, ord=2)


def r_mat(q):
    """
    Generate a symbolic rotation matrix from unit quaternion,
    following Trawny's paper transform. That is:
    p^G = r_mat(q) @ p^L

    Indirect Kalman Filter for 3D Attitude Estimation,
    by Nikolas Trawny and Stergios I. Roumeliotis.

    Note that in the reference the rotation matrix is
    transposed.

    :param q: unit quaternion
    :type q: ca.MX
    :return: rotation matrix, SO(3)
    :rtype: ca.MX
    """

    if type(q) == ca.casadi.DM:
        Rmat = ca.DM(3, 3)
    elif type(q) == ca.casadi.MX:
        Rmat = ca.MX(3, 3)
    elif type(q) == np.ndarray:
        Rmat = np.zeros((3, 3))
    else:
        raise ValueError("Unknown type for q")

    # Extract states
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]

    Rmat[0, 0] = 1.0 - 2 * qy**2 - 2 * qz**2
    Rmat[0, 1] = 2 * qx * qy - 2 * qz * qw
    Rmat[0, 2] = 2 * qx * qz + 2 * qy * qw

    Rmat[1, 0] = 2 * qx * qy + 2 * qz * qw
    Rmat[1, 1] = 1.0 - 2 * qx**2 - 2 * qz**2
    Rmat[1, 2] = 2 * qy * qz - 2 * qx * qw

    Rmat[2, 0] = 2 * qx * qz - 2 * qy * qw
    Rmat[2, 1] = 2 * qy * qz + 2 * qx * qw
    Rmat[2, 2] = 1.0 - 2 * qx**2 - 2 * qy**2

    return Rmat


def skew(v):
    """
    Returns the skew matrix of a vector v

    :param v: vector
    :type v: ca.MX
    :return: skew matrix of v
    :rtype: ca.MX
    """

    if type(v) == ca.casadi.DM:
        sk = ca.DM(3, 3)
    elif type(v) == ca.casadi.MX:
        sk = ca.MX(3, 3)
    elif type(v) == np.ndarray:
        sk = np.zeros((3, 3))
    else:
        raise ValueError("Unknown type for v")

    # Extract vector components
    x = v[0]
    y = v[1]
    z = v[2]

    sk[0, 1] = -z
    sk[1, 0] = z
    sk[0, 2] = y
    sk[2, 0] = -y
    sk[1, 2] = -x
    sk[2, 1] = x

    return sk


def inv_skew(sk):
    """
    Retrieve the vector from the skew-symmetric matrix.

    :param sk: skew symmetric matrix
    :type sk: ca.MX
    :return: vector corresponding to SK matrix
    :rtype: ca.MX
    """
    if type(sk) == ca.casadi.DM:
        v = ca.DM.zeros(3, 1)
    elif type(sk) == ca.casadi.MX:
        v = ca.MX.zeros(3, 1)
    elif type(sk) == np.ndarray:
        v = np.zeros((3, 1))
    else:
        raise ValueError("Unknown type for sk")

    v[0] = sk[2, 1]
    v[1] = sk[0, 2]
    v[2] = sk[1, 0]

    return v


def xi_mat(q):
    """
    Generate the matrix for quaternion dynamics Xi,
    from Trawney's Quaternion tutorial.

    :param q: unit quaternion
    :type q: ca.MX
    :return: Xi matrix
    :rtype: ca.MX
    """
    if type(q) == ca.casadi.DM:
        Xi = ca.DM(4, 3)
    elif type(q) == ca.casadi.MX:
        Xi = ca.MX(4, 3)
    elif type(q) == np.ndarray:
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

    # Return Xi matrix
    # Xi[0, 0] = -qw
    # Xi[0, 1] = qz
    # Xi[0, 2] = -qy

    # Xi[1, 0] = -qz
    # Xi[1, 1] = -qw
    # Xi[1, 2] = qx

    # Xi[2, 0] = qy
    # Xi[2, 1] = -qx
    # Xi[2, 2] = -qw

    # Xi[3, 0] = qx
    # Xi[3, 1] = qy
    # Xi[3, 2] = qz

    return Xi


def get_relative_pose(pose_from, pose_to):
    """
    Get relative pose from one frame to another.

    :param pose_from: [p, q], 1x7 position and quaternion
    :type pose_from: np.ndarray
    :param pose_to: [p, q], 1x7 position and quaternion
    :type pose_to: np.ndarray
    :return: relative rotation and translation vectors
    :rtype: np.ndarray, np.ndarray
    """

    p1 = pose_from[0:3].reshape((3, 1))
    R1 = r_mat(pose_from[3:])

    p2 = pose_to[0:3].reshape((3, 1))
    R2 = r_mat(pose_to[3:])

    R12 = R2.T @ R1
    t12 = R1.T @ (p2 - p1)

    return R12, t12


def get_random_quat_around(q, angle_dev):
    """
    Get random quat with specified angle deviation.

    :param q: original quaternion
    :type q: np.ndarray
    :param angle_dev: angular deviation from q
    :type angle_dev: float, degrees
    :return: rotated quaternion
    :rtype: np.ndarray
    """
    angle = np.random.uniform(low=-angle_dev / 2.0, high=angle_dev / 2.0, size=1)
    q_lib = np.array([q[3], q[0], q[1], q[2]]).reshape(
        4,
    )
    q1 = pq.Quaternion(q_lib)

    axis = np.array(
        [
            np.random.uniform(0, 1, size=1),
            np.random.uniform(0, 1, size=1),
            np.random.uniform(0, 1, size=1),
        ]
    )
    axis = axis / np.linalg.norm(axis)
    q2 = pq.Quaternion(axis=axis, degrees=angle)

    qt = q1.rotate(q2)
    q_rnd = np.array([qt[1], qt[2], qt[3], qt[0]])
    return q_rnd


def get_deviated_quat(q, angle_dev, axis=np.array([0, 0, 1])):
    """
    Get random quat with specified angle deviation.

    :param q: original quaternion
    :type q: np.ndarray
    :param angle_dev: angular deviation from q
    :type angle_dev: float, degrees
    :return: rotated quaternion
    :rtype: np.ndarray
    """
    angle = angle_dev
    q_lib = np.array([q[3], q[0], q[1], q[2]]).reshape(
        4,
    )
    q1 = pq.Quaternion(q_lib)

    axis = axis / np.linalg.norm(axis)
    q2 = pq.Quaternion(axis=axis, degrees=angle)

    qt = q1.rotate(q2)
    q_rnd = np.array([qt[1], qt[2], qt[3], qt[0]])
    return q_rnd


def get_matched_features(w_points_h, my_cam, cam1, cam2=None, visible=False, change_features=False, t_int: tuple = None):
    """
    Get matched features between two cameras.

    :param w_points_h: world points in homogeneous coordinates
    :type w_points_h: np.ndarray
    :param my_cam: camera 1
    :type my_cam: GeneralizedCamera
    :param cam1: camera 2
    :type cam1: GeneralizedCamera
    :param cam2: camera 3, defaults to None
    :type cam2: GeneralizedCamera, optional
    :param visible: use visible points, defaults to False
    :type visible: bool, optional
    :param change_features: change features either randomly or at time trigger, defaults to False
    :type change_features: bool, optional
    :param t_int: tuple of current time and trigger time, defaults to None
    :type t_int: tuple, optional
    :return: match dictionary
    :rtype: dict
    """
    if cam2 is None:
        # Do 2 agent case
        if visible:
            _, _, l_w_pts_idx = cam1.get_visible_points(w_points_h)
            _, _, f1_w_pts_idx = my_cam.get_visible_points(w_points_h)

            # Get common points
            all_common = list(set(l_w_pts_idx).intersection(f1_w_pts_idx))
            if len(all_common) < 5:
                print("Not enough common observations... Skipping test")
                exit()
            common_5 = all_common[0:5]

            # Get common camera observations
            l_px_points_h, l_u_points_h, l_viz_z = cam1.get_image_points(
                w_points_h[:, common_5]
            )
            f1_px_points_h, f1_u_points_h, f1_viz_z = my_cam.get_image_points(
                w_points_h[:, common_5]
            )

        else:
            if change_features:
                # Assign new features
                if t_int is not None:
                    # Time-triggered feature change
                    curr_time = t_int[0]
                    trigger_time = t_int[1]

                    if curr_time < trigger_time:
                        common_5 = [0, 1, 2, 3, 4]
                    else:
                        common_5 = [0, 1, 2, 3, 5]
                else:
                    # Random feature change
                    common_5 = np.random.choice(np.arange(w_points_h.shape[1]), 5, replace=False)
            else:
                # Keep same features
                common_5 = [0, 1, 2, 3, 4]
            l_px_points_h, l_u_points_h, l_viz_z = cam1.get_image_points(
                w_points_h[:, common_5]
            )
            f1_px_points_h, f1_u_points_h, f1_viz_z = my_cam.get_image_points(
                w_points_h[:, common_5]
            )
        mf_dict = {
            "cam1_u_h": l_u_points_h,
            "cam1_viz_z": l_viz_z,
            "my_cam_u_h": f1_u_points_h,
            "my_cam_viz_z": f1_viz_z,
            "neighbors_matched": l_u_points_h,
        }

    else:
        if visible:
            _, _, l_w_pts_idx = cam1.get_visible_points(w_points_h)
            _, _, f1_w_pts_idx = cam2.get_visible_points(w_points_h)
            _, _, f2_w_pts_idx = my_cam.get_visible_points(w_points_h)

            # Get common points
            l_common_f1 = list(set(l_w_pts_idx).intersection(f1_w_pts_idx))
            all_common = list(set(l_common_f1).intersection(f2_w_pts_idx))
            if len(all_common) < 5:
                print("Not enough common observations... Skipping test")
                exit()
            common_5 = all_common[0:5]

            # Get common camera observations
            l_px_points_h, l_u_points_h, l_viz_z = cam1.get_image_points(
                w_points_h[:, common_5]
            )
            f1_px_points_h, f1_u_points_h, f1_viz_z = cam2.get_image_points(
                w_points_h[:, common_5]
            )
            f2_px_points_h, f2_u_points_h, f2_viz_z = my_cam.get_image_points(
                w_points_h[:, common_5]
            )

        else:
            if change_features:
                # Assign new features
                if t_int is not None:
                    # Time-triggered feature change
                    curr_time = t_int[0]
                    trigger_time = t_int[1]

                    if curr_time < trigger_time:
                        common_5 = [0, 1, 2, 3, 4]
                    else:
                        common_5 = [0, 1, 2, 3, 5]
                else:
                    # Random feature change
                    common_5 = np.random.choice(np.arange(w_points_h.shape[1]), 5, replace=False)
            else:
                # Keep same features
                common_5 = [0, 1, 2, 3, 4]
            l_px_points_h, l_u_points_h, l_viz_z = cam1.get_image_points(
                w_points_h[:, common_5]
            )
            f1_px_points_h, f1_u_points_h, f1_viz_z = cam2.get_image_points(
                w_points_h[:, common_5]
            )
            f2_px_points_h, f2_u_points_h, f2_viz_z = my_cam.get_image_points(
                w_points_h[:, common_5]
            )

        # Stack normalized neighbors observations
        neighbors_matched = np.dstack((l_u_points_h, f1_u_points_h))

        # Return dictionary of matched features
        mf_dict = {
            "cam1_u_h": l_u_points_h,
            "cam1_viz_z": l_viz_z,
            "cam2_u_h": f1_u_points_h,
            "cam2_viz_z": f1_viz_z,
            "my_cam_u_h": f2_u_points_h,
            "my_cam_viz_z": f2_viz_z,
            "neighbors_matched": neighbors_matched,
        }

    return mf_dict


def get_matched_features_from_neighbors(w_points_h, my_cam, neighbors: list, visible=False):
    """
    Get matched features between two cameras.

    :param w_points_h: world points in homogeneous coordinates
    :type w_points_h: np.ndarray
    :param my_cam: camera 1
    :type my_cam: GeneralizedCamera
    :param cam1: camera 2
    :type cam1: GeneralizedCamera
    :param cam2: camera 3, defaults to None
    :type cam2: GeneralizedCamera, optional
    :return: match dictionary
    :rtype: dict
    """
    # Pairwise matching
    neighbors_mf_list = []
    for i, neighbor in enumerate(neighbors):
        if visible:
            _, _, l_w_pts_idx = neighbor.get_visible_points(w_points_h)
            _, _, f1_w_pts_idx = my_cam.get_visible_points(w_points_h)

            # Get common points
            all_common = list(set(l_w_pts_idx).intersection(f1_w_pts_idx))
            if len(all_common) < 5:
                print("Not enough common observations... Skipping test")
                exit()
            common_5 = all_common[0:5]

            # Get common camera observations
            l_px_points_h, l_u_points_h, l_viz_z = neighbor.get_image_points(
                w_points_h[:, common_5]
            )
            f1_px_points_h, f1_u_points_h, f1_viz_z = my_cam.get_image_points(
                w_points_h[:, common_5]
            )

        else:
            l_px_points_h, l_u_points_h, l_viz_z = neighbor.get_image_points(
                w_points_h
            )
            f1_px_points_h, f1_u_points_h, f1_viz_z = my_cam.get_image_points(
                w_points_h
            )
        mf_dict = {
            "cam1_u_h": l_u_points_h,
            "cam1_viz_z": l_viz_z,
            "my_cam_u_h": f1_u_points_h,
            "my_cam_viz_z": f1_viz_z,
            "neighbors_matched": l_u_points_h,
        }
        neighbors_mf_list.append(mf_dict)
    return neighbors_mf_list
