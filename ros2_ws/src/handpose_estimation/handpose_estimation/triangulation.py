"""Stereo triangulation helpers (ported from the original handpose3d utils.py).

The projection matrix for a camera is ``P = K @ [R | t]`` where:

* ``K`` (3x3) is the intrinsic matrix — taken from the camera_info topic.
* ``R`` (3x3), ``t`` (3,) are the camera's extrinsic pose in the world frame —
  taken from the handpose_estimation extrinsics config.

Given a 2D pixel correspondence in two such cameras, :func:`dlt` recovers the
3D world point via the Direct Linear Transform.
"""

import numpy as np


def make_projection_matrix(K, R, t):
    """Build a 3x4 projection matrix from intrinsics and extrinsics.

    Args:
        K: (3, 3) intrinsic matrix.
        R: (3, 3) rotation (world -> camera).
        t: (3,) translation (world -> camera).
    """
    K = np.asarray(K, dtype=float).reshape(3, 3)
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3, 1)
    Rt = np.hstack([R, t])  # 3x4
    return K @ Rt


def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to an (x, y, z, w) quaternion.

    Shepperd's method — numerically stable across all rotations and avoids a
    dependency on tf_transformations / scipy.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def dlt(P1, P2, point1, point2):
    """Direct Linear Transform: triangulate one 3D point from two views.

    Args:
        P1, P2: (3, 4) projection matrices.
        point1, point2: (x, y) pixel coordinates in each view.

    Returns:
        (3,) ndarray world-space point.
    """
    A = [
        point1[1] * P1[2, :] - P1[1, :],
        P1[0, :] - point1[0] * P1[2, :],
        point2[1] * P2[2, :] - P2[1, :],
        P2[0, :] - point2[0] * P2[2, :],
    ]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A
    # SVD; the solution is the singular vector with the smallest singular value.
    _, _, Vh = np.linalg.svd(B)
    return Vh[3, 0:3] / Vh[3, 3]
