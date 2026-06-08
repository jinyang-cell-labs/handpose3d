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
