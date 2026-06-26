"""Minimal triangulation / projection helpers for handpose_depth_estimation.

Self-contained (pure numpy) subset of the utilities used across the repo — just
what is needed to triangulate a single 2D point correspondence into a 3D world
point, propagate its pixel-noise covariance, and convert a rotation matrix to a
quaternion. Kept local so the package has no cross-package Python import.
"""

import numpy as np


def make_projection_matrix(K, R, t):
    """Build a 3x4 projection matrix ``P = K @ [R | t]``.

    Args:
        K: (3, 3) intrinsic matrix.
        R: (3, 3) rotation (world -> camera).
        t: (3,) translation (world -> camera).
    """
    K = np.asarray(K, dtype=float).reshape(3, 3)
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3, 1)
    return K @ np.hstack([R, t])


def dlt(P1, P2, point1, point2):
    """Triangulate one 3D point from two views via the Direct Linear Transform.

    Args:
        P1, P2: (3, 4) projection matrices.
        point1, point2: (x, y) pixel coordinates in each view.

    Returns:
        (3,) world-space point (NaN if degenerate).
    """
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)
    A = np.array([
        point1[1] * P1[2, :] - P1[1, :],
        P1[0, :] - point1[0] * P1[2, :],
        point2[1] * P2[2, :] - P2[1, :],
        P2[0, :] - point2[0] * P2[2, :],
    ]).reshape((4, 4))
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    if abs(X[3]) < 1e-12:
        return np.full(3, np.nan)
    return X[:3] / X[3]


def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to an (x, y, z, w) quaternion.

    Shepperd's method — numerically stable across all rotations, no scipy /
    tf_transformations dependency.
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
