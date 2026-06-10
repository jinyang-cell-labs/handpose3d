"""Stereo triangulation helpers (ported from the original handpose3d utils.py).

The projection matrix for a camera is ``P = K @ [R | t]`` where:

* ``K`` (3x3) is the intrinsic matrix — taken from the camera_info topic.
* ``R`` (3x3), ``t`` (3,) are the camera's extrinsic pose in the world frame —
  taken from the handpose_estimation extrinsics config.

Given a 2D pixel correspondence in two such cameras, :func:`dlt` recovers the
3D world point via the Direct Linear Transform.

New in this revision (wrist-pose pipeline stages A1/A2):

* :func:`weighted_dlt`       — confidence-weighted N-view DLT (Stage A1)
* :func:`project_point`      — project a 3D point into a view
* :func:`reprojection_error` — per-view pixel residual for a 3D point (Stage A2)
* :func:`triangulate_point`  — wrapper returning point + residuals
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


# --------------------------------------------------------------------------- #
#  Stage A1 / A2 — confidence-weighted DLT + reprojection residuals            #
# --------------------------------------------------------------------------- #
def _dlt_rows(projections, points, weights=None):
    """Assemble the 2N x 4 DLT matrix A.

    For each view i with projection P_i and pixel (u_i, v_i):
        row 2i   = u_i * P_i[2, :] - P_i[0, :]
        row 2i+1 = v_i * P_i[2, :] - P_i[1, :]
    If weights are supplied, both rows of view i are multiplied by w_i
    (confidence-weighted DLT: ``(w ∘ A) x = 0``).
    """
    n = len(projections)
    if weights is None:
        weights = np.ones(n, dtype=float)
    rows = []
    for P, pt, w in zip(projections, points, weights):
        P = np.asarray(P, dtype=float)
        u, v = float(pt[0]), float(pt[1])
        rows.append(w * (u * P[2, :] - P[0, :]))
        rows.append(w * (v * P[2, :] - P[1, :]))
    return np.asarray(rows, dtype=float)


def _solve_dlt(A):
    """Solve A x = 0 for homogeneous x by SVD; return cartesian 3-vector.

    SVD of A directly is more numerically robust than eigendecomposition of
    A^T A (used by the legacy :func:`dlt`).
    """
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    if abs(X[3]) < 1e-12:
        return np.full(3, np.nan)
    return X[:3] / X[3]


def weighted_dlt(projections, points, weights=None):
    """Confidence-weighted N-view DLT (Stage A1).

    Args:
        projections: list of (3, 4) projection matrices, length N >= 2.
        points: list of (2,) pixel coordinates, length N.
        weights: optional length-N non-negative per-view confidence weights.
            None means equal weighting (== unweighted DLT).

    Returns:
        (3,) triangulated 3D point (NaN if degenerate).

    Each camera contributes two rows to A; both are scaled by that view's
    scalar weight w_i, then the homogeneous system (w ∘ A) x = 0 is solved by
    SVD. Supports N >= 2 even though the head-mounted rig has 2 cameras.
    """
    if len(projections) < 2:
        raise ValueError("weighted_dlt requires at least 2 views")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        # Guard against an all-zero weight vector collapsing the system.
        if not np.any(weights > 0):
            weights = np.ones(len(projections))
    A = _dlt_rows(projections, points, weights)
    return _solve_dlt(A)


def project_point(P, X):
    """Project a 3D cartesian point X into a view with projection matrix P.

    Returns the (2,) pixel coordinate (or NaN if on the principal plane).
    """
    P = np.asarray(P, dtype=float)
    Xh = np.append(np.asarray(X, dtype=float), 1.0)
    x = P @ Xh
    if abs(x[2]) < 1e-12:
        return np.full(2, np.nan)
    return x[:2] / x[2]


def reprojection_error(P, X, point):
    """Pixel reprojection error for one 3D point in one view (Stage A2)."""
    proj = project_point(P, X)
    return float(np.linalg.norm(proj - np.asarray(point, dtype=float)))


def triangulate_point(projections, points, weights=None):
    """Triangulate one point and return ``(X, mean_resid, per_view_resid)``.

    Combines Stage A1 (weighted DLT) and Stage A2 (per-view reprojection
    residual). The mean residual is a per-joint trust signal used downstream
    by RANSAC and by the reprojection weighting scheme.
    """
    X = weighted_dlt(projections, points, weights)
    resid = np.array(
        [reprojection_error(P, X, pt) for P, pt in zip(projections, points)],
        dtype=float,
    )
    return X, float(np.nanmean(resid)), resid
