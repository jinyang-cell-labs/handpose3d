"""Minimal stereo triangulation helpers for stereo_handpose_estimation.

Self-contained (pure numpy) subset of the utilities in handpose_estimation —
just what is needed to triangulate a single 2D point correspondence into a 3D
world point and to convert a rotation matrix to a quaternion.
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


def _reprojection_jacobian_block(P, Xw):
    """2x3 Jacobian d(u, v)/d(X, Y, Z) of a pinhole projection at ``Xw``.

    For P (3, 4) and world point Xw (3,), with homogeneous Xh = [X, Y, Z, 1]:

        w = P[2] . Xh,   u = (P[0] . Xh) / w,   v = (P[1] . Xh) / w
        du/dXj = (P[0, j] - u * P[2, j]) / w     (j = 0..2)
        dv/dXj = (P[1, j] - v * P[2, j]) / w

    Only columns 0:3 of each row enter — the 4th (translation) column cancels in
    the quotient rule. Returns None if the point is on/behind the image plane
    (w ~ 0), where the projection derivative blows up.
    """
    Xh = np.array([Xw[0], Xw[1], Xw[2], 1.0])
    w = float(P[2] @ Xh)
    if abs(w) < 1e-12:
        return None
    u = float(P[0] @ Xh) / w
    v = float(P[1] @ Xh) / w
    du = (P[0, 0:3] - u * P[2, 0:3]) / w
    dv = (P[1, 0:3] - v * P[2, 0:3]) / w
    return np.vstack([du, dv])


def reprojection_covariance(P0, P1, Xw, sigma_px=0.5, cond_max=1e12):
    """Linearized covariance of a triangulated point from pixel noise.

    First-order (Gauss-Newton) propagation of isotropic pixel noise into the 3D
    point. The reprojection cost's curvature at ``Xw`` is the Fisher information
    ``J^T J`` (J = the 4x3 stack of both cameras' 2x3 reprojection Jacobians),
    and the point covariance is its scaled inverse:

        Cov = sigma_px**2 * (J^T J)^-1

    assuming independent pixel noise of std ``sigma_px`` on every image
    coordinate. Eigen-decomposing Cov gives the uncertainty ellipsoid; for a
    stereo pair its long axis is the viewing ray (depth), so the depth diagonal
    dominates. Units follow ``Xw`` (the projection matrices' solve frame); the
    caller scales to metres.

    Args:
        P0, P1: (3, 4) projection matrices used to triangulate ``Xw``.
        Xw: (3,) triangulated point, in the frame P0/P1 project into.
        sigma_px: per-coordinate pixel-noise std (pixels). For a centroid of
            many landmarks use the *observed* centroid jitter, not
            sigma_single / sqrt(N) (landmark errors are correlated).
        cond_max: treat as degenerate if cond(J^T J) exceeds this.

    Returns:
        (cov, axis_sigma): (3, 3) covariance and (3,) per-axis std sqrt(diag),
        or (None, None) if the geometry is degenerate (point behind a camera,
        or near-parallel rays / tiny baseline -> near-singular J^T J).
    """
    J0 = _reprojection_jacobian_block(np.asarray(P0, dtype=float), Xw)
    J1 = _reprojection_jacobian_block(np.asarray(P1, dtype=float), Xw)
    if J0 is None or J1 is None:
        return None, None
    J = np.vstack([J0, J1])              # (4, 3)
    JtJ = J.T @ J                        # (3, 3) Gauss-Newton Hessian / Fisher
    cond = np.linalg.cond(JtJ)
    if not np.isfinite(cond) or cond > cond_max:
        return None, None
    cov = (sigma_px ** 2) * np.linalg.inv(JtJ)
    axis_sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return cov, axis_sigma


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
