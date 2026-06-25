"""Unit tests for triangulation.reprojection_covariance (ROS-free).

    python -m pytest ros2_ws/src/stereo_handpose_estimation/test/ -q
"""

import numpy as np

from stereo_handpose_estimation.triangulation import (
    _reprojection_jacobian_block,
    dlt,
    reprojection_covariance,
)


def _rectified_pair(f=600.0, cx=320.0, cy=240.0, b=0.10):
    P0 = np.array([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0]], float)
    P1 = np.array([[f, 0, cx, -f * b], [0, f, cy, 0], [0, 0, 1, 0]], float)
    return P0, P1, f, b


def _project(P, X):
    Xh = np.array([X[0], X[1], X[2], 1.0])
    w = P[2] @ Xh
    return np.array([(P[0] @ Xh) / w, (P[1] @ Xh) / w])


def test_jacobian_matches_finite_difference():
    P0, P1, _, _ = _rectified_pair()
    X = np.array([0.05, -0.03, 0.5])
    for P in (P0, P1):
        Ja = _reprojection_jacobian_block(P, X)
        Jf = np.zeros((2, 3))
        h = 1e-6
        for j in range(3):
            dp = X.copy(); dp[j] += h
            dm = X.copy(); dm[j] -= h
            Jf[:, j] = (_project(P, dp) - _project(P, dm)) / (2 * h)
        assert np.max(np.abs(Ja - Jf)) < 1e-5


def test_depth_sigma_matches_closed_form():
    # sigma_Z = (Z^2 / (f*b)) * sqrt(2) * sigma_px  (disparity = u0 - u1).
    P0, P1, f, b = _rectified_pair()
    X = np.array([0.05, -0.03, 0.5])
    Z, sigma_px = X[2], 1.0
    _, axis_sigma = reprojection_covariance(P0, P1, X, sigma_px=sigma_px)
    closed = (Z ** 2 / (f * b)) * np.sqrt(2.0) * sigma_px
    assert abs(axis_sigma[2] - closed) / closed < 0.02


def test_depth_is_least_certain_axis():
    P0, P1, _, _ = _rectified_pair()
    _, s = reprojection_covariance(P0, P1, np.array([0.05, -0.03, 0.5]), 1.0)
    assert s[2] > s[0] and s[2] > s[1]


def test_matches_monte_carlo():
    P0, P1, _, _ = _rectified_pair()
    X = np.array([0.05, -0.03, 0.5])
    sigma_px = 1.0
    _, s = reprojection_covariance(P0, P1, X, sigma_px=sigma_px)
    rng = np.random.default_rng(0)
    p0, p1 = _project(P0, X), _project(P1, X)
    pts = np.array([
        dlt(P0, P1, p0 + rng.normal(0, sigma_px, 2),
            p1 + rng.normal(0, sigma_px, 2))
        for _ in range(50000)
    ])
    emp = pts.std(0)
    assert np.all(np.abs(emp - s) / s < 0.06)


def test_zero_baseline_is_degenerate():
    P0, _, _, _ = _rectified_pair()
    cov, s = reprojection_covariance(P0, P0, np.array([0.05, -0.03, 0.5]))
    assert cov is None and s is None


def test_point_on_image_plane_is_degenerate():
    P0, P1, _, _ = _rectified_pair()
    cov, s = reprojection_covariance(P0, P1, np.array([0.0, 0.0, 1e-13]))
    assert cov is None and s is None
