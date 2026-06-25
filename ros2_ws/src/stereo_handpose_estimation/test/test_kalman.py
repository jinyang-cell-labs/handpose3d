"""Unit tests for the constant-velocity Kalman filter (ROS-free).

    python -m pytest ros2_ws/src/stereo_handpose_estimation/test/ -q
"""

import numpy as np

from stereo_handpose_estimation.kalman import ConstantVelocityKF

DT = 1 / 30.0
# Anisotropic measurement noise like stereo: tight lateral, loose depth.
SIGMA = np.array([0.0008, 0.0008, 0.006])
R = np.diag(SIGMA ** 2)
TRUTH = np.array([0.05, -0.02, 0.5])


def _run_static(q, n=500, seed=0):
    rng = np.random.default_rng(seed)
    kf = ConstantVelocityKF(q, 1.0)
    raw, filt = [], []
    for _ in range(n):
        z = TRUTH + rng.normal(0, SIGMA)
        if not kf.initialized:
            kf.initialize(z, R)
        else:
            kf.predict(DT)
            kf.update(z, R)
        raw.append(z)
        filt.append(kf.position)
    return np.array(raw[100:]), np.array(filt[100:]), kf


def test_reduces_jitter_on_every_axis():
    raw, filt, _ = _run_static(0.5)
    assert np.all(filt.std(0) < raw.std(0))


def test_depth_smoothed_harder_than_lateral():
    # The per-measurement R (large on depth) makes the gain smooth depth more.
    raw, filt, _ = _run_static(0.5)
    red = raw.std(0) / filt.std(0)
    assert red[2] > red[0] and red[2] > red[1]


def test_unbiased_on_static_point():
    _, filt, _ = _run_static(0.5)
    assert np.all(np.abs(filt.mean(0) - TRUTH) < 0.001)


def test_tracks_constant_velocity_below_raw_noise():
    rng = np.random.default_rng(1)
    kf = ConstantVelocityKF(0.5, 1.0)
    vel = np.array([0.2, 0.0, -0.1])
    errs = []
    for k in range(400):
        pos = TRUTH + vel * (k * DT)
        z = pos + rng.normal(0, SIGMA)
        if not kf.initialized:
            kf.initialize(z, R)
        else:
            kf.predict(DT)
            kf.update(z, R)
        if k > 150:
            errs.append(np.linalg.norm(kf.position - pos))
    assert np.mean(errs) < np.linalg.norm(SIGMA)  # beats raw measurement noise
    assert np.allclose(kf.velocity, vel, atol=0.05)


def test_mahalanobis_gate_separates_outlier_from_inlier():
    rng = np.random.default_rng(2)
    kf = ConstantVelocityKF(0.5, 1.0)
    kf.initialize(TRUTH, R)
    for _ in range(20):
        kf.predict(DT)
        kf.update(TRUTH + rng.normal(0, SIGMA), R)
    kf.predict(DT)
    outlier = TRUTH + np.array([0.0, 0.0, 0.3])  # 30 cm depth jump
    assert kf.innovation_mahalanobis2(outlier, R) > 16.27
    inlier = TRUTH + rng.normal(0, SIGMA)
    assert kf.innovation_mahalanobis2(inlier, R) < 16.27


def test_covariance_stays_symmetric_positive_definite():
    _, _, kf = _run_static(0.5)
    assert np.allclose(kf.P, kf.P.T)
    assert np.all(np.linalg.eigvalsh(kf.P) > 0)
