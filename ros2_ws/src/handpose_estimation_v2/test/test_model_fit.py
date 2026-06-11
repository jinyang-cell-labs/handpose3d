"""Unit tests for the ROS-free model-based fit (v2).

Run from anywhere (conftest.py wires up the package paths):
    python -m pytest ros2_ws/src/handpose_estimation_v2/test/ -q
"""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from handpose_estimation.triangulation import make_projection_matrix
from handpose_estimation_v2.model_fit import (
    bootstrap_pose,
    project_points,
    ransac_rigid_reprojection,
    reprojection_residuals,
    rigid_fit_reprojection,
    so3_exp,
    skew,
)
from handpose_estimation_v2.model_pose_pipeline import ModelFitWristTracker

TEMPLATE_YAML = (
    Path(__file__).resolve().parents[2]
    / "handpose_estimation" / "config" / "hand_template.yaml"
)


@pytest.fixture(scope="module")
def template():
    with open(TEMPLATE_YAML) as f:
        doc = yaml.safe_load(f)
    return np.asarray(doc["template"]["landmarks"], dtype=float)


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = skew(axis)
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _stereo_rig(baseline=0.1):
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    P0 = make_projection_matrix(K, np.eye(3), np.zeros(3))
    P1 = make_projection_matrix(K, np.eye(3), np.array([-baseline, 0.0, 0.0]))
    return [P0, P1]


GT_R = _rotation([0.3, 1.0, 0.2], 0.7)
GT_T = np.array([0.05, -0.03, 0.45])


def _observe(template, R, t, projections, noise=0.0, rng=None):
    """Project the posed template into every view; optional pixel noise."""
    X = (R @ template.T).T + t
    obs = []
    for P in projections:
        uv, _ = project_points(P, X)
        if noise > 0.0:
            uv = uv + rng.normal(0.0, noise, uv.shape)
        obs.append(uv)
    return obs


def _perturbed_init(scale_rot=0.15, dt=(0.03, -0.02, 0.04)):
    R0 = _rotation([1.0, -0.2, 0.5], scale_rot) @ GT_R
    t0 = GT_T + np.asarray(dt)
    return R0, t0


def _rot_angle_deg(Ra, Rb):
    cos = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


# --------------------------------------------------------------- SO(3) utils
def test_so3_exp_matches_rodrigues():
    w = np.array([0.3, -0.5, 0.8])
    R = so3_exp(w)
    R_ref = _rotation(w, np.linalg.norm(w))
    assert np.allclose(R, R_ref, atol=1e-12)
    assert np.allclose(so3_exp(np.zeros(3)), np.eye(3))


def test_project_points_pinhole():
    Ps = _stereo_rig()
    X = np.array([[0.0, 0.0, 1.0], [0.1, -0.05, 0.5]])
    uv, depth = project_points(Ps[0], X)
    assert np.allclose(uv[0], [320.0, 240.0])
    assert np.allclose(uv[1], [320.0 + 600 * 0.2, 240.0 - 600 * 0.1])
    assert np.allclose(depth, [1.0, 0.5])


# ------------------------------------------------------------------ core fit
def test_fit_recovers_pose_noise_free(template):
    Ps = _stereo_rig()
    obs = _observe(template, GT_R, GT_T, Ps)
    R0, t0 = _perturbed_init()
    fit = rigid_fit_reprojection(obs, Ps, template, R0, t0)
    assert fit is not None
    assert _rot_angle_deg(fit["R"], GT_R) < 0.1
    assert np.linalg.norm(fit["t"] - GT_T) < 1e-3
    assert fit["rmse_px"] < 0.1


def test_fit_with_pixel_noise(template):
    rng = np.random.default_rng(7)
    Ps = _stereo_rig()
    obs = _observe(template, GT_R, GT_T, Ps, noise=1.0, rng=rng)
    R0, t0 = _perturbed_init()
    fit = rigid_fit_reprojection(obs, Ps, template, R0, t0)
    assert fit is not None
    assert _rot_angle_deg(fit["R"], GT_R) < 2.0
    assert np.linalg.norm(fit["t"] - GT_T) < 0.01
    assert fit["rmse_px"] < 3.0


def test_huber_downweights_outliers(template):
    rng = np.random.default_rng(3)
    Ps = _stereo_rig()
    obs = _observe(template, GT_R, GT_T, Ps, noise=0.5, rng=rng)
    obs[0][[4, 8, 12]] += 40.0  # three gross outliers in view 0
    R0, t0 = _perturbed_init()
    fit = rigid_fit_reprojection(obs, Ps, template, R0, t0)
    assert fit is not None
    assert _rot_angle_deg(fit["R"], GT_R) < 2.0
    assert np.linalg.norm(fit["t"] - GT_T) < 0.01


def test_single_view_fit_with_prior(template):
    Ps = _stereo_rig()
    obs = _observe(template, GT_R, GT_T, Ps)
    obs[1][:] = np.nan  # view 1 misses the hand entirely
    R0, t0 = _perturbed_init(scale_rot=0.05, dt=(0.01, 0.0, 0.02))
    fit = rigid_fit_reprojection(obs, Ps, template, R0, t0)
    assert fit is not None
    assert _rot_angle_deg(fit["R"], GT_R) < 1.0
    assert np.linalg.norm(fit["t"] - GT_T) < 5e-3
    # Residuals of the unseen view are all NaN.
    assert np.all(np.isnan(fit["resid"][1]))


def test_fit_returns_none_with_too_few_obs(template):
    Ps = _stereo_rig()
    obs = [np.full((21, 2), np.nan), np.full((21, 2), np.nan)]
    obs[0][0] = [320.0, 240.0]
    assert rigid_fit_reprojection(obs, Ps, template, np.eye(3),
                                  np.array([0, 0, 0.5])) is None


def test_partial_joint_visibility(template):
    Ps = _stereo_rig()
    obs = _observe(template, GT_R, GT_T, Ps)
    obs[0][10:] = np.nan  # half the joints missing in view 0
    obs[1][:5] = np.nan   # a few missing in view 1
    R0, t0 = _perturbed_init()
    fit = rigid_fit_reprojection(obs, Ps, template, R0, t0)
    assert fit is not None
    assert _rot_angle_deg(fit["R"], GT_R) < 0.5
    assert np.linalg.norm(fit["t"] - GT_T) < 2e-3


# -------------------------------------------------------------------- RANSAC
def test_ransac_rejects_corrupted_joints(template):
    rng = np.random.default_rng(11)
    Ps = _stereo_rig()
    obs = _observe(template, GT_R, GT_T, Ps, noise=0.5, rng=rng)
    bad = [4, 12, 20]
    for o in obs:  # consistent corruption in BOTH views (3D-plausible outlier)
        o[bad] += rng.normal(0.0, 60.0, (len(bad), 2))
    R0, t0 = _perturbed_init()
    fit = ransac_rigid_reprojection(obs, Ps, template, R0, t0,
                                    rng=np.random.default_rng(0))
    assert fit is not None
    assert not fit["inliers"][bad].any()
    assert fit["n_inliers"] >= 15
    assert _rot_angle_deg(fit["R"], GT_R) < 2.0
    assert np.linalg.norm(fit["t"] - GT_T) < 0.01


# ----------------------------------------------------------------- bootstrap
def test_bootstrap_pose_close_to_gt(template):
    Ps = _stereo_rig()
    obs = _observe(template, GT_R, GT_T, Ps)
    boot = bootstrap_pose(obs, Ps, template)
    assert boot is not None
    R0, t0 = boot
    assert _rot_angle_deg(R0, GT_R) < 1.0
    assert np.linalg.norm(t0 - GT_T) < 5e-3


def test_bootstrap_requires_two_views(template):
    Ps = _stereo_rig()
    obs = _observe(template, GT_R, GT_T, Ps)
    obs[1][:] = np.nan
    assert bootstrap_pose(obs, Ps, template) is None


def test_reprojection_residuals_shape(template):
    Ps = _stereo_rig()
    obs = _observe(template, GT_R, GT_T, Ps)
    obs[1][3] = np.nan
    resid = reprojection_residuals(obs, Ps, template, GT_R, GT_T)
    assert resid.shape == (2, 21)
    assert np.isnan(resid[1, 3])
    assert np.nanmax(resid) < 1e-6


# ------------------------------------------------------------------- tracker
def _tracker_params():
    return {
        "nominal_fps": 30.0,
        "model_fit.max_iters": 20,
        "model_fit.huber_px": 5.0,
        "model_fit.min_joints": 6,
        "model_fit.max_rmse_px": 25.0,
        "single_view.conf_scale": 0.5,
        "ransac.iterations": 30,
        "ransac.sample_size": 4,
        "ransac.inlier_thresh_px": 8.0,
        "kalman.process_noise_pos": 10.0,
        "kalman.measurement_noise_pos": 0.0006,
        "kalman.gate_threshold": 11.345,
        "kalman.orientation_lpf": 0.5,
        "kalman.max_coast_frames": 10,
        "one_euro.min_cutoff": 1.0,
        "one_euro.beta": 0.007,
        "one_euro.d_cutoff": 1.0,
    }


def _tracker_flags(**overrides):
    flags = {
        "model_fit": True,
        "single_view": True,
        "reachability_gate": False,
        "ransac": False,
        "kalman": True,
        "one_euro": True,
    }
    flags.update(overrides)
    return flags


def _make_tracker(template, **flag_overrides):
    return ModelFitWristTracker(
        handedness="Right",
        template=template,
        flags=_tracker_flags(**flag_overrides),
        params=_tracker_params(),
        shell=None,
        rng_seed=0,
        world_scale=1.0,
    )


def test_tracker_tracks_moving_hand(template):
    rng = np.random.default_rng(5)
    Ps = _stereo_rig()
    tracker = _make_tracker(template)
    dt = 1.0 / 30.0
    for k in range(30):
        t_k = GT_T + np.array([0.002, 0.001, -0.001]) * k  # slow drift
        obs = _observe(template, GT_R, t_k, Ps, noise=0.5, rng=rng)
        res = tracker.update(k * dt, obs, Ps, np.array([0.9, 0.9]), 0.9)
        assert res is not None
        assert res["valid"]
    wrist_gt = GT_R @ template[0] + t_k
    assert np.linalg.norm(res["pos"] - wrist_gt) < 0.02


def test_tracker_single_view_continuation(template):
    rng = np.random.default_rng(9)
    Ps = _stereo_rig()
    tracker = _make_tracker(template)
    dt = 1.0 / 30.0
    for k in range(10):  # establish the track with both views
        obs = _observe(template, GT_R, GT_T, Ps, noise=0.5, rng=rng)
        tracker.update(k * dt, obs, Ps, np.array([0.9, 0.9]), 0.9)
    for k in range(10, 20):  # view 1 drops out
        obs = _observe(template, GT_R, GT_T, Ps, noise=0.5, rng=rng)
        obs[1][:] = np.nan
        res = tracker.update(k * dt, obs, Ps, np.array([0.9, 0.0]), 0.9)
        assert res is not None
        assert res["valid"]  # still measuring (monocular), not coasting
    wrist_gt = GT_R @ template[0] + GT_T
    assert np.linalg.norm(res["pos"] - wrist_gt) < 0.03


def test_tracker_coasts_without_single_view(template):
    rng = np.random.default_rng(13)
    Ps = _stereo_rig()
    tracker = _make_tracker(template, single_view=False)
    dt = 1.0 / 30.0
    for k in range(5):
        obs = _observe(template, GT_R, GT_T, Ps, noise=0.5, rng=rng)
        tracker.update(k * dt, obs, Ps, np.array([0.9, 0.9]), 0.9)
    obs = _observe(template, GT_R, GT_T, Ps, noise=0.5, rng=rng)
    obs[1][:] = np.nan
    res = tracker.update(5 * dt, obs, Ps, np.array([0.9, 0.0]), 0.9)
    assert res is not None
    assert not res["valid"]  # coasting on the Kalman prediction


def test_tracker_ransac_path(template):
    rng = np.random.default_rng(17)
    Ps = _stereo_rig()
    tracker = _make_tracker(template, ransac=True)
    dt = 1.0 / 30.0
    for k in range(10):
        obs = _observe(template, GT_R, GT_T, Ps, noise=0.5, rng=rng)
        for o in obs:
            o[[4, 12]] += 50.0  # consistent outlier joints
        res = tracker.update(k * dt, obs, Ps, np.array([0.9, 0.9]), 0.9)
        assert res is not None
        assert res["valid"]
        assert not res["inlier_mask"][[4, 12]].any()
    wrist_gt = GT_R @ template[0] + GT_T
    assert np.linalg.norm(res["pos"] - wrist_gt) < 0.02


def test_tracker_ablation_dlt_procrustes(template):
    rng = np.random.default_rng(21)
    Ps = _stereo_rig()
    tracker = _make_tracker(template, model_fit=False)
    dt = 1.0 / 30.0
    for k in range(10):
        obs = _observe(template, GT_R, GT_T, Ps, noise=0.5, rng=rng)
        res = tracker.update(k * dt, obs, Ps, np.array([0.9, 0.9]), 0.9)
        assert res is not None
        assert res["valid"]
    wrist_gt = GT_R @ template[0] + GT_T
    assert np.linalg.norm(res["pos"] - wrist_gt) < 0.02


def test_tracker_world_scale(template):
    """In extrinsics mode world units != metres; outputs must stay metric."""
    scale = 0.05  # 1 world unit = 5 cm
    Ps = _stereo_rig(baseline=0.1 / scale)
    tracker = ModelFitWristTracker(
        handedness="Right", template=template, flags=_tracker_flags(),
        params=_tracker_params(), shell=None, rng_seed=0, world_scale=scale)
    R_w, t_w = GT_R, GT_T / scale  # pose in world units
    dt = 1.0 / 30.0
    res = None
    for k in range(10):
        obs = _observe(template / scale, R_w, t_w, Ps)
        res = tracker.update(k * dt, obs, Ps, np.array([0.9, 0.9]), 0.9)
    wrist_gt_m = GT_R @ template[0] + GT_T  # metres
    assert np.linalg.norm(res["pos"] - wrist_gt_m) < 0.01
