"""Unit tests for the ROS-free wrist-pose pipeline math.

Run from the package root:
    python3 -m pytest test/test_wrist_pose_pipeline.py
"""

import math

import numpy as np
import pytest

from handpose_estimation.triangulation import (
    dlt,
    make_projection_matrix,
    project_point,
    reprojection_error,
    rotation_matrix_to_quaternion,
    triangulate_point,
    weighted_dlt,
)
from handpose_estimation.wrist_pose_pipeline import (
    ConstantVelocityKF,
    OneEuroFilter,
    OneEuroVec,
    ReachabilityShell,
    WristTracker,
    procrustes_fit,
    quat_angle,
    quat_normalize,
    quat_slerp,
    ransac_procrustes,
)


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _stereo_rig(baseline=0.1):
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    P0 = make_projection_matrix(K, np.eye(3), np.zeros(3))
    P1 = make_projection_matrix(K, np.eye(3), np.array([-baseline, 0.0, 0.0]))
    return P0, P1


def _hand_template():
    rng = np.random.default_rng(7)
    tmpl = np.zeros((21, 3))
    tmpl[1:] = rng.uniform(-0.05, 0.15, size=(20, 3))
    return tmpl


# --------------------------------------------------------------------- A1/A2
class TestTriangulation:
    def test_weighted_dlt_matches_legacy_dlt_when_unweighted(self):
        P0, P1 = _stereo_rig()
        X_true = np.array([0.05, -0.02, 0.6])
        p0, p1 = project_point(P0, X_true), project_point(P1, X_true)
        assert np.allclose(weighted_dlt([P0, P1], [p0, p1]), X_true, atol=1e-9)
        assert np.allclose(dlt(P0, P1, p0, p1), X_true, atol=1e-6)

    def test_weighted_dlt_downweights_noisy_view(self):
        P0, P1 = _stereo_rig()
        X_true = np.array([0.05, -0.02, 0.6])
        p0 = project_point(P0, X_true)
        p1 = project_point(P1, X_true) + np.array([8.0, -6.0])  # corrupted
        err_eq = np.linalg.norm(
            weighted_dlt([P0, P1], [p0, p1], [1.0, 1.0]) - X_true)
        err_dn = np.linalg.norm(
            weighted_dlt([P0, P1], [p0, p1], [1.0, 0.1]) - X_true)
        assert err_dn < err_eq

    def test_all_zero_weights_fall_back_to_uniform(self):
        P0, P1 = _stereo_rig()
        X_true = np.array([0.0, 0.0, 0.5])
        p0, p1 = project_point(P0, X_true), project_point(P1, X_true)
        X = weighted_dlt([P0, P1], [p0, p1], [0.0, 0.0])
        assert np.allclose(X, X_true, atol=1e-9)

    def test_reprojection_residual_flags_bad_point(self):
        P0, P1 = _stereo_rig()
        X_true = np.array([0.05, -0.02, 0.6])
        p0 = project_point(P0, X_true)
        p1 = project_point(P1, X_true)
        _, clean_resid, _ = triangulate_point([P0, P1], [p0, p1])
        # NOTE: with only 2 views, a corruption ALONG the epipolar line (x in
        # a rectified pair) is geometrically consistent (it just shifts depth)
        # and leaves zero residual. Only off-epipolar error is detectable.
        _, bad_resid, _ = triangulate_point(
            [P0, P1], [p0, p1 + np.array([0.0, 15.0])])
        assert clean_resid < 1e-6
        assert bad_resid > 3.0
        assert reprojection_error(P0, X_true, p0) < 1e-9


# ------------------------------------------------------------------------- C
class TestProcrustes:
    def test_recovers_known_rigid_transform(self):
        tmpl = _hand_template()
        R_true = _rotation([0.3, 1.0, -0.2], 0.8)
        t_true = np.array([0.1, -0.2, 0.5])
        obs = (R_true @ tmpl.T).T + t_true
        R, t, rmsd = procrustes_fit(tmpl, obs)
        assert np.allclose(R, R_true, atol=1e-9)
        assert np.allclose(t, t_true, atol=1e-9)
        assert rmsd < 1e-9

    def test_determinant_correction_prevents_reflection(self):
        tmpl = _hand_template()
        obs = tmpl.copy()
        obs[:, 2] *= -1.0  # a pure reflection cannot be fit by a rotation
        R, _, _ = procrustes_fit(tmpl, obs)
        assert np.linalg.det(R) == pytest.approx(1.0)

    def test_weights_pull_fit_toward_trusted_points(self):
        tmpl = _hand_template()
        obs = tmpl.copy()
        obs[5] += np.array([0.3, 0.0, 0.0])  # one gross outlier
        w = np.ones(21)
        w[5] = 1e-6
        R, t, _ = procrustes_fit(tmpl, obs, w)
        assert np.allclose(R, np.eye(3), atol=1e-4)
        assert np.allclose(t, 0.0, atol=1e-4)


# ------------------------------------------------------------------------ B2
class TestRansac:
    def test_rejects_outlier_joints(self):
        tmpl = _hand_template()
        R_true = _rotation([0, 0, 1], 0.5)
        t_true = np.array([0.0, 0.1, 0.4])
        obs = (R_true @ tmpl.T).T + t_true
        outliers = [4, 8, 12]  # fingertip occlusions
        obs[outliers] += np.array([0.2, -0.15, 0.1])
        fit = ransac_procrustes(tmpl, obs, iterations=100, sample_size=4,
                                inlier_thresh=0.02,
                                rng=np.random.default_rng(0))
        assert fit is not None
        assert not fit["inliers"][outliers].any()
        assert fit["n_inliers"] == 18
        assert np.allclose(fit["R"], R_true, atol=1e-6)
        assert np.allclose(fit["t"], t_true, atol=1e-6)

    def test_returns_none_with_too_few_valid_joints(self):
        tmpl = _hand_template()
        obs = np.full((21, 3), np.nan)
        obs[:3] = tmpl[:3]
        assert ransac_procrustes(tmpl, obs, sample_size=4) is None


# ------------------------------------------------------------------------ B1
class TestReachabilityShell:
    def setup_method(self):
        self.shell = ReachabilityShell(
            shoulder_left=[-0.18, 0.25, 0.0], shoulder_right=[0.18, 0.25, 0.0],
            d_min=0.10, d_max=0.85, forward_axis=[0, 0, 1], behind_margin=0.10)

    def test_gates_by_distance_and_direction(self):
        ok, _ = self.shell.check([0.0, 0.0, 0.4], "Right")
        assert ok
        assert self.shell.check([0.18, 0.25, 0.02], "Right") == (False, "too_close")
        assert self.shell.check([0.18, 0.25, 1.5], "Right") == (False, "too_far")
        assert self.shell.check([0.18, 0.25, -0.5], "Right") == (False, "behind_head")

    def test_filter_joints_masks_nans(self):
        joints = np.array([[0.0, 0.0, 0.4], [np.nan, 0.0, 0.4], [0.18, 0.25, 5.0]])
        mask = self.shell.filter_joints(joints, "Left")
        assert mask.tolist() == [True, False, False]


# ------------------------------------------------------------------------- D
class TestKalman:
    def test_tracks_constant_velocity_and_gates_outlier(self):
        kf = ConstantVelocityKF(process_noise_pos=10.0,
                                measurement_noise_pos=1e-4, gate=11.345)
        q = np.array([0.0, 0.0, 0.0, 1.0])
        dt, v = 1 / 30.0, np.array([0.5, 0.0, 0.0])
        pos = np.zeros(3)
        for _ in range(60):
            pos = pos + v * dt
            out_pos, _ = kf.update(dt, pos, q)
        assert np.allclose(out_pos, pos, atol=5e-3)
        # A 1 m teleport must be rejected: output stays near the track.
        out_pos, _ = kf.update(dt, pos + np.array([1.0, 0.0, 0.0]), q)
        assert np.linalg.norm(out_pos - pos) < 0.1

    def test_orientation_slerp_moves_toward_measurement(self):
        kf = ConstantVelocityKF(1.0, 1e-4, orientation_lpf=0.5)
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        qz = rotation_matrix_to_quaternion(_rotation([0, 0, 1], 0.4))
        kf.update(0.033, np.zeros(3), q0)
        _, q = kf.update(0.033, np.zeros(3), qz)
        assert 0.0 < quat_angle(q0, q) < quat_angle(q0, qz) + 1e-9


class TestQuaternions:
    def test_slerp_endpoints_and_shorter_arc(self):
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        q1 = rotation_matrix_to_quaternion(_rotation([0, 1, 0], 1.0))
        assert np.allclose(quat_slerp(q0, q1, 0.0), q0)
        assert np.allclose(np.abs(quat_slerp(q0, q1, 1.0)), np.abs(q1), atol=1e-9)
        # -q1 represents the same rotation; slerp must take the short way.
        mid_a = quat_slerp(q0, q1, 0.5)
        mid_b = quat_slerp(q0, -q1, 0.5)
        assert quat_angle(mid_a, mid_b) < 1e-9

    def test_normalize_handles_zero(self):
        assert np.allclose(quat_normalize([0, 0, 0, 0]), [0, 0, 0, 1])


# ------------------------------------------------------------------------- E
class TestOneEuro:
    def test_smooths_jitter(self):
        rng = np.random.default_rng(1)
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        xs = 0.5 + rng.normal(0.0, 0.01, size=200)
        ys = [f(i / 30.0, x) for i, x in enumerate(xs)]
        assert np.std(ys[50:]) < np.std(xs[50:])

    def test_vector_variant_filters_componentwise(self):
        f = OneEuroVec(3, min_cutoff=1.0)
        out = f(0.0, np.array([1.0, 2.0, 3.0]))
        assert np.allclose(out, [1.0, 2.0, 3.0])
        out = f(1 / 30.0, np.array([1.1, 2.0, 3.0]))
        assert 1.0 < out[0] < 1.1


# ------------------------------------------------------------ end-to-end-ish
def _tracker(flags=None):
    params = {
        "nominal_fps": 30.0,
        "procrustes.min_joints": 6,
        "ransac.iterations": 100,
        "ransac.sample_size": 4,
        "ransac.inlier_thresh": 0.02,
        "kalman.process_noise_pos": 10.0,
        "kalman.measurement_noise_pos": 1e-4,
        "kalman.gate_threshold": 11.345,
        "kalman.orientation_lpf": 0.7,
        "kalman.max_coast_frames": 5,
        "one_euro.min_cutoff": 1.0,
        "one_euro.beta": 0.007,
        "one_euro.d_cutoff": 1.0,
    }
    all_flags = {"reachability_gate": False, "procrustes": True,
                 "ransac": True, "kalman": True, "one_euro": True}
    if flags:
        all_flags.update(flags)
    return WristTracker("Right", _hand_template(), all_flags, params)


class TestWristTracker:
    def test_static_hand_converges_to_truth(self):
        tr = _tracker()
        R_true = _rotation([1, 0, 0], 0.3)
        t_true = np.array([0.05, 0.1, 0.45])
        obs = (R_true @ tr.template.T).T + t_true
        rng = np.random.default_rng(3)
        res = None
        for i in range(60):
            noisy = obs + rng.normal(0.0, 0.002, size=obs.shape)
            res = tr.update(i / 30.0, noisy, np.ones(21), 0.9)
        assert res is not None and res["valid"]
        # Template wrist is at the origin -> wrist position == t_true.
        assert np.allclose(res["pos"], t_true, atol=0.01)
        q_true = rotation_matrix_to_quaternion(R_true)
        assert quat_angle(res["quat"], q_true) < 0.1

    def test_coasts_through_short_dropout_then_resets(self):
        tr = _tracker()
        obs = tr.template + np.array([0.0, 0.0, 0.4])
        for i in range(10):
            tr.update(i / 30.0, obs, np.ones(21), 0.9)
        nan = np.full((21, 3), np.nan)
        res = tr.update(11 / 30.0, nan, np.zeros(21), 0.0)
        assert res is not None and not res["valid"]  # coasting
        for i in range(12, 18):  # exceed max_coast_frames=5
            res = tr.update(i / 30.0, nan, np.zeros(21), 0.0)
        assert res is None
        assert not tr.kf.initialised

    def test_procrustes_disabled_falls_back_to_raw_wrist(self):
        tr = _tracker({"procrustes": False, "ransac": False,
                       "kalman": False, "one_euro": False})
        obs = tr.template + np.array([0.0, 0.0, 0.4])
        res = tr.update(0.0, obs, np.ones(21), 0.9)
        assert np.allclose(res["pos"], [0.0, 0.0, 0.4])
        assert np.allclose(res["quat"], [0.0, 0.0, 0.0, 1.0])

    def test_reachability_gate_blocks_out_of_shell_hand(self):
        shell = ReachabilityShell([-0.18, 0.25, 0.0], [0.18, 0.25, 0.0],
                                  d_min=0.10, d_max=0.85)
        tr = _tracker({"reachability_gate": True})
        tr.shell = shell
        obs = tr.template + np.array([0.0, 0.0, 3.0])  # 3 m away: unreachable
        assert tr.update(0.0, obs, np.ones(21), 0.9) is None
