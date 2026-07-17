"""Synthetic ground-truth tests for the pinhole-radtan / pinhole-equi paths.

Pure math (cv2 + numpy + scipy), no ROS: run with the repo .venv, e.g.
    .venv/bin/python -m pytest ros2_ws/src/calibration_multi_cam/test/
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from calibration_multi_cam import camera_model, se3
from calibration_multi_cam.bundle_adjust import bundle_adjust
from calibration_multi_cam.extrinsics import estimate_target_pose, init_extrinsics
from calibration_multi_cam.intrinsics import K_from_intrinsics, calibrate_intrinsics

W, H = 1280, 800
# ~140 deg FOV equidistant fisheye: f ~= (W/2) / (fov/2 in rad)
K_EQUI = np.array([[520.0, 0.0, 645.0], [0.0, 522.0, 398.0], [0.0, 0.0, 1.0]])
D_EQUI = np.array([0.03, -0.005, 0.002, -0.0005])
# ~70 deg FOV pinhole-radtan
K_RADTAN = np.array([[900.0, 0.0, 640.0], [0.0, 905.0, 400.0], [0.0, 0.0, 1.0]])
D_RADTAN = np.array([-0.28, 0.09, 0.0006, -0.0004])

# 12x12 planar grid of "corners", ~0.66 m across (stand-in for the AprilGrid)
_g = np.stack(np.meshgrid(np.arange(12), np.arange(12)), -1).reshape(-1, 2) * 0.06
OBJP = np.concatenate([_g - _g.mean(0), np.zeros((len(_g), 1))], axis=1)


def _in_view_ids(rvec, tvec, K, D, model, rng, noise=0.05):
    """Project OBJP; return (pids, noisy pixels) of corners inside the image."""
    proj = camera_model.project_points(OBJP, rvec, tvec, K, D, model)
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    z = (OBJP @ R.T + np.asarray(tvec, dtype=np.float64).reshape(1, 3))[:, 2]
    keep = ((proj[:, 0] > 5) & (proj[:, 0] < W - 5)
            & (proj[:, 1] > 5) & (proj[:, 1] < H - 5) & (z > 0.05))
    pids = np.flatnonzero(keep)
    pts = proj[keep] + rng.normal(0.0, noise, (pids.size, 2))
    return pids, pts


def _random_board_pose(rng, max_off_axis=1.05):
    """Board pose spread across the FOV, incl. wide angles for the fisheye."""
    rvec = np.concatenate([rng.uniform(-0.7, 0.7, 2), rng.uniform(-0.5, 0.5, 1)])
    ang = rng.uniform(0.0, max_off_axis)
    az = rng.uniform(0.0, 2.0 * np.pi)
    dist = rng.uniform(0.5, 1.5)
    tvec = dist * np.array([np.sin(ang) * np.cos(az),
                            np.sin(ang) * np.sin(az), np.cos(ang)])
    return rvec, tvec


@pytest.mark.parametrize("model,K,D", [
    (camera_model.RADTAN, K_RADTAN, D_RADTAN),
    (camera_model.EQUI, K_EQUI, D_EQUI),
])
def test_project_points_jac_matches_numeric(model, K, D):
    """Analytic pose jacobian == numeric diff (catches wrong column slicing)."""
    rvec = np.array([0.1, -0.2, 0.3])
    tvec = np.array([0.05, -0.1, 0.9])
    objp = OBJP[::16]
    _, J = camera_model.project_points_jac(objp, rvec, tvec, K, D, model)
    assert J.shape == (2 * len(objp), 6)
    eps = 1e-7
    for i in range(6):
        r1, t1, r0, t0 = rvec.copy(), tvec.copy(), rvec.copy(), tvec.copy()
        (r1 if i < 3 else t1)[i % 3] += eps
        (r0 if i < 3 else t0)[i % 3] -= eps
        p1 = camera_model.project_points(objp, r1, t1, K, D, model)
        p0 = camera_model.project_points(objp, r0, t0, K, D, model)
        num = (p1 - p0).reshape(-1) / (2 * eps)
        assert np.max(np.abs(J[:, i] - num)) < 1e-4


@pytest.mark.parametrize("model,K,D", [
    (camera_model.RADTAN, K_RADTAN, D_RADTAN),
    (camera_model.EQUI, K_EQUI, D_EQUI),
])
def test_calibrate_intrinsics_recovers_ground_truth(model, K, D):
    rng = np.random.default_rng(2)
    observations = []
    while len(observations) < 40:
        rvec, tvec = _random_board_pose(rng)
        pids, pts = _in_view_ids(rvec, tvec, K, D, model, rng)
        if pids.size >= 20:
            observations.append((pids, pts.astype(np.float32)))

    r = calibrate_intrinsics(OBJP, observations, (W, H), model=model)
    assert r["model"] == model
    assert r["reproj_rms"] < 0.2
    fx, fy, cx, cy = r["intrinsics"]
    assert abs(fx - K[0, 0]) < 0.01 * K[0, 0]
    assert abs(fy - K[1, 1]) < 0.01 * K[1, 1]
    assert abs(cx - K[0, 2]) < 3.0 and abs(cy - K[1, 2]) < 3.0
    assert np.allclose(r["distortion"], D, atol=5e-3)


def test_estimate_target_pose_equi():
    rng = np.random.default_rng(3)
    rvec, tvec = _random_board_pose(rng, max_off_axis=0.9)
    pids, pts = _in_view_ids(rvec, tvec, K_EQUI, D_EQUI, camera_model.EQUI, rng)
    assert pids.size >= 20
    T = estimate_target_pose(OBJP, pids, pts, K_EQUI, D_EQUI, camera_model.EQUI)
    assert T is not None
    T_gt = se3.rt_to_T(rvec, tvec)
    assert np.allclose(T[:3, 3], T_gt[:3, 3], atol=2e-3)
    assert np.allclose(T[:3, :3], T_gt[:3, :3], atol=2e-3)


def test_extrinsics_and_ba_mixed_rig():
    """Mixed rig (radtan world camera + equi fisheye): init + BA recover the
    ground-truth relative pose."""
    rng = np.random.default_rng(4)
    # camera1 (equi) 0.4 m to the right of camera0, toed in ~15 deg
    r_gt = np.array([0.0, -0.26, 0.0])
    t_gt = np.array([-0.35, 0.02, 0.05])
    T_cam1_world = se3.rt_to_T(r_gt, t_gt)   # world (= camera0) -> camera1

    intr = {
        "camera0": {"model": camera_model.RADTAN,
                    "intrinsics": [K_RADTAN[0, 0], K_RADTAN[1, 1],
                                   K_RADTAN[0, 2], K_RADTAN[1, 2]],
                    "distortion": list(D_RADTAN)},
        "camera1": {"model": camera_model.EQUI,
                    "intrinsics": [K_EQUI[0, 0], K_EQUI[1, 1],
                                   K_EQUI[0, 2], K_EQUI[1, 2]],
                    "distortion": list(D_EQUI)},
    }
    names = ["camera0", "camera1"]
    Ks = {n: K_from_intrinsics(intr[n]["intrinsics"]) for n in names}
    Ds = {"camera0": D_RADTAN, "camera1": D_EQUI}
    Ms = {"camera0": camera_model.RADTAN, "camera1": camera_model.EQUI}
    T_cam = {"camera0": np.eye(4), "camera1": T_cam1_world}

    views = []
    while len(views) < 30:
        rvec, tvec = _random_board_pose(rng, max_off_axis=0.5)
        T_target_w = se3.rt_to_T(rvec, tvec)   # target -> world(cam0)
        view = {}
        for n in names:
            rv, tv = se3.T_to_rt(T_cam[n] @ T_target_w)
            pids, pts = _in_view_ids(rv, tv, Ks[n], Ds[n], Ms[n], rng)
            if pids.size >= 20:
                view[n] = (pids, pts.astype(np.float32))
        if len(view) == 2:
            views.append(view)

    cam_world, board_world, obs_struct, info = init_extrinsics(views, names, intr, OBJP)
    assert info["connected"]
    cam_world, board_world, ba = bundle_adjust(cam_world, board_world, obs_struct,
                                               names, intr, OBJP)
    assert ba["success"]
    assert ba["rms_after"] < 0.2
    assert np.allclose(cam_world[1][:3, 3], T_cam1_world[:3, 3], atol=2e-3)
    assert np.allclose(cam_world[1][:3, :3], T_cam1_world[:3, :3], atol=2e-3)


def test_radtan_files_without_model_key_still_work():
    """Backward compat: intrinsics entries written before the model key existed."""
    assert camera_model.model_of({"intrinsics": [1, 1, 0, 0]}) == camera_model.RADTAN
    with pytest.raises(ValueError):
        camera_model.check_model("pinhole-fov")
