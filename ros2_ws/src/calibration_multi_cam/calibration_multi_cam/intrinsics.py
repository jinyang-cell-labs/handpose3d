"""Per-camera intrinsic calibration (pinhole + radial-tangential).

Wraps ``cv2.calibrateCamera`` with ``CALIB_FIX_K3`` so the distortion vector is
the 4-parameter ``[k1, k2, p1, p2]`` radtan model that matches kalibr's
``pinhole-radtan``.
"""
from __future__ import annotations

import cv2
import numpy as np


def calibrate_intrinsics(object_points_all, observations, image_size, min_views=6,
                         reject_outliers=True, outlier_sigma=3.0,
                         outlier_floor_px=1.0, max_iters=10, min_corners_view=6):
    """Calibrate one camera, with iterative robust outlier rejection.

    A single ``cv2.calibrateCamera`` has no outlier handling, so a few
    blurry/oblique views inflate the global RMS. Mirroring kalibr's robust
    pipeline, we iterate: calibrate, drop *individual corners* whose
    reprojection error exceeds a robust threshold (median + sigma*MAD, with an
    absolute floor), and recalibrate until no corners are removed.

    Parameters
    ----------
    object_points_all : (N, 3) float
        Full set of target corner coordinates (target frame).
    observations : list of (point_ids (Mi,), image_points (Mi, 2))
        One entry per accepted view this camera contributed.
    image_size : (width, height)
    min_views : int
        Minimum number of usable views required.
    reject_outliers : bool
        Enable the iterative rejection loop.
    outlier_sigma : float
        Robust threshold = median + outlier_sigma * 1.4826 * MAD.
    outlier_floor_px : float
        Never reject corners below this reprojection error (avoids over-pruning
        an already-good calibration).
    min_corners_view : int
        Drop a whole view if rejection leaves it with fewer corners than this.

    Returns
    -------
    dict with keys: intrinsics [fx, fy, cx, cy], distortion [k1, k2, p1, p2],
    resolution [w, h], reproj_rms (float), num_views (int), per_view_rms (list),
    num_corners (int), num_rejected (int).
    """
    obj_pts = []
    img_pts = []
    for pids, pts in observations:
        pids = np.asarray(pids).reshape(-1)
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if pids.size < min_corners_view:
            continue  # too few corners for a stable view
        obj_pts.append(object_points_all[pids].astype(np.float32).reshape(-1, 1, 3))
        img_pts.append(pts.reshape(-1, 1, 2))

    if len(obj_pts) < min_views:
        raise ValueError(
            f"Not enough usable views: {len(obj_pts)} < min_views={min_views}"
        )

    w, h = int(image_size[0]), int(image_size[1])
    flags = cv2.CALIB_FIX_K3  # -> 4-param radtan [k1, k2, p1, p2]

    n_initial = sum(len(o) for o in obj_pts)
    for _ in range(max_iters):
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts, img_pts, (w, h), None, None, flags=flags
        )
        if not reject_outliers:
            break
        obj_pts, img_pts, removed = _reject_corner_outliers(
            obj_pts, img_pts, K, dist, rvecs, tvecs,
            outlier_sigma, outlier_floor_px, min_corners_view,
        )
        if len(obj_pts) < min_views:
            raise ValueError(
                f"Outlier rejection left only {len(obj_pts)} views (< {min_views})"
            )
        if removed == 0:
            break

    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    per_view = _per_view_rms(obj_pts, img_pts, K, dist, rvecs, tvecs)
    n_final = sum(len(o) for o in obj_pts)

    return {
        "model": "pinhole-radtan",
        "resolution": [w, h],
        "intrinsics": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        "distortion": [float(dist[0]), float(dist[1]), float(dist[2]), float(dist[3])],
        "reproj_rms": float(rms),
        "num_views": len(obj_pts),
        "num_corners": int(n_final),
        "num_rejected": int(n_initial - n_final),
        "per_view_rms": per_view,
    }


def _reject_corner_outliers(obj_pts, img_pts, K, dist, rvecs, tvecs,
                            sigma, floor_px, min_corners_view):
    """Drop corners with a robustly-high reprojection error; rebuild views.

    Returns (obj_pts, img_pts, num_removed). Views falling below
    ``min_corners_view`` after rejection are dropped entirely.
    """
    # per-corner reprojection error across all views
    all_err = []
    proj_per_view = []
    for op, ip, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        e = np.linalg.norm(ip.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
        proj_per_view.append(e)
        all_err.append(e)
    err = np.concatenate(all_err)
    median = float(np.median(err))
    mad = float(np.median(np.abs(err - median)))
    thresh = max(floor_px, median + sigma * 1.4826 * mad)

    new_obj, new_img, removed = [], [], 0
    for op, ip, e in zip(obj_pts, img_pts, proj_per_view):
        keep = e <= thresh
        removed += int((~keep).sum())
        if int(keep.sum()) < min_corners_view:
            continue  # drop the whole view
        new_obj.append(op[keep])
        new_img.append(ip[keep])
    return new_obj, new_img, removed


def _per_view_rms(obj_pts, img_pts, K, dist, rvecs, tvecs):
    """Reprojection RMS [px] for each view, sorted descending (worst first)."""
    errs = []
    for op, ip, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        proj = proj.reshape(-1, 2)
        d = ip.reshape(-1, 2) - proj
        errs.append(float(np.sqrt(np.mean(np.sum(d * d, axis=1)))))
    return sorted(errs, reverse=True)


def K_from_intrinsics(intr):
    """[fx, fy, cx, cy] -> 3x3 camera matrix."""
    fx, fy, cx, cy = [float(v) for v in intr]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def dist_array(distortion):
    """[k1, k2, p1, p2] -> (4,) float array for OpenCV."""
    return np.asarray(distortion, dtype=np.float64).reshape(-1)[:4]
