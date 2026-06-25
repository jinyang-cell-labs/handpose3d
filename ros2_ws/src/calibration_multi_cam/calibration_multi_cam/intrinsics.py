"""Per-camera intrinsic calibration (pinhole + radial-tangential).

Wraps ``cv2.calibrateCamera`` with ``CALIB_FIX_K3`` so the distortion vector is
the 4-parameter ``[k1, k2, p1, p2]`` radtan model that matches kalibr's
``pinhole-radtan``.
"""
from __future__ import annotations

import cv2
import numpy as np


def calibrate_intrinsics(object_points_all, observations, image_size, min_views=6):
    """Calibrate one camera.

    Parameters
    ----------
    object_points_all : (N, 3) float
        Full set of target corner coordinates (target frame).
    observations : list of (point_ids (Mi,), image_points (Mi, 2))
        One entry per accepted view this camera contributed.
    image_size : (width, height)
    min_views : int
        Minimum number of usable views required.

    Returns
    -------
    dict with keys: intrinsics [fx, fy, cx, cy], distortion [k1, k2, p1, p2],
    resolution [w, h], reproj_rms (float), num_views (int).
    """
    obj_pts = []
    img_pts = []
    for pids, pts in observations:
        pids = np.asarray(pids).reshape(-1)
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if pids.size < 6:
            continue  # too few corners for a stable view
        obj_pts.append(object_points_all[pids].astype(np.float32).reshape(-1, 1, 3))
        img_pts.append(pts.reshape(-1, 1, 2))

    if len(obj_pts) < min_views:
        raise ValueError(
            f"Not enough usable views: {len(obj_pts)} < min_views={min_views}"
        )

    w, h = int(image_size[0]), int(image_size[1])
    flags = cv2.CALIB_FIX_K3  # -> 4-param radtan [k1, k2, p1, p2]
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, (w, h), None, None, flags=flags
    )
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)

    return {
        "model": "pinhole-radtan",
        "resolution": [w, h],
        "intrinsics": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        "distortion": [float(dist[0]), float(dist[1]), float(dist[2]), float(dist[3])],
        "reproj_rms": float(rms),
        "num_views": len(obj_pts),
    }


def K_from_intrinsics(intr):
    """[fx, fy, cx, cy] -> 3x3 camera matrix."""
    fx, fy, cx, cy = [float(v) for v in intr]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def dist_array(distortion):
    """[k1, k2, p1, p2] -> (4,) float array for OpenCV."""
    return np.asarray(distortion, dtype=np.float64).reshape(-1)[:4]
