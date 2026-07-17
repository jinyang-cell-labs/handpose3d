"""Camera projection models: pinhole-radtan and pinhole-equi (fisheye).

Central switch for every model-dependent OpenCV call, so the rest of the
pipeline (intrinsics, PnP init, bundle adjustment, publishing) stays
model-agnostic.

- ``pinhole-radtan``: 4-parameter radial-tangential ``[k1, k2, p1, p2]``
  (kalibr's ``pinhole-radtan``). Good up to ~100 deg FOV.
- ``pinhole-equi``: 4-parameter equidistant fisheye ``[k1, k2, k3, k4]``
  (kalibr's ``pinhole-equi``, OpenCV ``cv2.fisheye``). Use for wide-FOV
  lenses (~100-180 deg) where the radtan polynomial diverges at the edges.

Both models keep the same file schema: 4 distortion floats + [fx, fy, cx, cy];
only the ``model`` string tells consumers how to interpret them.

Jacobian layouts differ between the two OpenCV APIs (verified numerically):
``cv2.projectPoints`` puts d/d(rvec,tvec) in columns 0:6 of its jacobian,
``cv2.fisheye.projectPoints`` in columns 8:14 (f, c, k come first).
"""
from __future__ import annotations

import cv2
import numpy as np

RADTAN = "pinhole-radtan"
EQUI = "pinhole-equi"
MODELS = (RADTAN, EQUI)


def check_model(model):
    if model not in MODELS:
        raise ValueError(f"Unknown camera model '{model}'; expected one of {MODELS}")
    return model


def model_of(cam_entry):
    """Model string of an intrinsics-file camera entry (radtan when absent)."""
    return check_model(cam_entry.get("model", RADTAN))


def project_points(objp, rvec, tvec, K, D, model):
    """Project 3D points (target/object frame) to pixels. Returns (N, 2)."""
    objp = np.asarray(objp, dtype=np.float64).reshape(-1, 1, 3)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    if model == EQUI:
        proj, _ = cv2.fisheye.projectPoints(objp, rvec, tvec, K, D)
    else:
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
    return proj.reshape(-1, 2)


def project_points_jac(objp, rvec, tvec, K, D, model):
    """Project + pose jacobian. Returns (proj (N, 2), J (2N, 6)).

    J columns are d(pixel)/d(rvec, tvec) regardless of model, hiding the
    differing column layouts of the two OpenCV APIs.
    """
    objp = np.asarray(objp, dtype=np.float64).reshape(-1, 1, 3)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    if model == EQUI:
        proj, J = cv2.fisheye.projectPoints(objp, rvec, tvec, K, D)
        Jrt = J[:, 8:14]
    else:
        proj, J = cv2.projectPoints(objp, rvec, tvec, K, D)
        Jrt = J[:, 0:6]
    return proj.reshape(-1, 2), np.asarray(Jrt)


def undistort_to_normalized(pixels, K, D, model):
    """Pixels -> ideal normalized image coordinates (z=1 plane). Returns (N, 2).

    Feeding these to solvePnP with K=identity and no distortion makes PnP
    model-agnostic (cv2.solvePnP itself only understands radtan).
    """
    pix = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    if model == EQUI:
        und = cv2.fisheye.undistortPoints(pix, K, D)
    else:
        und = cv2.undistortPoints(pix, K, D)
    return und.reshape(-1, 2)


def calibrate_camera(obj_pts, img_pts, image_size, model):
    """Model-dispatched intrinsic calibration.

    obj_pts/img_pts: lists of per-view (Ni, 1, 3) / (Ni, 1, 2) arrays.
    Returns (rms, K, dist (4,), rvecs, tvecs) — distortion is
    [k1, k2, p1, p2] for radtan, [k1, k2, k3, k4] for equi.
    """
    if model == EQUI:
        obj64 = [np.asarray(o, dtype=np.float64).reshape(-1, 1, 3) for o in obj_pts]
        img64 = [np.asarray(i, dtype=np.float64).reshape(-1, 1, 2) for i in img_pts]
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
        rms, K, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            obj64, img64, tuple(image_size), None, None,
            flags=flags, criteria=criteria,
        )
    else:
        flags = cv2.CALIB_FIX_K3  # -> 4-param radtan [k1, k2, p1, p2]
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts, img_pts, tuple(image_size), None, None, flags=flags
        )
    return rms, K, np.asarray(dist, dtype=np.float64).reshape(-1)[:4], rvecs, tvecs


def ros_distortion_model(model):
    """sensor_msgs/CameraInfo distortion_model string for this model."""
    return "equidistant" if model == EQUI else "plumb_bob"
