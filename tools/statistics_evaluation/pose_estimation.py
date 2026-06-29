#!/usr/bin/env python3
"""
Model-based 6-DoF hand-pose estimation (monocular PnP).

Given ONE camera's intrinsics + extrinsics, its detected 2D hand landmarks, and
MediaPipe's hand-local metric 3D model (``hand_world_landmarks``), estimate the
rigid 6-DoF pose of the hand-local frame expressed in the WORLD frame, by
minimising the reprojection error.

Formulation
-----------
Variable: ``T_world_hand`` (hand-local -> world), 6 DoF = an so(3) rotation
vector ``r`` (3) + a translation ``t`` (3). For each joint i:

    X_world = R(r) @ X_hand_i + t                 # hand-local -> world
    X_cam   = R_cw @ X_world + t_cw                # world -> camera (extrinsics)
    uv_i    = K @ (X_cam / X_cam.z)                # pinhole (undistorted)

Cost (minimised with Levenberg-Marquardt via scipy.optimize.least_squares):

    sum_i || uv_i - uv_detected_i ||^2

OpenCV ``solvePnP`` seeds the optimisation (good initial guess avoids the
near-planar flip ambiguity / local minima); the least-squares step then refines
exactly the reprojection cost above.

The pose comes out in the world frame, so independent per-camera estimates are
directly comparable — agreement across cameras is a calibration sanity check.
``estimate_hand_pose_multicam`` fuses several views into one shared pose.

This module is calibration/topic agnostic: feed it plain numpy arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

N_LANDMARKS = 21


# --------------------------------------------------------------------- helpers
def world_to_cam(T_world_cam):
    """Extrinsics ``T_world_cam`` (4x4, cam->world) -> (R_cw, t_cw) world->cam."""
    T = np.asarray(T_world_cam, float).reshape(4, 4)
    R_wc, c = T[:3, :3], T[:3, 3]
    R_cw = R_wc.T
    return R_cw, -R_cw @ c


def _project(K, R_cw, t_cw, X_world):
    """Pinhole-project (N,3) world points -> (N,2) pixels (no distortion)."""
    Xc = X_world @ R_cw.T + t_cw            # (N,3)
    z = Xc[:, 2:3]
    uv = (Xc[:, :2] / z) @ np.array([[K[0, 0], 0.0], [0.0, K[1, 1]]])
    uv += np.array([K[0, 2], K[1, 2]])
    return uv, Xc[:, 2]


def _pose_to_matrix(rotvec, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


# ------------------------------------------------------------------- residuals
def _residuals(params, views):
    """Stacked reprojection residuals over all views for a shared T_world_hand.

    ``views`` is a list of dicts: {K, R_cw, t_cw, X_hand (n,3), uv (n,2)} already
    filtered to finite joints. ``params`` = [rx,ry,rz, tx,ty,tz].
    """
    R_wh = Rotation.from_rotvec(params[:3]).as_matrix()
    t_wh = params[3:6]
    out = []
    for v in views:
        X_world = v["X_hand"] @ R_wh.T + t_wh
        uv, _ = _project(v["K"], v["R_cw"], v["t_cw"], X_world)
        out.append((uv - v["uv"]).ravel())
    return np.concatenate(out) if out else np.zeros(0)


# ---------------------------------------------------------------------- result
@dataclass
class PoseResult:
    success: bool
    message: str
    T_world_hand: np.ndarray          # (4,4) hand-local -> world
    rotvec: np.ndarray                # (3,) so(3) rotation, world frame
    translation: np.ndarray           # (3,) hand-local origin in world (m)
    reproj_rms_px: float              # RMS per-joint reprojection error
    reproj_mean_px: float
    n_points: int                     # joints used (finite in all views)
    per_joint_error_px: np.ndarray    # (N_LANDMARKS,) NaN where unused

    def euler_deg(self, seq="xyz"):
        """Convenience: world-frame orientation as Euler angles (degrees)."""
        return Rotation.from_rotvec(self.rotvec).as_euler(seq, degrees=True)


def _finalize(params, views_full, label):
    """Build a PoseResult from optimised params + the full (unfiltered) views."""
    R_wh = Rotation.from_rotvec(params[:3]).as_matrix()
    t_wh = params[3:6]
    per_joint = np.full(N_LANDMARKS, np.nan)
    all_d = []
    for v in views_full:
        X_world = v["X_hand_full"] @ R_wh.T + t_wh
        uv, _ = _project(v["K"], v["R_cw"], v["t_cw"], X_world)
        d = np.linalg.norm(uv - v["uv_full"], axis=1)
        for j, idx in enumerate(v["idx"]):
            # keep the (single-view) per-joint error; for multicam this is the
            # last view's value, which is fine for the typical single-cam call.
            per_joint[idx] = d[idx] if idx < len(d) else np.nan
        all_d.append(d[np.isfinite(d)])
    dist = np.concatenate(all_d) if all_d else np.zeros(0)
    rms = float(np.sqrt(np.mean(dist ** 2))) if dist.size else float("nan")
    mean = float(np.mean(dist)) if dist.size else float("nan")
    return PoseResult(
        success=True,
        message=label,
        T_world_hand=_pose_to_matrix(params[:3], t_wh),
        rotvec=params[:3].copy(),
        translation=t_wh.copy(),
        reproj_rms_px=rms,
        reproj_mean_px=mean,
        n_points=int(dist.size),
        per_joint_error_px=per_joint,
    )


# --------------------------------------------------------------- main entry pts
def _prepare_view(K, T_world_cam, landmarks_image, landmarks_world):
    """Validate + slice one view; return (view_filtered, view_full) or (None,..)."""
    K = np.asarray(K, float).reshape(3, 3)
    uv = np.asarray(landmarks_image, float)[:, :2]
    Xh = np.asarray(landmarks_world, float)[:, :3]
    R_cw, t_cw = world_to_cam(T_world_cam)
    finite = np.all(np.isfinite(uv), axis=1) & np.all(np.isfinite(Xh), axis=1)
    idx = np.nonzero(finite)[0]
    filt = {"K": K, "R_cw": R_cw, "t_cw": t_cw,
            "X_hand": Xh[finite], "uv": uv[finite]}
    full = {"K": K, "R_cw": R_cw, "t_cw": t_cw,
            "X_hand_full": Xh, "uv_full": uv, "idx": idx}
    return filt, full, idx


def _pnp_init(K, R_cw, t_cw, X_hand, uv, distortion):
    """Seed T_world_hand via OpenCV solvePnP (hand->cam), composed to world."""
    import cv2
    dist = np.zeros(5) if distortion is None else np.asarray(distortion, float)
    ok, rvec, tvec = cv2.solvePnP(
        X_hand.astype(np.float64), uv.astype(np.float64), K, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    R_ch, _ = cv2.Rodrigues(rvec)                 # hand -> cam (X_cam=R_ch X+tvec)
    # Compose T_world_hand = T_world_cam @ T_cam_hand. With world->cam (R_cw,t_cw),
    # cam->world is R_wc=R_cw.T, so:
    #   X_world = R_wc (R_ch X_hand + tvec) - R_wc t_cw
    R_wc = R_cw.T
    R_wh = R_wc @ R_ch
    t_wh = R_wc @ (tvec.ravel() - t_cw)
    return np.concatenate([Rotation.from_matrix(R_wh).as_rotvec(), t_wh])


def estimate_hand_pose(K, T_world_cam, landmarks_image, landmarks_world,
                       *, distortion=None, init_pose=None, use_pnp_init=True,
                       loss="linear", f_scale=4.0, max_nfev=200):
    """Estimate ``T_world_hand`` for ONE camera. Returns a ``PoseResult``.

    Args:
        K: (3,3) intrinsics.
        T_world_cam: (4,4) camera->world extrinsics.
        landmarks_image: (21,2|3) detected pixels (z ignored).
        landmarks_world: (21,3) MediaPipe hand-local metric model (metres).
        distortion: optional (k1,k2,p1,p2[,k3]) for the solvePnP seed only; the
            refinement is pinhole (landmarks are assumed undistorted).
        init_pose: optional (4,4) T_world_hand initial guess (overrides PnP).
        use_pnp_init: seed with OpenCV solvePnP (recommended).
        loss: scipy least_squares loss ("linear", "soft_l1", "huber", ...).
        f_scale: robust-loss soft threshold (px) when loss != "linear".
        max_nfev: max optimiser iterations.
    """
    filt, full, idx = _prepare_view(
        K, T_world_cam, landmarks_image, landmarks_world)
    if len(idx) < 4:
        return PoseResult(False, f"need >=4 finite joints, got {len(idx)}",
                          np.eye(4), np.zeros(3), np.zeros(3),
                          float("nan"), float("nan"), len(idx),
                          np.full(N_LANDMARKS, np.nan))

    # ---- initial guess --------------------------------------------------
    p0 = None
    if init_pose is not None:
        T0 = np.asarray(init_pose, float).reshape(4, 4)
        p0 = np.concatenate([Rotation.from_matrix(T0[:3, :3]).as_rotvec(),
                             T0[:3, 3]])
    elif use_pnp_init:
        try:
            p0 = _pnp_init(filt["K"], filt["R_cw"], filt["t_cw"],
                           filt["X_hand"], filt["uv"], distortion)
        except Exception:  # noqa: BLE001  (cv2 missing / degenerate)
            p0 = None
    if p0 is None:
        # Crude fallback: identity rotation, translation = camera centre + 0.5 m
        # along its optical axis (keeps the hand in front of the camera).
        R_wc = filt["R_cw"].T
        cam_c = -R_wc @ filt["t_cw"]
        p0 = np.concatenate([np.zeros(3), cam_c + R_wc @ np.array([0, 0, 0.5])])

    # ---- non-linear refinement -----------------------------------------
    method = "lm" if loss == "linear" else "trf"
    sol = least_squares(
        _residuals, p0, args=([filt],), method=method,
        loss=loss, f_scale=f_scale, max_nfev=max_nfev,
    )
    res = _finalize(sol.x, [full], "ok (single camera)")
    res.success = bool(sol.success or sol.status > 0)
    res.message = f"single-cam: {sol.message}"
    return res


def estimate_hand_pose_multicam(observations, *, init_pose=None,
                                loss="linear", f_scale=4.0, max_nfev=300):
    """Fuse several cameras into ONE shared ``T_world_hand``.

    ``observations``: list of dicts, each with keys
    ``K, T_world_cam, landmarks_image, landmarks_world``. Each view contributes
    reprojection residuals for the shared pose, using its own hand-local model.
    """
    filts, fulls, all_idx, kept = [], [], [], []
    for o in observations:
        filt, full, idx = _prepare_view(
            o["K"], o["T_world_cam"], o["landmarks_image"], o["landmarks_world"])
        if len(idx) >= 1:
            filts.append(filt)
            fulls.append(full)
            all_idx.append(idx)
            kept.append(o)
    total = sum(len(i) for i in all_idx)
    if not filts or total < 4:
        return PoseResult(False, f"need >=4 finite joints total, got {total}",
                          np.eye(4), np.zeros(3), np.zeros(3),
                          float("nan"), float("nan"), total,
                          np.full(N_LANDMARKS, np.nan))

    # Seed from the single best view (most points).
    if init_pose is not None:
        T0 = np.asarray(init_pose, float).reshape(4, 4)
        p0 = np.concatenate([Rotation.from_matrix(T0[:3, :3]).as_rotvec(),
                             T0[:3, 3]])
    else:
        # Seed from the single view with the most finite joints.
        best = max(range(len(filts)), key=lambda k: len(all_idx[k]))
        ob = kept[best]
        seed = estimate_hand_pose(ob["K"], ob["T_world_cam"],
                                  ob["landmarks_image"], ob["landmarks_world"])
        p0 = np.concatenate([seed.rotvec, seed.translation])

    method = "lm" if loss == "linear" else "trf"
    sol = least_squares(_residuals, p0, args=(filts,), method=method,
                        loss=loss, f_scale=f_scale, max_nfev=max_nfev)
    res = _finalize(sol.x, fulls, "multicam")
    res.success = bool(sol.success or sol.status > 0)
    res.message = f"multicam ({len(filts)} views): {sol.message}"
    return res
