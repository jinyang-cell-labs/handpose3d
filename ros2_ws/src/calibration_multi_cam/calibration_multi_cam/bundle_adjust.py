"""Global bundle adjustment over the rig extrinsics + per-view board poses.

Minimizes the reprojection error of every observed corner across all cameras
and views with ``scipy.optimize.least_squares`` (Trust Region Reflective) and a
robust loss. Camera intrinsics are held fixed (already calibrated); cam0 is
pinned to identity so the world frame stays aligned with the first camera.

Parameter vector: [cam_1 .. cam_{C-1}] (6 each, world->cam) followed by
[board_0 .. board_{V-1}] (6 each, target->world). A residual block for a given
(camera, view) depends only on that camera's 6 params and that view's 6 params,
which gives the sparse Jacobian pattern passed to least_squares.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from calibration_multi_cam import se3
from calibration_multi_cam.intrinsics import K_from_intrinsics, dist_array


def bundle_adjust(cam_world, board_world, obs_struct, camera_names,
                  intrinsics_by_name, object_points_all,
                  robust_loss="huber", loss_scale=1.0, verbose=False):
    C = len(cam_world)
    V = len(board_world)
    Ks = [K_from_intrinsics(intrinsics_by_name[n]["intrinsics"]) for n in camera_names]
    Ds = [dist_array(intrinsics_by_name[n]["distortion"]) for n in camera_names]
    n_cam_params = (C - 1) * 6
    n_params = n_cam_params + V * 6

    # Flatten observations into residual blocks (cam_idx, view_idx, objp, pixels).
    blocks = []
    for v, entries in enumerate(obs_struct):
        for (c, pids, pixels) in entries:
            objp = object_points_all[np.asarray(pids).reshape(-1)].astype(np.float64)
            pix = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
            blocks.append((c, v, objp, pix))
    total_res = int(sum(2 * len(b[3]) for b in blocks))

    def pack():
        x = np.zeros(n_params)
        for c in range(1, C):
            r, t = se3.T_to_rt(cam_world[c])
            x[(c - 1) * 6:(c - 1) * 6 + 3] = r
            x[(c - 1) * 6 + 3:(c - 1) * 6 + 6] = t
        for v in range(V):
            r, t = se3.T_to_rt(board_world[v])
            base = n_cam_params + v * 6
            x[base:base + 3] = r
            x[base + 3:base + 6] = t
        return x

    def unpack(x):
        cams = [np.eye(4)]
        for c in range(1, C):
            cams.append(se3.rt_to_T(x[(c - 1) * 6:(c - 1) * 6 + 3],
                                    x[(c - 1) * 6 + 3:(c - 1) * 6 + 6]))
        boards = []
        for v in range(V):
            base = n_cam_params + v * 6
            boards.append(se3.rt_to_T(x[base:base + 3], x[base + 3:base + 6]))
        return cams, boards

    def residuals(x):
        cams, boards = unpack(x)
        out = np.empty(total_res)
        k = 0
        for (c, v, objp, pix) in blocks:
            T_cam_target = cams[c] @ boards[v]   # (world->cam) @ (target->world)
            rvec, tvec = se3.T_to_rt(T_cam_target)
            proj, _ = cv2.projectPoints(objp, rvec.reshape(3, 1), tvec.reshape(3, 1),
                                        Ks[c], Ds[c])
            r = (proj.reshape(-1, 2) - pix).reshape(-1)
            out[k:k + r.size] = r
            k += r.size
        return out

    def jac_sparsity():
        S = lil_matrix((total_res, n_params), dtype=np.uint8)
        k = 0
        for (c, v, objp, pix) in blocks:
            m = 2 * len(pix)
            if c >= 1:
                S[k:k + m, (c - 1) * 6:(c - 1) * 6 + 6] = 1
            base = n_cam_params + v * 6
            S[k:k + m, base:base + 6] = 1
            k += m
        return S

    x0 = pack()
    rms0 = float(np.sqrt(np.mean(residuals(x0) ** 2)))

    result = least_squares(
        residuals, x0, jac_sparsity=jac_sparsity(), method="trf",
        loss=robust_loss, f_scale=float(loss_scale),
        x_scale="jac", verbose=2 if verbose else 0,
    )

    cams, boards = unpack(result.x)
    rms = float(np.sqrt(np.mean(residuals(result.x) ** 2)))
    info = {
        "rms_before": rms0,
        "rms_after": rms,
        "num_residuals": total_res,
        "num_params": n_params,
        "success": bool(result.success),
        "num_views": V,
    }
    return cams, boards, info


def per_camera_rms(cam_world, board_world, obs_struct, camera_names,
                   intrinsics_by_name, object_points_all):
    """Reprojection RMS [px] per camera, for the final report."""
    Ks = [K_from_intrinsics(intrinsics_by_name[n]["intrinsics"]) for n in camera_names]
    Ds = [dist_array(intrinsics_by_name[n]["distortion"]) for n in camera_names]
    acc = {n: [] for n in camera_names}
    for v, entries in enumerate(obs_struct):
        for (c, pids, pixels) in entries:
            objp = object_points_all[np.asarray(pids).reshape(-1)].astype(np.float64)
            pix = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
            T_cam_target = cam_world[c] @ board_world[v]
            rvec, tvec = se3.T_to_rt(T_cam_target)
            proj, _ = cv2.projectPoints(objp, rvec.reshape(3, 1), tvec.reshape(3, 1),
                                        Ks[c], Ds[c])
            acc[camera_names[c]].append((proj.reshape(-1, 2) - pix).reshape(-1))
    out = {}
    for n, errs in acc.items():
        if errs:
            e = np.concatenate(errs)
            out[n] = float(np.sqrt(np.mean(e ** 2)))
    return out
