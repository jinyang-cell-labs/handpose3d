"""Extrinsic initialization: per-view PnP -> pairwise relative poses ->
max-weight spanning-tree chaining to camera0.

Produces an initial pose for every camera in the camera0 (world) frame, plus an
initial board pose per view, ready for the bundle adjustment. Intrinsics are
taken as given (already calibrated).

Pose convention (see se3.py): ``T_cam_world`` maps a world point into the
camera frame. camera0 is the world, so its pose is identity.
"""
from __future__ import annotations

import cv2
import numpy as np

from calibration_multi_cam import camera_model, se3
from calibration_multi_cam.intrinsics import K_from_intrinsics, dist_array


def estimate_target_pose(object_points_all, pids, pixels, K, dist,
                         model=camera_model.RADTAN):
    """solvePnP for a planar target -> T_cam_target, or None on failure.

    cv2.solvePnP only understands the radtan model, so for other models the
    pixels are first undistorted to ideal normalized coordinates and PnP runs
    with an identity camera matrix.
    """
    pids = np.asarray(pids).reshape(-1)
    if pids.size < 4:
        return None
    objp = object_points_all[pids].astype(np.float64)
    imgp = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if model != camera_model.RADTAN:
        try:
            imgp = camera_model.undistort_to_normalized(imgp, K, dist, model)
        except cv2.error:
            return None
        K = np.eye(3)
        dist = None
    rvec = tvec = None
    ok = False
    try:
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        ok = False
    if not ok:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                objp, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
        except cv2.error:
            ok = False
    if not ok:
        return None
    return se3.rt_to_T(rvec, tvec)


def _max_spanning_tree(num_cams, weights):
    """Prim MST maximizing edge weight, rooted at camera 0.

    weights: dict {(i, j): w} with i < j. Returns (tree_edges, connected) where
    tree_edges is a parent-before-child ordered list of (parent, child)."""
    in_tree = {0}
    tree_edges = []
    while len(in_tree) < num_cams:
        best = None  # (weight, parent_in_tree, child_out)
        for (i, j), w in weights.items():
            for u, v in ((i, j), (j, i)):
                if u in in_tree and v not in in_tree:
                    if best is None or w > best[0]:
                        best = (w, u, v)
        if best is None:
            break  # remaining cameras are disconnected
        _, u, v = best
        in_tree.add(v)
        tree_edges.append((u, v))
    return tree_edges, len(in_tree) == num_cams


def init_extrinsics(views, camera_names, intrinsics_by_name, object_points_all,
                    min_corners=6):
    """Initialize camera extrinsics + per-view board poses.

    Parameters
    ----------
    views : list of dict {camera_name: (point_ids, image_points)}
    camera_names : list[str]   (camera_names[0] is the world camera)
    intrinsics_by_name : dict camera_name -> {"model", "intrinsics", "distortion"}
    object_points_all : (N, 3) target corners

    Returns
    -------
    cam_world : list[4x4]            T_cam_world per camera (camera0 = identity)
    board_world : list[4x4]          T_world_target per kept view
    obs_struct : list[list[(cam_idx, pids, pixels)]]   aligned with board_world
    info : dict                      {connected, pair_views, tree_edges, ...}
    """
    idx = {name: i for i, name in enumerate(camera_names)}
    C = len(camera_names)
    Ks = [K_from_intrinsics(intrinsics_by_name[n]["intrinsics"]) for n in camera_names]
    Ds = [dist_array(intrinsics_by_name[n]["distortion"]) for n in camera_names]
    Ms = [camera_model.model_of(intrinsics_by_name[n]) for n in camera_names]

    rel_samples = {}      # (i, j) i<j -> list of T_j_i  (maps i -> j)
    pair_views = {}       # (i, j) i<j -> count
    board_world = []
    obs_struct = []

    for view in views:
        # PnP per camera present in this view
        cam_T = {}        # cam_idx -> T_cam_target
        cam_obs = {}      # cam_idx -> (pids, pixels)
        for name, (pids, pixels) in view.items():
            c = idx[name]
            pids = np.asarray(pids).reshape(-1)
            pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
            if pids.size < min_corners:
                continue
            T = estimate_target_pose(object_points_all, pids, pixels,
                                     Ks[c], Ds[c], Ms[c])
            if T is None:
                continue
            cam_T[c] = T
            cam_obs[c] = (pids, pixels)

        if len(cam_T) < 2:
            continue  # a view needs >= 2 cameras to constrain extrinsics

        # pairwise relative poses T_b_a = T_b_t @ inv(T_a_t)
        present = sorted(cam_T.keys())
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                ia, ib = present[a], present[b]   # ia < ib
                T_b_a = cam_T[ib] @ se3.invert_T(cam_T[ia])
                rel_samples.setdefault((ia, ib), []).append(T_b_a)
                pair_views[(ia, ib)] = pair_views.get((ia, ib), 0) + 1

        # board pose (world frame) from the camera that saw the most corners
        ref = max(cam_T.keys(), key=lambda c: cam_obs[c][0].size)
        # filled in after camera poses are known; store ref + its T_cam_target
        obs_struct.append([(c, cam_obs[c][0], cam_obs[c][1]) for c in present])
        board_world.append(("__pending__", ref, cam_T[ref]))

    if not obs_struct:
        raise RuntimeError("No multi-camera views with valid PnP; cannot calibrate extrinsics.")

    # average pairwise relatives + spanning-tree chaining to camera0
    rel_avg = {k: se3.average_transforms(v) for k, v in rel_samples.items()}
    tree_edges, connected = _max_spanning_tree(C, pair_views)
    if not connected:
        linked = {0} | {v for _, v in tree_edges}
        missing = [camera_names[i] for i in range(C) if i not in linked]
        raise RuntimeError(
            f"Cameras not connected by shared views: {missing} cannot be chained "
            f"to {camera_names[0]}. Collect views where each links to a neighbour."
        )

    def rel_dir(p, c):
        return rel_avg[(p, c)] if p < c else se3.invert_T(rel_avg[(c, p)])

    cam_world = [None] * C
    cam_world[0] = np.eye(4)
    for p, c in tree_edges:
        cam_world[c] = rel_dir(p, c) @ cam_world[p]

    # resolve board poses now that camera poses exist:
    #   T_world_target = T_world_cam[ref] @ T_cam_target[ref]
    resolved_board = []
    for _tag, ref, T_cam_target in board_world:
        T_world_cam_ref = se3.invert_T(cam_world[ref])
        resolved_board.append(T_world_cam_ref @ T_cam_target)

    info = {
        "connected": connected,
        "pair_views": {f"{camera_names[i]}-{camera_names[j]}": n
                       for (i, j), n in sorted(pair_views.items())},
        "tree_edges": [(camera_names[p], camera_names[c]) for p, c in tree_edges],
        "num_views_used": len(obs_struct),
    }
    return cam_world, resolved_board, obs_struct, info
