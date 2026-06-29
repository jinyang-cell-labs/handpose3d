#!/usr/bin/env python3
"""
Replay a handpose JSONL log: animate the 2D landmarks of all three cameras and
the 3D hand triangulated from a chosen camera pair, with the three camera
positions drawn in the 3D view.

Run it from the evaluation venv (see start_evaluation.bash):

    python replay_logs.py

Everything is configured in the USER CONFIG block just below — edit the log
path, the camera pair to triangulate, and the replay speed there.
"""

import bisect
import os
import sys

import numpy as np

# ============================ USER CONFIG ==================================
# Absolute path to the log to replay.
LOG_PATH = (
    "/home/jinyang/repo/handpose3d/tools/statistics_evaluation/logs/"
    "handpose_log_20260626_155546.jsonl"
)
# The two cameras (by name, as in the log meta) to triangulate the 3D hand from.
TRIANGULATE_CAMERAS = ("camera0", "camera2")
# Replay speed: 1.0 = real time, 2.0 = twice as fast, 0.5 = half speed.
REPLAY_SPEED = 1.0
# Which hand to show: "Left", "Right", or None for whatever is present.
HANDEDNESS = None
# Max time difference (s) for a camera's detection to count as "the same moment".
SYNC_TOLERANCE_S = 0.05
# Reproject the triangulated 3D hand back onto ALL three cameras' 2D views (in a
# distinct colour) to eyeball reprojection error. For the two triangulated
# cameras this should hug the detection; the third is a cross-check.
ENABLE_REPROJECT = True
# Monocular model-based 6-DoF pose estimation (PnP on MediaPipe's hand_world
# model) from a SINGLE camera. Drawn ALONGSIDE the triangulated 3D hand in its
# own colour (does not replace it).
ENABLE_POSE_ESTIMATION = True
# The single camera the pose is estimated from (needs its 2D + world landmarks).
POSE_ESTIMATION_SOURCE = "camera0"
# Also reproject the estimated-pose hand onto all three cameras' 2D views (in a
# third colour, distinct from detection and the triangulation reprojection).
ENABLE_POSE_ESTIMATION_REPROJECT = True
# Loop the replay until the window is closed.
LOOP = True
# ===========================================================================

# Make the log loader importable whether or not PYTHONPATH was set up.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(
    0,
    os.path.join(_REPO_ROOT, "ros2_ws", "src",
                 "mediapie_landmarks_extraction", "scripts"),
)
from load_handpose_log import load_log  # noqa: E402

import pose_estimation as pe  # noqa: E402 (sibling module)

# Select an interactive matplotlib backend BEFORE importing pyplot, so a window
# actually opens. The venv's matplotlib otherwise falls back to the headless
# 'agg' backend (no tkinter/Qt in a pyenv venv) and nothing is shown.
import importlib.util  # noqa: E402

import matplotlib  # noqa: E402

for _mod, _backend in (("PyQt5", "QtAgg"), ("PySide6", "QtAgg"),
                       ("tkinter", "TkAgg")):
    if importlib.util.find_spec(_mod) is not None:
        matplotlib.use(_backend)
        break
else:
    print("WARNING: no interactive matplotlib backend found (no PyQt5/tkinter); "
          "the window will not open. Run start_evaluation.bash to install PyQt5.")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401 (registers 3d proj)

# 21-landmark MediaPipe hand skeleton.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]
N_LANDMARKS = 21
HAND_COLORS = {"Left": "#3399ff", "Right": "#ff8033"}
REPROJECT_COLOR = "#ff33cc"   # magenta — triangulation reprojected into 2D
POSE_COLOR = "#2ca02c"        # green — estimated 6-DoF pose (3D + its reprojection)
CAM_COLORS = ["#e74c3c", "#2ecc71", "#9b59b6", "#f1c40f", "#1abc9c"]


# ----------------------------------------------------------------- geometry
def projection_matrix(K, T_world_cam):
    """P = K [R_cw | t_cw] from T_world_cam (camera->world)."""
    K = np.asarray(K, float).reshape(3, 3)
    T = np.asarray(T_world_cam, float).reshape(4, 4)
    R_wc, c = T[:3, :3], T[:3, 3]
    R_cw = R_wc.T
    t_cw = -R_cw @ c
    return K @ np.hstack([R_cw, t_cw.reshape(3, 1)])


def dlt(P0, P1, p0, p1):
    """Triangulate one point from two views (DLT). NaN if degenerate."""
    A = np.array([
        p0[1] * P0[2] - P0[1],
        P0[0] - p0[0] * P0[2],
        p1[1] * P1[2] - P1[1],
        P1[0] - p1[0] * P1[2],
    ])
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    if abs(X[3]) < 1e-12:
        return np.full(3, np.nan)
    return X[:3] / X[3]


def triangulate_hand(pts0, pts1, P0, P1):
    """Triangulate 21 joint correspondences -> (21, 3) world points (NaN bad)."""
    out = np.full((N_LANDMARKS, 3), np.nan)
    for i in range(N_LANDMARKS):
        out[i] = dlt(P0, P1, pts0[i], pts1[i])
    return out


def project_points(P, pts3d):
    """Pinhole-project (21, 3) world points to (21, 2) pixels (NaN preserved).

    Landmarks are in the undistorted pinhole image (landmarks_undistorted), so
    P = K[R|t] alone reprojects them — no distortion term needed.
    """
    out = np.full((N_LANDMARKS, 2), np.nan)
    for i in range(N_LANDMARKS):
        X = pts3d[i]
        if not np.all(np.isfinite(X)):
            continue
        xh = P @ np.array([X[0], X[1], X[2], 1.0])
        if abs(xh[2]) < 1e-9:
            continue
        out[i] = xh[:2] / xh[2]
    return out


def equal_cube_limits(pts, pad=1.15, min_radius=0.1):
    """Fixed equal-aspect cube limits enclosing the given (N,3) points.

    Returns ``[(xlo,xhi),(ylo,yhi),(zlo,zhi)]`` — a single cube (same half-width
    on every axis) so the 3D view keeps a constant, undistorted scale.
    """
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return [(-1, 1), (-1, 1), (-1, 1)]
    mins, maxs = pts.min(0), pts.max(0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0 * pad, min_radius)
    return [(center[i] - radius, center[i] + radius) for i in range(3)]


def apply_limits_3d(ax, limits):
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1))


# ------------------------------------------------------------------- loading
def index_frames(log, camera, handedness):
    """Per-camera sorted (stamps, hands).

    hands = {label: {"image": (21,2) px, "world": (21,3) m | None}} — the
    hand-local world model is kept so the pose-estimation source camera can run
    PnP; it's None when the log didn't record world landmarks.
    """
    stamps, hands = [], []
    for rec in log.frames:
        if rec["camera"] != camera:
            continue
        by_label = {}
        for h in rec["hands"]:
            if handedness is not None and h["handedness"] != handedness:
                continue
            img = np.asarray(h["landmarks_image"], float)[:, :2]  # x,y px
            world = h.get("landmarks_world")
            world = np.asarray(world, float) if world is not None else None
            by_label[h["handedness"]] = {"image": img, "world": world}
        stamps.append(rec["stamp_ns"])
        hands.append(by_label)
    order = np.argsort(stamps)
    return np.asarray(stamps)[order], [hands[i] for i in order]


def pose_world_joints(T_world_hand, world_model):
    """Place the hand-local model into the world via the estimated pose."""
    R = T_world_hand[:3, :3]
    t = T_world_hand[:3, 3]
    return world_model @ R.T + t


def nearest(stamps, hands, t, tol_ns):
    """Hands dict for the record nearest to t within tol_ns, else {}."""
    if len(stamps) == 0:
        return {}
    i = bisect.bisect_left(stamps, t)
    best, best_dt = None, None
    for j in (i - 1, i):
        if 0 <= j < len(stamps):
            dt = abs(int(stamps[j]) - t)
            if best_dt is None or dt < best_dt:
                best, best_dt = j, dt
    if best is None or best_dt > tol_ns:
        return {}
    return hands[best]


# ----------------------------------------------------------------- plotting
def _overlay_2d(ax, reproj, hands, style, marker, color, lw):
    """Draw a reprojected skeleton overlay; return its mean px error vs detection."""
    errs = []
    for label, rp in reproj.items():
        fin = np.all(np.isfinite(rp), axis=1)
        for a, b in HAND_CONNECTIONS:
            if fin[a] and fin[b]:
                ax.plot([rp[a, 0], rp[b, 0]], [rp[a, 1], rp[b, 1]],
                        style, color=color, lw=lw, zorder=4)
        ax.scatter(rp[fin, 0], rp[fin, 1], s=14, marker=marker,
                   color=color, zorder=5)
        det = hands.get(label)
        if det is not None:
            m = fin & np.all(np.isfinite(det), axis=1)
            if m.any():
                errs.append(np.linalg.norm(rp[m] - det[m], axis=1))
    return float(np.concatenate(errs).mean()) if errs else None


def draw_hand_2d(ax, hands, width, height, title, reproj=None, pose_reproj=None):
    ax.clear()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)          # image coords: y down
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    # Detected 2D landmarks (solid, per-hand colour).
    for label, pts in hands.items():
        color = HAND_COLORS.get(label, "#888888")
        for a, b in HAND_CONNECTIONS:
            ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                    "-", color=color, lw=1.5)
        ax.scatter(pts[:, 0], pts[:, 1], s=10, color=color, zorder=3)

    # Triangulation reprojection (dashed magenta x) and pose reprojection
    # (dotted green +), each with its mean px error vs the detection.
    extra = ""
    if reproj:
        e = _overlay_2d(ax, reproj, hands, "--", "x", REPROJECT_COLOR, 1.0)
        if e is not None:
            extra += f"  tri {e:.1f}px"
    if pose_reproj:
        e = _overlay_2d(ax, pose_reproj, hands, ":", "+", POSE_COLOR, 1.2)
        if e is not None:
            extra += f"  pose {e:.1f}px"
    ax.set_title(title + extra, fontsize=9)


def draw_cameras_3d(ax, centers, axes_dirs, names):
    """Draw each camera center + a short optical-axis arrow, labelled."""
    for i, name in enumerate(names):
        c = centers[name]
        color = CAM_COLORS[i % len(CAM_COLORS)]
        ax.scatter(*c, s=60, color=color, marker="^", depthshade=False)
        ax.text(c[0], c[1], c[2], f"  {name}", color=color, fontsize=8)
        d = axes_dirs[name] * 0.1     # 10 cm optical-axis indicator
        ax.quiver(c[0], c[1], c[2], d[0], d[1], d[2],
                  color=color, length=1.0, normalize=False, alpha=0.7)


def _skeleton_3d(ax, J, color, style, lw, marker):
    fin = np.all(np.isfinite(J), axis=1)
    for a, b in HAND_CONNECTIONS:
        if fin[a] and fin[b]:
            ax.plot([J[a, 0], J[b, 0]], [J[a, 1], J[b, 1]], [J[a, 2], J[b, 2]],
                    style, color=color, lw=lw)
    ax.scatter(J[fin, 0], J[fin, 1], J[fin, 2], s=18, color=color,
               marker=marker, depthshade=False)


def draw_hand_3d(ax, joints_by_hand, centers, axes_dirs, names, info, limits,
                 pose_by_hand=None):
    # Preserve the user's current view orientation across the per-frame clear.
    elev, azim = ax.elev, ax.azim
    ax.clear()
    ax.set_title(info, fontsize=9)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")

    draw_cameras_3d(ax, centers, axes_dirs, names)

    # Triangulated hand (solid, per-hand colour).
    for label, J in joints_by_hand.items():
        _skeleton_3d(ax, J, HAND_COLORS.get(label, "#888888"), "-", 2, "o")

    # Estimated-pose hand alongside it (dashed green diamonds), not replacing it.
    if pose_by_hand:
        for label, J in pose_by_hand.items():
            _skeleton_3d(ax, J, POSE_COLOR, "--", 1.5, "D")

    apply_limits_3d(ax, limits)       # FIXED limits every frame
    ax.view_init(elev=elev, azim=azim)


# --------------------------------------------------------------------- main
def main():
    if not os.path.isfile(LOG_PATH):
        sys.exit(f"LOG_PATH not found: {LOG_PATH}")
    log = load_log(LOG_PATH)
    meta = log.meta
    cam_names = list(meta["cameras"])
    print(f"loaded {len(log.frames)} frame records; cameras={cam_names}")
    print(f"world_frame={meta.get('world_frame')} "
          f"landmarks_undistorted={meta.get('landmarks_undistorted')}")

    for c in TRIANGULATE_CAMERAS:
        if c not in meta["cameras"]:
            sys.exit(f"TRIANGULATE_CAMERAS has unknown camera '{c}'")
    if ENABLE_POSE_ESTIMATION and POSE_ESTIMATION_SOURCE not in meta["cameras"]:
        sys.exit(f"POSE_ESTIMATION_SOURCE has unknown camera "
                 f"'{POSE_ESTIMATION_SOURCE}'")

    # Per-camera calibration: projection matrix, center, optical axis, plus the
    # raw K / T_world_cam (needed by the pose estimator).
    P, centers, axes_dirs, resolution = {}, {}, {}, {}
    Kmat, Tmat = {}, {}
    for name, c in meta["cameras"].items():
        intr = c.get("intrinsics")
        T = c.get("T_world_cam")
        if intr is None or T is None:
            sys.exit(f"camera '{name}' is missing intrinsics/extrinsics in the log")
        K = np.asarray(intr["K"], float).reshape(3, 3)
        T = np.asarray(T, float).reshape(4, 4)
        Kmat[name], Tmat[name] = K, T
        P[name] = projection_matrix(K, T)
        centers[name] = T[:3, 3]
        axes_dirs[name] = T[:3, :3] @ np.array([0.0, 0.0, 1.0])  # +Z in world
        resolution[name] = intr.get("resolution", [1280, 720])

    print("camera positions (world frame):")
    for name in cam_names:
        print(f"  {name}: {np.round(centers[name], 4).tolist()}")

    # Per-camera time-indexed landmarks.
    indexed = {n: index_frames(log, n, HANDEDNESS) for n in cam_names}
    ref = TRIANGULATE_CAMERAS[0]
    timeline = indexed[ref][0]
    if len(timeline) == 0:
        sys.exit(f"no frames for reference camera '{ref}'")
    tol_ns = int(SYNC_TOLERANCE_S * 1e9)
    print(f"replaying {len(timeline)} steps off '{ref}' at {REPLAY_SPEED}x; "
          "close the window to stop.")

    # Inter-step wall times (s), scaled by replay speed.
    dts = np.diff(timeline).astype(float) / 1e9 / max(REPLAY_SPEED, 1e-6)

    cam_a, cam_b = TRIANGULATE_CAMERAS

    # FIXED 3D limits: one pass over the whole log to bound every triangulated
    # joint + the camera positions, so the 3D view never rescales during replay.
    bounds = [np.array([centers[n] for n in cam_names])]
    for t in timeline.tolist():
        ha = nearest(*indexed[cam_a], int(t), tol_ns)
        hb = nearest(*indexed[cam_b], int(t), tol_ns)
        for label in set(ha) & set(hb):
            J = triangulate_hand(ha[label]["image"], hb[label]["image"],
                                 P[cam_a], P[cam_b])
            bounds.append(J[np.all(np.isfinite(J), axis=1)])
    limits3d = equal_cube_limits(np.vstack(bounds))
    print("fixed 3D limits (m): "
          f"X{tuple(np.round(limits3d[0], 2))} "
          f"Y{tuple(np.round(limits3d[1], 2))} "
          f"Z{tuple(np.round(limits3d[2], 2))}")

    # Figure: 3 camera 2D views on top, one 3D view spanning the bottom.
    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.6])
    ax2d = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax3d = fig.add_subplot(gs[1, :], projection="3d")

    src = POSE_ESTIMATION_SOURCE

    def update(step):
        t = int(timeline[step])
        per_cam = {n: nearest(*indexed[n], t, tol_ns) for n in cam_names}
        # Detected pixels per camera as {label: (21,2)} for drawing/overlays.
        det = {n: {lbl: d["image"] for lbl, d in per_cam[n].items()}
               for n in cam_names}

        # Triangulate the chosen pair for every hand present in both views.
        joints = {}
        ha, hb = per_cam.get(cam_a, {}), per_cam.get(cam_b, {})
        for label in set(ha) & set(hb):
            joints[label] = triangulate_hand(ha[label]["image"],
                                             hb[label]["image"],
                                             P[cam_a], P[cam_b])

        # Monocular 6-DoF pose from the source camera -> hand placed in world.
        pose_joints = {}     # {label: (21,3) world} estimated-pose hand
        pose_info = ""
        if ENABLE_POSE_ESTIMATION:
            for label, d in per_cam.get(src, {}).items():
                if d["world"] is None:
                    continue
                r = pe.estimate_hand_pose(Kmat[src], Tmat[src],
                                          d["image"], d["world"])
                if r.success:
                    pose_joints[label] = pose_world_joints(r.T_world_hand,
                                                           d["world"])
            if pose_joints:
                pose_info = f"   pose<-{src}: {sorted(pose_joints)}"

        # Reprojections onto each camera (triangulation and/or estimated pose).
        reproj_by_cam, pose_reproj_by_cam = {}, {}
        for name in cam_names:
            if ENABLE_REPROJECT:
                reproj_by_cam[name] = {lbl: project_points(P[name], J)
                                       for lbl, J in joints.items()}
            if ENABLE_POSE_ESTIMATION and ENABLE_POSE_ESTIMATION_REPROJECT:
                pose_reproj_by_cam[name] = {lbl: project_points(P[name], J)
                                            for lbl, J in pose_joints.items()}

        # 2D: up to three cameras (pad if the log has fewer).
        for k in range(3):
            if k < len(cam_names):
                name = cam_names[k]
                w, h = resolution[name]
                draw_hand_2d(ax2d[k], det[name], w, h, f"{name} (2D)",
                             reproj=reproj_by_cam.get(name),
                             pose_reproj=pose_reproj_by_cam.get(name))
            else:
                ax2d[k].clear(); ax2d[k].axis("off")

        secs = (t - int(timeline[0])) / 1e9
        info = (f"3D hand  tri({cam_a}+{cam_b})={sorted(joints) or 'none'}"
                f"{pose_info}   t={secs:5.2f}s   step {step + 1}/{len(timeline)}")
        draw_hand_3d(ax3d, joints, centers, axes_dirs, cam_names, info, limits3d,
                     pose_by_hand=pose_joints)

    fig.tight_layout()
    plt.show(block=False)
    while plt.fignum_exists(fig.number):
        for step in range(len(timeline)):
            if not plt.fignum_exists(fig.number):
                break
            update(step)
            fig.canvas.draw_idle()
            pause = dts[step - 1] if step > 0 else 0.03
            plt.pause(float(np.clip(pause, 1e-3, 1.0)))
        if not LOOP:
            break
    if plt.fignum_exists(fig.number):
        plt.show()


if __name__ == "__main__":
    main()
