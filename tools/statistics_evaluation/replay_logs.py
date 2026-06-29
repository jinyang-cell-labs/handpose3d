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
TRIANGULATE_CAMERAS = ("camera0", "camera1")
# Replay speed: 1.0 = real time, 2.0 = twice as fast, 0.5 = half speed.
REPLAY_SPEED = 1.0
# Which hand to show: "Left", "Right", or None for whatever is present.
HANDEDNESS = None
# Max time difference (s) for a camera's detection to count as "the same moment".
SYNC_TOLERANCE_S = 0.05
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


def set_axes_equal_3d(ax, pts):
    """Equal aspect for a 3D axis around the given (N,3) points."""
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return
    mins, maxs = pts.min(0), pts.max(0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 0.1)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


# ------------------------------------------------------------------- loading
def index_frames(log, camera, handedness):
    """Per-camera sorted (stamps, hands) where hands = {label: (21,2) pixels}."""
    stamps, hands = [], []
    for rec in log.frames:
        if rec["camera"] != camera:
            continue
        by_label = {}
        for h in rec["hands"]:
            if handedness is not None and h["handedness"] != handedness:
                continue
            img = np.asarray(h["landmarks_image"], float)[:, :2]  # x,y px
            by_label[h["handedness"]] = img
        stamps.append(rec["stamp_ns"])
        hands.append(by_label)
    order = np.argsort(stamps)
    return np.asarray(stamps)[order], [hands[i] for i in order]


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
def draw_hand_2d(ax, hands, width, height, title):
    ax.clear()
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)          # image coords: y down
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for label, pts in hands.items():
        color = HAND_COLORS.get(label, "#888888")
        for a, b in HAND_CONNECTIONS:
            ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                    "-", color=color, lw=1.5)
        ax.scatter(pts[:, 0], pts[:, 1], s=10, color=color, zorder=3)


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


def draw_hand_3d(ax, joints_by_hand, centers, axes_dirs, names, info):
    ax.clear()
    ax.set_title(info, fontsize=9)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")

    draw_cameras_3d(ax, centers, axes_dirs, names)

    all_pts = [np.array([centers[n] for n in names])]
    for label, J in joints_by_hand.items():
        color = HAND_COLORS.get(label, "#888888")
        fin = np.all(np.isfinite(J), axis=1)
        for a, b in HAND_CONNECTIONS:
            if fin[a] and fin[b]:
                ax.plot([J[a, 0], J[b, 0]], [J[a, 1], J[b, 1]],
                        [J[a, 2], J[b, 2]], "-", color=color, lw=2)
        ax.scatter(J[fin, 0], J[fin, 1], J[fin, 2], s=18, color=color,
                   depthshade=False)
        all_pts.append(J[fin])
    set_axes_equal_3d(ax, np.vstack(all_pts))


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

    # Per-camera calibration: projection matrix, center, optical axis (world).
    P, centers, axes_dirs, resolution = {}, {}, {}, {}
    for name, c in meta["cameras"].items():
        intr = c.get("intrinsics")
        T = c.get("T_world_cam")
        if intr is None or T is None:
            sys.exit(f"camera '{name}' is missing intrinsics/extrinsics in the log")
        K = np.asarray(intr["K"], float).reshape(3, 3)
        T = np.asarray(T, float).reshape(4, 4)
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

    # Figure: 3 camera 2D views on top, one 3D view spanning the bottom.
    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.6])
    ax2d = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax3d = fig.add_subplot(gs[1, :], projection="3d")
    cam_a, cam_b = TRIANGULATE_CAMERAS

    def update(step):
        t = int(timeline[step])
        per_cam = {n: nearest(*indexed[n], t, tol_ns) for n in cam_names}
        # 2D: up to three cameras (pad if the log has fewer).
        for k in range(3):
            if k < len(cam_names):
                name = cam_names[k]
                w, h = resolution[name]
                draw_hand_2d(ax2d[k], per_cam[name], w, h, f"{name} (2D)")
            else:
                ax2d[k].clear(); ax2d[k].axis("off")
        # 3D: triangulate the chosen pair for every hand present in both.
        joints = {}
        ha, hb = per_cam.get(cam_a, {}), per_cam.get(cam_b, {})
        for label in set(ha) & set(hb):
            joints[label] = triangulate_hand(ha[label], hb[label],
                                             P[cam_a], P[cam_b])
        secs = (t - int(timeline[0])) / 1e9
        info = (f"3D hand  ({cam_a} + {cam_b})   "
                f"t={secs:5.2f}s   step {step + 1}/{len(timeline)}   "
                f"hands: {sorted(joints) or 'none'}")
        draw_hand_3d(ax3d, joints, centers, axes_dirs, cam_names, info)

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
