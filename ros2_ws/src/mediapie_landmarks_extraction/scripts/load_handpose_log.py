#!/usr/bin/env python3
"""Load a handpose JSONL log (written by mediapie_landmarks_extraction) for
offline evaluation in Python.

The file is newline-delimited JSON: line 1 is the session ``meta`` (per-camera
intrinsics/extrinsics, world frame, joint names); every later line is one
processed frame's detections for one camera.

Usage
-----
    from load_handpose_log import load_log, stack_camera

    log = load_log("handpose_log_20260626_120000.jsonl")
    log.meta["cameras"]["camera0"]["intrinsics"]["K"]      # 9-list, row-major
    log.meta["cameras"]["camera0"]["T_world_cam"]          # 4x4 (cam->world)

    # Dense arrays for one camera + handedness (NaN where the hand was absent):
    s = stack_camera(log, "camera0", "Left")
    s["stamp_ns"]   # (T,)
    s["image"]      # (T, 21, 3)  x_px, y_px, z_rel
    s["world"]      # (T, 21, 3)  metres, hand-local
    s["score"]      # (T,)

Or run it directly to print a summary:
    python3 load_handpose_log.py handpose_log_*.jsonl
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np

N_LANDMARKS = 21


@dataclass
class HandposeLog:
    meta: dict
    frames: list  # list of frame records (dicts), in file order


def load_log(path):
    """Parse a JSONL handpose log into ``HandposeLog(meta, frames)``."""
    meta = None
    frames = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "meta":
                meta = rec
            elif rec.get("type") == "frame":
                frames.append(rec)
    if meta is None:
        raise ValueError(f"{path}: no meta header line found")
    return HandposeLog(meta=meta, frames=frames)


def stack_camera(log, camera, handedness):
    """Dense per-frame arrays for one (camera, handedness).

    Frames where that hand is absent are filled with NaN, so all arrays share
    the camera's frame timeline. Returns a dict with keys
    ``stamp_ns (T,), image (T,21,3), world (T,21,3), score (T,)``.
    """
    recs = [f for f in log.frames if f["camera"] == camera]
    n = len(recs)
    stamp_ns = np.zeros(n, dtype=np.int64)
    image = np.full((n, N_LANDMARKS, 3), np.nan)
    world = np.full((n, N_LANDMARKS, 3), np.nan)
    score = np.full(n, np.nan)
    for i, rec in enumerate(recs):
        stamp_ns[i] = rec["stamp_ns"]
        hand = next(
            (h for h in rec["hands"] if h["handedness"] == handedness), None
        )
        if hand is None:
            continue
        score[i] = hand["score"]
        image[i] = np.asarray(hand["landmarks_image"], dtype=float)
        if hand.get("landmarks_world") is not None:
            world[i] = np.asarray(hand["landmarks_world"], dtype=float)
    return {"stamp_ns": stamp_ns, "image": image, "world": world, "score": score}


def _summary(path):
    log = load_log(path)
    cams = log.meta.get("cameras", {})
    print(f"{path}")
    print(f"  schema_version={log.meta.get('schema_version')} "
          f"created={log.meta.get('created')} "
          f"world_frame={log.meta.get('world_frame')!r} "
          f"landmarks_undistorted={log.meta.get('landmarks_undistorted')}")
    print(f"  cameras: {list(cams)}")
    for name, c in cams.items():
        has_K = c.get("intrinsics") is not None
        has_T = c.get("T_world_cam") is not None
        n = sum(1 for f in log.frames if f["camera"] == name)
        print(f"    {name}: frames={n} intrinsics={'yes' if has_K else 'NULL'} "
              f"extrinsics={'yes' if has_T else 'NULL'}")
    total_hands = sum(len(f["hands"]) for f in log.frames)
    print(f"  total frame records={len(log.frames)} hand detections={total_hands}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        _summary(p)
