#!/usr/bin/env python3
"""Offline multi-camera calibration from recorder sessions.

Reuses the (ROS-free) solver library of ros2_ws/src/calibration_multi_cam —
AprilGrid detection, keep-most-informative view selection, per-camera
intrinsics (radtan/equi), PnP + spanning-tree extrinsic init and bundle
adjustment — and replaces its ROS collection layer with a reader for the
session folders written by recorder.py.

Cameras are grouped into *rigs* (e.g. a fixed table pair and a head-mounted
pair). The relative pose between rigs is dynamic, so extrinsics are solved
independently per rig, each expressed in its user-chosen reference camera's
frame. Synchronization uses timestamps.csv: a rig view is kept only when every
rig camera has a *fresh* (non-duplicated) frame and the true capture-time skew
within the rig is below `sync_max_skew` — this matters for the moving rig,
where pairing a fresh frame with a stale one injects real pose error.

Outputs, written to <session>/calibration/:
    intrinsics.yaml   per camera: model, [fx,fy,cx,cy], distortion, rms
    extrinsics.yaml   per rig: reference camera + T_ref_cam per camera
    report.yaml       views, per-camera/per-rig RMS, coverage, warnings
"""
import csv
import os
import sys
import traceback

import cv2
import numpy as np
import yaml
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

MODELS = ["pinhole-radtan", "pinhole-equi"]

DEFAULT_SOLVER_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "ros2_ws", "src", "calibration_multi_cam",
))

DEFAULT_CALIBRATION = {
    "rigs": {
        "fixed": {"cameras": ["video0", "video2"], "reference": "video0"},
        "head": {"cameras": [], "reference": None},
    },
    "default_rig": "head",
    "models": {"video0": "pinhole-radtan", "video2": "pinhole-radtan"},
    "default_model": "pinhole-equi",
    "target": {
        "type": "aprilgrid",
        "family": "36h11",
        "tag_rows": 6,
        "tag_cols": 6,
        "tag_size": 0.03,
        "tag_spacing": 0.333,
        "border_bits": 2,
    },
    "frame_step": 1,
    "min_corners": 8,
    "novelty_min_pixel_motion": 12.0,
    "min_views_per_camera": 20,
    "max_views_per_camera": 80,
    "min_views_extrinsic": 30,
    "max_views_extrinsic": 150,
    "sync_max_skew": 0.015,
    "robust_loss": "huber",
    "robust_loss_scale": 1.0,
    "solver_path": None,
}


def merged_calibration_config(user_cfg):
    """DEFAULT_CALIBRATION overlaid with the user's `calibration:` section."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v)
           for k, v in DEFAULT_CALIBRATION.items()}
    for k, v in (user_cfg or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        elif v is not None:
            cfg[k] = v
    return cfg


def import_solver(solver_path=None):
    """Put the calibration_multi_cam package on sys.path and import it."""
    path = solver_path or DEFAULT_SOLVER_PATH
    if not os.path.isdir(os.path.join(path, "calibration_multi_cam")):
        raise RuntimeError(
            f"solver package not found at {path} "
            "(set calibration.solver_path in config.yaml)"
        )
    if path not in sys.path:
        sys.path.insert(0, path)
    from calibration_multi_cam import se3  # noqa: F401
    from calibration_multi_cam.bundle_adjust import bundle_adjust, per_camera_rms
    from calibration_multi_cam.extrinsics import init_extrinsics
    from calibration_multi_cam.intrinsics import calibrate_intrinsics
    from calibration_multi_cam.observations import ObservationDatabase
    from calibration_multi_cam.target import AprilGridTarget
    from calibration_multi_cam.view_buffer import MaximinViewBuffer, corner_features
    return {
        "se3": se3,
        "bundle_adjust": bundle_adjust,
        "per_camera_rms": per_camera_rms,
        "init_extrinsics": init_extrinsics,
        "calibrate_intrinsics": calibrate_intrinsics,
        "ObservationDatabase": ObservationDatabase,
        "AprilGridTarget": AprilGridTarget,
        "MaximinViewBuffer": MaximinViewBuffer,
        "corner_features": corner_features,
    }


def load_session(session_dir):
    """Read a recorder session: cameras, video paths, fps and per-tick sync data.

    Returns (labels, video_paths, fps, ticks) where ticks is a list of
    {label: (seq, capture_time)} — one entry per recorded frame index. For
    sessions without timestamps.csv every frame counts as fresh.
    """
    meta = {}
    meta_path = os.path.join(session_dir, "meta.yaml")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}

    video_paths = {}
    for name in sorted(os.listdir(session_dir)):
        if name.startswith("cam_") and not name.endswith((".yaml", ".csv")):
            label = os.path.splitext(name)[0][len("cam_"):]
            video_paths[label] = os.path.join(session_dir, name)
    if not video_paths:
        raise RuntimeError(f"no cam_* videos in {session_dir}")
    labels = meta.get("cameras") or sorted(video_paths)
    labels = [l for l in labels if l in video_paths]
    fps = float(meta.get("fps", 30.0))

    ticks = []
    csv_path = os.path.join(session_dir, "timestamps.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                tick = {}
                for l in labels:
                    try:
                        tick[l] = (int(row[f"{l}_seq"]),
                                   float(row[f"{l}_capture_time"]))
                    except (KeyError, ValueError):
                        tick[l] = (len(ticks), len(ticks) / fps)
                ticks.append(tick)
    else:
        n = min(int(cv2.VideoCapture(p).get(cv2.CAP_PROP_FRAME_COUNT))
                for p in video_paths.values())
        ticks = [{l: (i, i / fps) for l in labels} for i in range(n)]
    return labels, video_paths, fps, ticks


def coverage_fraction(views, resolution, grid=(8, 6)):
    """Fraction of image grid cells touched by any corner across the views."""
    w, h = resolution
    gx, gy = grid
    hit = np.zeros((gy, gx), dtype=bool)
    for _pids, pts in views:
        pts = np.asarray(pts).reshape(-1, 2)
        cx = np.clip((pts[:, 0] / w * gx).astype(int), 0, gx - 1)
        cy = np.clip((pts[:, 1] / h * gy).astype(int), 0, gy - 1)
        hit[cy, cx] = True
    return float(hit.mean())


class CalibrationWorker(QThread):
    """Runs the full pipeline off the GUI thread, reporting progress."""

    stage = pyqtSignal(str)          # current stage title
    progress = pyqtSignal(int, int)  # done, total (within the stage)
    log = pyqtSignal(str)
    preview = pyqtSignal(QImage)     # detection mosaic during extraction
    done = pyqtSignal(bool, str)     # success, summary/error text

    def __init__(self, session_dir, rig_of, reference_of, model_of, calib_cfg,
                 parent=None):
        """rig_of: {label: rig}; reference_of: {rig: label}; model_of: {label: model}."""
        super().__init__(parent)
        self.session_dir = session_dir
        self.rig_of = dict(rig_of)
        self.reference_of = dict(reference_of)
        self.model_of = dict(model_of)
        self.cfg = calib_cfg
        self._abort = False

    def stop(self):
        self._abort = True

    def run(self):
        try:
            summary = self._run()
        except Exception:  # noqa: BLE001
            self.done.emit(False, traceback.format_exc(limit=8))
            return
        self.done.emit(summary is not None, summary or "aborted")

    # ------------------------------------------------------------------ #
    def _run(self):
        cfg = self.cfg
        solver = import_solver(cfg.get("solver_path"))
        target = solver["AprilGridTarget"].from_params(cfg["target"])
        self.log.emit(f"target: {target}")

        labels, video_paths, fps, ticks = load_session(self.session_dir)
        cams = [l for l in labels if l in self.rig_of]
        if not cams:
            raise RuntimeError("no cameras assigned to a rig")
        rigs = {}
        for l in cams:
            rigs.setdefault(self.rig_of[l], []).append(l)
        self.log.emit(
            f"session: {os.path.basename(self.session_dir)} — "
            f"{len(ticks)} ticks @ {fps:.0f} fps, cameras {cams}, rigs {rigs}"
        )

        data = self._extract(solver, target, cams, rigs, video_paths, ticks)
        if data is None:
            return None
        intr_buf, rig_dbs, resolution = data

        intrinsics, intr_report = self._solve_intrinsics(
            solver, target, cams, intr_buf, resolution)
        extr, extr_report = self._solve_extrinsics(
            solver, target, rigs, rig_dbs, intrinsics)
        return self._write_outputs(rigs, intrinsics, intr_report, extr, extr_report)

    # ---- stage 1: corner extraction ----------------------------------- #
    def _extract(self, solver, target, cams, rigs, video_paths, ticks):
        cfg = self.cfg
        step = max(1, int(cfg["frame_step"]))
        min_corners = int(cfg["min_corners"])
        novelty_px = float(cfg["novelty_min_pixel_motion"])
        skew_max = float(cfg["sync_max_skew"])

        self.stage.emit("Extracting corners")
        caps = {}
        for l in cams:
            cap = cv2.VideoCapture(video_paths[l])
            if not cap.isOpened():
                raise RuntimeError(f"cannot open {video_paths[l]}")
            caps[l] = cap

        intr_buf = {l: solver["MaximinViewBuffer"](cfg["max_views_per_camera"])
                    for l in cams}
        intr_last = {}                  # label -> {pid: (x, y)} of last kept view
        rig_dbs = {}
        rig_last = {r: {} for r in rigs}
        resolution = {}
        last_seq = {l: None for l in cams}
        preview_every = max(1, len(ticks) // 40)

        for i, tick in enumerate(ticks):
            if self._abort:
                for c in caps.values():
                    c.release()
                return None
            process = (i % step == 0)
            frames, fresh = {}, {}
            for l in cams:
                if process:
                    ok, frame = caps[l].read()
                    if ok:
                        frames[l] = frame
                else:
                    caps[l].grab()
                seq = tick.get(l, (i, 0.0))[0]
                fresh[l] = last_seq[l] is None or seq != last_seq[l]
                last_seq[l] = seq
            if not process:
                continue
            if not frames:
                break  # all videos exhausted

            detections = {}
            for l, frame in frames.items():
                if l not in resolution:
                    resolution[l] = (frame.shape[1], frame.shape[0])
                if not fresh[l]:
                    continue  # duplicated frame (slow camera) — nothing new
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                pids, pts = target.detect(gray)
                if pids.size >= min_corners:
                    detections[l] = (pids, pts)

            # intrinsics: per camera, novelty-gated, diversity-thinned
            for l, (pids, pts) in detections.items():
                if _novel({l: (pids, pts)}, intr_last, novelty_px):
                    w, h = resolution[l]
                    feat = solver["corner_features"](pids, pts, w, h)
                    intr_buf[l].add((pids, pts), feat)
                    intr_last[l] = {int(p): tuple(q) for p, q in zip(pids, pts)}

            # extrinsics: per rig, all members fresh + seen + small time skew
            for rig, members in rigs.items():
                if len(members) < 2:
                    continue
                det = {l: detections[l] for l in members if l in detections}
                if len(det) < 2:
                    continue
                stamps = [tick[l][1] for l in det]
                if max(stamps) - min(stamps) > skew_max:
                    continue
                if not _novel(det, rig_last[rig], novelty_px):
                    continue
                if rig not in rig_dbs:
                    rig_dbs[rig] = solver["ObservationDatabase"](
                        members, capacity=cfg["max_views_extrinsic"],
                        resolutions=resolution)
                rig_dbs[rig].add_view(float(np.mean(stamps)), det)
                for l, (pids, pts) in det.items():
                    rig_last[rig][l] = {int(p): tuple(q) for p, q in zip(pids, pts)}

            if i % preview_every == 0:
                self._emit_preview(frames, detections, cams)
            self.progress.emit(i + 1, len(ticks))

        for c in caps.values():
            c.release()

        counts = {l: len(intr_buf[l]) for l in cams}
        self.log.emit(f"intrinsic views per camera: {counts}")
        for rig, members in rigs.items():
            if len(members) < 2:
                self.log.emit(f"rig '{rig}': single camera, extrinsics skipped")
            else:
                n = rig_dbs[rig].num_views if rig in rig_dbs else 0
                self.log.emit(f"rig '{rig}': {n} synchronized views")
        return intr_buf, rig_dbs, resolution

    def _emit_preview(self, frames, detections, cams):
        tiles = []
        for l in cams:
            if l not in frames:
                continue
            img = frames[l].copy()
            if l in detections:
                for x, y in np.asarray(detections[l][1]).reshape(-1, 2):
                    cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
            scale = 320.0 / img.shape[1]
            img = cv2.resize(img, (320, int(img.shape[0] * scale)))
            cv2.putText(img, f"{l}: {len(detections.get(l, ((), ()))[0])} corners",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(img)
        if not tiles:
            return
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 2,
                                    cv2.BORDER_CONSTANT) for t in tiles]
        mosaic = cv2.cvtColor(np.hstack(tiles), cv2.COLOR_BGR2RGB)
        mh, mw, ch = mosaic.shape
        self.preview.emit(QImage(mosaic.data, mw, mh, ch * mw,
                                 QImage.Format_RGB888).copy())

    # ---- stage 2: intrinsics ------------------------------------------- #
    def _solve_intrinsics(self, solver, target, cams, intr_buf, resolution):
        cfg = self.cfg
        self.stage.emit("Calibrating intrinsics")
        intrinsics, report = {}, {}
        for i, l in enumerate(cams):
            self.progress.emit(i, len(cams))
            if self._abort:
                raise RuntimeError("aborted")
            n = len(intr_buf[l])
            if n < int(cfg["min_views_per_camera"]):
                raise RuntimeError(
                    f"{l}: only {n}/{cfg['min_views_per_camera']} intrinsic views — "
                    "record a longer take with the board filling this camera's view"
                )
            model = self.model_of.get(l, cfg["default_model"])
            r = solver["calibrate_intrinsics"](
                target.object_points, intr_buf[l].items, resolution[l], model=model)
            cov = coverage_fraction(intr_buf[l].items, resolution[l])
            r["coverage"] = round(cov, 3)
            self.log.emit(
                f"{l}: {model} rms={r['reproj_rms']:.3f}px "
                f"({r['num_views']} views, coverage {cov * 100:.0f}%)"
            )
            if cov < 0.5:
                self.log.emit(
                    f"⚠ {l}: board covered only {cov * 100:.0f}% of the image — "
                    "distortion is poorly constrained, especially at the edges"
                )
            report[l] = {k: r[k] for k in
                         ("reproj_rms", "num_views", "coverage", "num_rejected")}
            intrinsics[l] = {k: r[k] for k in
                             ("model", "resolution", "intrinsics", "distortion",
                              "reproj_rms", "num_views")}
        self.progress.emit(len(cams), len(cams))
        return intrinsics, report

    # ---- stage 3: extrinsics per rig ------------------------------------ #
    def _solve_extrinsics(self, solver, target, rigs, rig_dbs, intrinsics):
        cfg = self.cfg
        extr, report = {}, {}
        for rig, members in rigs.items():
            if len(members) < 2:
                continue
            self.stage.emit(f"Calibrating extrinsics: rig '{rig}'")
            self.progress.emit(0, 3)
            if self._abort:
                raise RuntimeError("aborted")
            db = rig_dbs.get(rig)
            n = db.num_views if db else 0
            if n < int(cfg["min_views_extrinsic"]):
                raise RuntimeError(
                    f"rig '{rig}': only {n}/{cfg['min_views_extrinsic']} synchronized "
                    "views — record a take where both cameras see the board together"
                )
            ref = self.reference_of.get(rig) or members[0]
            names = [ref] + [l for l in members if l != ref]

            self.log.emit(f"rig '{rig}': PnP + chaining ({n} views, ref={ref})")
            cam_world, board_world, obs_struct, info = solver["init_extrinsics"](
                db.views, names, intrinsics, target.object_points,
                min_corners=int(cfg["min_corners"]))
            self.progress.emit(1, 3)
            self.log.emit(f"rig '{rig}': bundle adjustment "
                          f"({info['num_views_used']} views)…")
            cam_world, board_world, ba = solver["bundle_adjust"](
                cam_world, board_world, obs_struct, names, intrinsics,
                target.object_points, robust_loss=cfg["robust_loss"],
                loss_scale=cfg["robust_loss_scale"])
            self.progress.emit(2, 3)
            rms = solver["per_camera_rms"](
                cam_world, board_world, obs_struct, names, intrinsics,
                target.object_points)
            self.log.emit(
                f"rig '{rig}': BA rms {ba['rms_before']:.3f} → "
                f"{ba['rms_after']:.3f}px | per-camera "
                f"{ {k: round(v, 3) for k, v in rms.items()} }"
            )
            extr[rig] = {
                "reference": ref,
                "cameras": {name: {"T_ref_cam":
                                   solver["se3"].invert_T(cam_world[i]).tolist()}
                            for i, name in enumerate(names)},
            }
            report[rig] = {
                "num_views": info["num_views_used"],
                "pair_views": info["pair_views"],
                "rms_before": round(ba["rms_before"], 4),
                "rms_after": round(ba["rms_after"], 4),
                "per_camera_rms": {k: round(v, 4) for k, v in rms.items()},
            }
            self.progress.emit(3, 3)
        return extr, report

    # ---- stage 4: outputs ------------------------------------------------ #
    def _write_outputs(self, rigs, intrinsics, intr_report, extr, extr_report):
        out_dir = os.path.join(self.session_dir, "calibration")
        os.makedirs(out_dir, exist_ok=True)
        self.stage.emit("Writing results")

        with open(os.path.join(out_dir, "intrinsics.yaml"), "w") as f:
            yaml.safe_dump({"cameras": intrinsics}, f,
                           default_flow_style=None, sort_keys=False)
        with open(os.path.join(out_dir, "extrinsics.yaml"), "w") as f:
            yaml.safe_dump({"rigs": extr}, f,
                           default_flow_style=None, sort_keys=False)
        with open(os.path.join(out_dir, "report.yaml"), "w") as f:
            yaml.safe_dump({
                "session": os.path.basename(self.session_dir),
                "rigs": {r: list(m) for r, m in rigs.items()},
                "intrinsics": intr_report,
                "extrinsics": extr_report,
            }, f, default_flow_style=None, sort_keys=False)

        lines = [f"results → {out_dir}"]
        for l, r in intr_report.items():
            lines.append(f"  {l}: intrinsic rms {r['reproj_rms']:.3f}px")
        for rig, r in extr_report.items():
            lines.append(f"  rig '{rig}': extrinsic rms {r['rms_after']:.3f}px "
                         f"({r['num_views']} views)")
        summary = "\n".join(lines)
        self.log.emit(summary)
        return summary


def _novel(detections, last_corners, novelty_px):
    """True if any camera's corners moved >= novelty_px since its last kept view."""
    for l, (pids, pts) in detections.items():
        prev = last_corners.get(l)
        if prev is None:
            return True
        shared = [(p, prev[int(i)]) for i, p in zip(pids, pts) if int(i) in prev]
        if not shared:
            return True
        if np.mean([np.hypot(p[0] - q[0], p[1] - q[1])
                    for p, q in shared]) >= novelty_px:
            return True
    return False
