#!/usr/bin/env python3
"""Multi-camera synchronized recorder / player GUI.

Opens N V4L2 cameras (configured in config.yaml), shows live previews in a
grid, and records one video file per camera. Recording is frame-locked: a
single clock thread ticks at the target fps and, on every tick, writes the
most recent frame of every camera. Frame i in every file therefore belongs
to the same tick, and timestamps.csv stores each frame's true capture time
so synchronization can be verified offline.

Each recording session is a folder:

    <output_dir>/session_YYYYmmdd_HHMMSS/
        cam_video0.avi
        cam_video2.avi
        ...
        meta.yaml         # cameras, resolution, fps, start time, clock anchor
        timestamps.csv    # tick_idx, tick_time, per-camera seq + capture time

The Playback tab lists sessions and replays all files of a session
frame-locked, with play/pause and a seek slider.

Usage:
    python recorder.py [path/to/config.yaml]
"""
import csv
import glob
import os
import re
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import yaml
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from calibrator import MODELS, CalibrationWorker, merged_calibration_config
from phone_panel import PhonePanel
from pose_sync import DEFAULT_MAX_GAP_MS, OUTPUT_NAMES, SyncWorker

DEFAULT_CONFIG = {
    "cameras": [0, 2, 4, 6],
    "width": 1280,
    "height": 720,
    "fps": 30,
    "capture_fourcc": "MJPG",
    "record_fourcc": "MJPG",
    "record_extension": "avi",
    "output_dir": "~/recordings/multi_cam",
    "preview_fps": 20,
    "grid_columns": 2,
}


def load_config(path):
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in user.items() if v is not None})
    cfg["output_dir"] = os.path.expanduser(str(cfg["output_dir"]))
    return cfg


def camera_label(dev):
    """0 -> 'video0', '/dev/video4' -> 'video4', by-id symlink -> 'videoN'."""
    if isinstance(dev, int):
        return f"video{dev}"
    # /dev/v4l/by-id/... symlinks resolve to the current /dev/videoN node.
    real = os.path.realpath(str(dev)) if os.path.exists(str(dev)) else str(dev)
    m = re.search(r"(video\d+)$", real)
    return m.group(1) if m else re.sub(r"\W+", "_", str(dev)).strip("_")


def sysfs_device_name(node):
    """Human-readable camera name of /dev/videoN from sysfs."""
    base = os.path.basename(node)
    try:
        with open(f"/sys/class/video4linux/{base}/name") as f:
            return f.read().strip()
    except OSError:
        return base


def list_capture_devices():
    """Enumerate capture-capable V4L2 devices as [(device, display_name)].

    Prefers /dev/v4l/by-id: the *-video-index0 symlink of each camera is its
    capture node (index1 is the UVC metadata node) and the path is stable
    across replugging. Nodes without any by-id link (e.g. v4l2loopback) are
    added by their /dev/videoN path.
    """
    devices, linked = [], set()
    by_id = "/dev/v4l/by-id"
    if os.path.isdir(by_id):
        for entry in sorted(os.listdir(by_id)):
            path = os.path.join(by_id, entry)
            node = os.path.realpath(path)
            linked.add(node)
            if entry.endswith("video-index0") and re.search(r"video\d+$", node):
                name = sysfs_device_name(node)
                devices.append((path, f"{os.path.basename(node)} — {name}"))
    for node in glob.glob("/dev/video[0-9]*"):
        if node in linked:
            continue
        # sysfs index is 0 for the capture node, 1+ for metadata siblings.
        try:
            with open(f"/sys/class/video4linux/{os.path.basename(node)}/index") as f:
                if int(f.read().strip()) != 0:
                    continue
        except (OSError, ValueError):
            pass
        devices.append((node, f"{os.path.basename(node)} — {sysfs_device_name(node)}"))

    def node_number(dev):
        m = re.search(r"(\d+)$", camera_label(dev))
        return int(m.group(1)) if m else 9999

    devices.sort(key=lambda d: node_number(d[0]))
    return devices


def to_qimage(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    # copy() detaches the QImage from the numpy buffer.
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
class CameraWorker(QThread):
    """Continuously reads one camera and keeps only the latest frame.

    The recorder thread samples `latest()` so a slow or stalled camera never
    blocks the common recording clock.
    """

    opened = pyqtSignal(str, int, int, float)  # label, width, height, fps
    failed = pyqtSignal(str, str)  # label, message

    def __init__(self, device, cfg, parent=None):
        super().__init__(parent)
        self.device = device
        self.label = camera_label(device)
        self._cfg = cfg
        self._running = True
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0.0
        self._seq = -1

    def latest(self):
        """Returns (frame, capture_time, seq) or (None, 0.0, -1)."""
        with self._lock:
            return self._frame, self._stamp, self._seq

    def stop(self):
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*self._cfg["capture_fourcc"]),
            )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg["height"])
            cap.set(cv2.CAP_PROP_FPS, self._cfg["fps"])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self.failed.emit(self.label, "cannot open device")
            return

        ok, frame = cap.read()
        if not ok:
            cap.release()
            self.failed.emit(self.label, "device opened but returns no frames")
            return
        h, w = frame.shape[:2]
        self.opened.emit(self.label, w, h, cap.get(cv2.CAP_PROP_FPS))

        while self._running:
            ok, frame = cap.read()
            if not ok:
                self.failed.emit(self.label, "stream ended / read error")
                break
            with self._lock:
                self._frame = frame
                self._stamp = time.monotonic()
                self._seq += 1
        cap.release()


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
class RecordWorker(QThread):
    """Frame-locked recorder: one clock, one writer per camera.

    On every tick (1/fps) it writes the latest frame of every camera, so all
    files stay frame-aligned. If a camera has not produced a new frame since
    the last tick its previous frame is duplicated; the per-camera `seq` in
    timestamps.csv exposes such duplicates.
    """

    started = pyqtSignal(str)  # session dir
    finished_session = pyqtSignal(str, int)  # session dir, frames
    error = pyqtSignal(str)

    def __init__(self, workers, session_dir, cfg, parent=None):
        super().__init__(parent)
        self._workers = workers  # only live cameras
        self._dir = session_dir
        self._cfg = cfg
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        fps = float(self._cfg["fps"])
        fourcc = cv2.VideoWriter_fourcc(*self._cfg["record_fourcc"])
        ext = self._cfg["record_extension"]
        os.makedirs(self._dir, exist_ok=True)

        writers = {}
        for w in self._workers:
            frame, _, _ = w.latest()
            if frame is None:
                continue
            h, wdt = frame.shape[:2]
            path = os.path.join(self._dir, f"cam_{w.label}.{ext}")
            vw = cv2.VideoWriter(path, fourcc, fps, (wdt, h))
            if not vw.isOpened():
                self.error.emit(f"cannot open writer {path}")
                for v in writers.values():
                    v.release()
                return
            writers[w.label] = vw
        if not writers:
            self.error.emit("no camera is delivering frames")
            return

        labels = list(writers.keys())
        # Clock anchor bridging the two logs' time bases: camera frame times
        # (timestamps.csv) are time.monotonic, phone poses (phone_pose.jsonl)
        # are time.time_ns wall-clock — different epochs. Capturing both clocks
        # back-to-back lets an offline consumer convert between them:
        #   wall_ns = monotonic_ns + (clock_anchor.wall_ns - clock_anchor.monotonic_ns)
        # Their relative drift over a session is sub-millisecond (well below the
        # phone's own clock-sync uncertainty), so one anchor per session suffices.
        anchor_mono_ns = time.monotonic_ns()
        anchor_wall_ns = time.time_ns()
        with open(os.path.join(self._dir, "meta.yaml"), "w") as f:
            yaml.safe_dump(
                {
                    "cameras": labels,
                    # device each camera was opened from (by-id paths are
                    # stable identities across replug/renumbering)
                    "devices": {w.label: str(w.device) for w in self._workers
                                if w.label in writers},
                    "width": self._cfg["width"],
                    "height": self._cfg["height"],
                    "fps": fps,
                    "record_fourcc": self._cfg["record_fourcc"],
                    "started": datetime.now().isoformat(timespec="seconds"),
                    "clock_anchor": {
                        "monotonic_ns": anchor_mono_ns,
                        "wall_ns": anchor_wall_ns,
                    },
                },
                f,
            )

        csv_file = open(os.path.join(self._dir, "timestamps.csv"), "w", newline="")
        writer_csv = csv.writer(csv_file)
        writer_csv.writerow(
            ["tick_idx", "tick_time"]
            + [f"{l}_seq" for l in labels]
            + [f"{l}_capture_time" for l in labels]
        )

        self.started.emit(self._dir)
        period = 1.0 / fps
        t0 = time.monotonic()
        tick = 0
        by_label = {w.label: w for w in self._workers}
        try:
            while self._running:
                target = t0 + tick * period
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                now = time.monotonic()
                seqs, stamps = [], []
                for l in labels:
                    frame, stamp, seq = by_label[l].latest()
                    writers[l].write(frame)
                    seqs.append(seq)
                    stamps.append(f"{stamp:.6f}")
                writer_csv.writerow([tick, f"{now:.6f}"] + seqs + stamps)
                tick += 1
        finally:
            for v in writers.values():
                v.release()
            csv_file.close()
            self.finished_session.emit(self._dir, tick)


# ---------------------------------------------------------------------------
# Record tab
# ---------------------------------------------------------------------------
class CameraTile(QWidget):
    def __init__(self, label, selectable=False):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self.combo = QComboBox()
        self.combo.setVisible(selectable)
        lay.addWidget(self.combo)
        self.view = QLabel("no camera")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(320, 180)
        self.view.setStyleSheet("background:#111; color:#888;")
        self.info = QLabel(label)
        self.info.setStyleSheet("color:#444;")
        lay.addWidget(self.view, stretch=1)
        lay.addWidget(self.info)

    def show_frame(self, frame_bgr):
        pix = QPixmap.fromImage(to_qimage(frame_bgr)).scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.FastTransformation
        )
        self.view.setPixmap(pix)

    def show_dead(self, msg):
        self.view.setPixmap(QPixmap())
        self.view.setText(f"no signal\n{msg}")


class RecordTab(QWidget):
    session_saved = pyqtSignal()

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.devices = list_capture_devices()
        self.tiles = []    # one per preview window
        self.workers = []  # CameraWorker or None, aligned with self.tiles
        self.recorder = None
        self._build_ui()
        for idx, dev in enumerate(self.cfg["cameras"]):
            if dev is not None:
                self._select_device(idx, dev)

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._refresh_previews)
        self.preview_timer.start(int(1000 / max(1, cfg["preview_fps"])))

    def _build_ui(self):
        root = QVBoxLayout(self)

        grid_box = QWidget()
        self.grid = QGridLayout(grid_box)
        root.addWidget(grid_box, stretch=1)

        cols = max(1, int(self.cfg["grid_columns"]))
        for idx in range(len(self.cfg["cameras"])):
            tile = CameraTile(f"window {idx}", selectable=True)
            self._populate_combo(tile.combo, None)
            tile.combo.currentIndexChanged.connect(
                lambda _i, i=idx: self._on_device_chosen(i)
            )
            self.grid.addWidget(tile, idx // cols, idx % cols)
            self.tiles.append(tile)
            self.workers.append(None)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output folder"))
        self.dir_edit = QLineEdit(self.cfg["output_dir"])
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_dir)
        out_row.addWidget(self.dir_edit, stretch=1)
        out_row.addWidget(browse)
        root.addLayout(out_row)

        # Phone pose (ARCore) — recorded into the same session folder, started
        # and stopped together with the cameras by the Record button below.
        self.phone = PhonePanel(self.cfg)
        root.addWidget(self.phone)

        btn_row = QHBoxLayout()
        self.record_btn = QPushButton("Start recording")
        self.record_btn.setMinimumHeight(40)
        self.record_btn.clicked.connect(self._toggle_record)
        self.rescan_btn = QPushButton("Rescan devices")
        self.rescan_btn.clicked.connect(self._rescan_devices)
        btn_row.addWidget(self.record_btn, stretch=1)
        btn_row.addWidget(self.rescan_btn)
        root.addLayout(btn_row)

        self.status = QLabel("idle")
        self.status.setStyleSheet("color:#444;")
        root.addWidget(self.status)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Choose output folder", self.dir_edit.text()
        )
        if d:
            self.dir_edit.setText(d)

    # ---- device selection --------------------------------------------------
    @staticmethod
    def _same_device(a, b):
        if a is None or b is None:
            return False

        def norm(d):
            path = f"/dev/video{d}" if isinstance(d, int) else str(d)
            return os.path.realpath(path)

        return norm(a) == norm(b)

    def _combo_index_of(self, combo, dev):
        for i in range(combo.count()):
            if self._same_device(combo.itemData(i), dev):
                return i
        return -1

    def _populate_combo(self, combo, selected_dev):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(none)", None)
        for dev, name in self.devices:
            combo.addItem(name, dev)
        if selected_dev is not None:
            i = self._combo_index_of(combo, selected_dev)
            if i < 0:
                combo.addItem(f"{camera_label(selected_dev)} (not found)", selected_dev)
                i = combo.count() - 1
            combo.setCurrentIndex(i)
        combo.blockSignals(False)

    def _select_device(self, idx, dev):
        """Programmatically select `dev` in window idx and start its worker."""
        self._populate_combo(self.tiles[idx].combo, dev)
        self._stop_worker(idx)
        self._start_worker(idx, dev)

    def _on_device_chosen(self, idx):
        dev = self.tiles[idx].combo.currentData()
        if dev is not None:
            for j, tile in enumerate(self.tiles):
                if j != idx and self._same_device(tile.combo.currentData(), dev):
                    QMessageBox.warning(
                        self,
                        "Device in use",
                        f"{camera_label(dev)} is already selected in window {j}.",
                    )
                    combo = self.tiles[idx].combo
                    combo.blockSignals(True)
                    combo.setCurrentIndex(0)
                    combo.blockSignals(False)
                    dev = None
                    break
        self._stop_worker(idx)
        if dev is None:
            tile = self.tiles[idx]
            tile.view.setPixmap(QPixmap())
            tile.view.setText("no camera")
            tile.info.setText(f"window {idx}")
        else:
            self._start_worker(idx, dev)

    def _rescan_devices(self):
        if self.recorder:
            return
        self.devices = list_capture_devices()
        for tile in self.tiles:
            self._populate_combo(tile.combo, tile.combo.currentData())
        self.status.setText(f"found {len(self.devices)} capture devices")

    # ---- cameras ---------------------------------------------------------
    def _start_worker(self, idx, dev):
        worker = CameraWorker(dev, self.cfg)
        tile = self.tiles[idx]
        tile.view.setPixmap(QPixmap())
        tile.view.setText("opening…")
        tile.info.setText(f"{worker.label} — opening")
        worker.opened.connect(
            lambda l, w, h, f, i=idx: self._on_camera_opened(i, l, w, h, f)
        )
        worker.failed.connect(lambda l, m, i=idx: self._on_camera_failed(i, l, m))
        worker.start()
        self.workers[idx] = worker

    def _stop_worker(self, idx):
        worker = self.workers[idx]
        if worker:
            worker.stop()
            worker.wait(2000)
            self.workers[idx] = None

    def _close_cameras(self):
        for w in self.workers:
            if w:
                w.stop()
        for i, w in enumerate(self.workers):
            if w:
                w.wait(2000)
                self.workers[i] = None

    def _on_camera_opened(self, idx, label, w, h, fps):
        if self.sender() is not self.workers[idx]:
            return  # stale signal from a replaced worker
        self.tiles[idx].info.setText(f"{label} — {w}x{h} @ {fps:.0f} fps")

    def _on_camera_failed(self, idx, label, msg):
        if self.sender() is not self.workers[idx]:
            return
        self.tiles[idx].show_dead(msg)
        self.tiles[idx].info.setText(f"{label} — {msg}")

    def _live_workers(self):
        return [w for w in self.workers if w and w.latest()[0] is not None]

    def _refresh_previews(self):
        for tile, worker in zip(self.tiles, self.workers):
            if worker:
                frame, _, _ = worker.latest()
                if frame is not None:
                    tile.show_frame(frame)

    # ---- recording ---------------------------------------------------------
    def _toggle_record(self):
        if self.recorder:
            self.recorder.stop()
            self.phone.end_recording()
            self.record_btn.setEnabled(False)
            return

        live = self._live_workers()
        if not live:
            QMessageBox.warning(self, "No cameras", "No camera is delivering frames.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.dir_edit.text(), f"session_{ts}")
        self.recorder = RecordWorker(live, session_dir, self.cfg)
        self.recorder.started.connect(
            lambda d: self.status.setText(f"● recording → {d}")
        )
        self.recorder.finished_session.connect(self._on_record_finished)
        self.recorder.error.connect(self._on_record_error)
        self.recorder.start()
        # Log the phone pose into the same session folder (no-op if the phone
        # is not connected or 'Record with cameras' is unchecked).
        self.phone.begin_recording(session_dir)
        self.phone.set_locked(True)
        self.record_btn.setText("Stop recording")
        self.record_btn.setStyleSheet("background:#b00; color:white;")
        self.rescan_btn.setEnabled(False)
        for tile in self.tiles:
            tile.combo.setEnabled(False)

    def _on_record_finished(self, session_dir, frames):
        self.status.setText(f"saved {frames} frames per camera → {session_dir}")
        self._reset_record_ui()
        self.session_saved.emit()

    def _on_record_error(self, msg):
        self.status.setText(f"⚠ {msg}")
        self._reset_record_ui()

    def _reset_record_ui(self):
        # finished_session is emitted from run()'s finally, so the QThread may
        # not have fully exited yet; wait before dropping the last reference,
        # or Python GC destroys a still-running QThread and aborts.
        if self.recorder:
            self.recorder.wait(3000)
        self.recorder = None
        self.phone.set_locked(False)
        self.record_btn.setText("Start recording")
        self.record_btn.setStyleSheet("")
        self.record_btn.setEnabled(True)
        self.rescan_btn.setEnabled(True)
        for tile in self.tiles:
            tile.combo.setEnabled(True)

    def shutdown(self):
        if self.recorder:
            self.recorder.stop()
            self.recorder.wait(3000)
        self._close_cameras()
        self.phone.shutdown()


# ---------------------------------------------------------------------------
# Playback tab
# ---------------------------------------------------------------------------
class PlaybackTab(QWidget):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.caps = {}  # label -> VideoCapture
        self.tiles = {}
        self.total_frames = 0
        self.pos = 0
        self.playing = False
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._step)
        self.refresh_sessions()

    def _build_ui(self):
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        left.addWidget(QLabel("Recorded sessions"))
        self.session_list = QListWidget()
        self.session_list.itemSelectionChanged.connect(self._load_selected)
        left.addWidget(self.session_list, stretch=1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_sessions)
        left.addWidget(refresh)
        left_box = QWidget()
        left_box.setLayout(left)
        left_box.setMaximumWidth(280)
        root.addWidget(left_box)

        right = QVBoxLayout()
        grid_box = QWidget()
        self.grid = QGridLayout(grid_box)
        right.addWidget(grid_box, stretch=1)

        ctl = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self._seek)
        self.pos_label = QLabel("0 / 0")
        ctl.addWidget(self.play_btn)
        ctl.addWidget(self.slider, stretch=1)
        ctl.addWidget(self.pos_label)
        right.addLayout(ctl)

        self.status = QLabel("select a session")
        self.status.setStyleSheet("color:#444;")
        right.addWidget(self.status)
        root.addLayout(right, stretch=1)

    # ---- session handling --------------------------------------------------
    def refresh_sessions(self):
        self.session_list.clear()
        base = self.cfg["output_dir"]
        if not os.path.isdir(base):
            return
        for name in sorted(os.listdir(base), reverse=True):
            path = os.path.join(base, name)
            if os.path.isdir(path) and any(
                f.startswith("cam_") for f in os.listdir(path)
            ):
                self.session_list.addItem(name)

    def _close_session(self):
        self.timer.stop()
        self.playing = False
        self.play_btn.setText("Play")
        for cap in self.caps.values():
            cap.release()
        self.caps = {}
        for tile in self.tiles.values():
            tile.setParent(None)
        self.tiles = {}

    def _load_selected(self):
        items = self.session_list.selectedItems()
        if not items:
            return
        self._close_session()
        session_dir = os.path.join(self.cfg["output_dir"], items[0].text())

        fps = float(self.cfg["fps"])
        meta_path = os.path.join(session_dir, "meta.yaml")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = yaml.safe_load(f) or {}
            fps = float(meta.get("fps", fps))
        self.fps = fps

        videos = sorted(
            f
            for f in os.listdir(session_dir)
            if f.startswith("cam_") and not f.endswith((".yaml", ".csv"))
        )
        cols = max(1, int(self.cfg["grid_columns"]))
        totals = []
        for i, name in enumerate(videos):
            cap = cv2.VideoCapture(os.path.join(session_dir, name))
            if not cap.isOpened():
                continue
            label = os.path.splitext(name)[0]
            self.caps[label] = cap
            totals.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            tile = CameraTile(label)
            self.grid.addWidget(tile, i // cols, i % cols)
            self.tiles[label] = tile
        if not self.caps:
            self.status.setText("⚠ no playable videos in this session")
            self.play_btn.setEnabled(False)
            self.slider.setEnabled(False)
            return

        self.total_frames = min(totals)
        self.pos = 0
        self.slider.setRange(0, max(0, self.total_frames - 1))
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.status.setText(
            f"{len(self.caps)} videos, {self.total_frames} frames @ {fps:.0f} fps"
        )
        self._show_frame_at(0)

    # ---- playback ------------------------------------------------------------
    def _toggle_play(self):
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.play_btn.setText("Play")
        else:
            if self.pos >= self.total_frames - 1:
                self._seek(0)
            self.timer.start(int(1000 / self.fps))
            self.playing = True
            self.play_btn.setText("Pause")

    def _step(self):
        if self.pos >= self.total_frames - 1:
            self._toggle_play()
            return
        self.pos += 1
        for label, cap in self.caps.items():
            ok, frame = cap.read()
            if ok:
                self.tiles[label].show_frame(frame)
        self.slider.blockSignals(True)
        self.slider.setValue(self.pos)
        self.slider.blockSignals(False)
        self.pos_label.setText(f"{self.pos} / {self.total_frames - 1}")

    def _seek(self, frame_idx):
        self.pos = frame_idx
        self._show_frame_at(frame_idx)

    def _show_frame_at(self, frame_idx):
        for label, cap in self.caps.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if ok:
                self.tiles[label].show_frame(frame)
        self.pos_label.setText(f"{frame_idx} / {max(0, self.total_frames - 1)}")

    def shutdown(self):
        self._close_session()


# ---------------------------------------------------------------------------
# Calibrate tab
# ---------------------------------------------------------------------------
class CalibrateTab(QWidget):
    """Offline rig calibration from a recorded session.

    Assign each camera of the selected session to a rig (fixed / head-mounted),
    pick each rig's reference camera (its 0,0,0), then run: intrinsics per
    camera first, then per-rig extrinsics. Progress, detection previews and the
    solver log are shown live; results land in <session>/calibration/.
    """

    EXCLUDE = "(exclude)"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.calib_cfg = merged_calibration_config(cfg.get("calibration"))
        self.rig_names = list(self.calib_cfg["rigs"].keys())
        self.worker = None
        self._build_ui()
        self.refresh_sessions()

    def _build_ui(self):
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        left.addWidget(QLabel("Recorded sessions"))
        self.session_list = QListWidget()
        self.session_list.itemSelectionChanged.connect(self._load_selected)
        left.addWidget(self.session_list, stretch=1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_sessions)
        left.addWidget(refresh)
        left_box = QWidget()
        left_box.setLayout(left)
        left_box.setMaximumWidth(280)
        root.addWidget(left_box)

        right = QVBoxLayout()
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Camera", "Rig", "Model", "Reference (0,0,0)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(180)
        right.addWidget(self.table)

        self.run_btn = QPushButton("Run calibration")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._toggle_run)
        right.addWidget(self.run_btn)

        self.stage_label = QLabel("select a session")
        self.stage_label.setStyleSheet("font-weight: bold;")
        right.addWidget(self.stage_label)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        right.addWidget(self.progress)

        self.preview = QLabel("detection preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(200)
        self.preview.setStyleSheet("background:#111; color:#888;")
        right.addWidget(self.preview, stretch=1)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.log_view.setMinimumHeight(120)
        right.addWidget(self.log_view)
        root.addLayout(right, stretch=1)

    # ---- sessions ---------------------------------------------------------
    def refresh_sessions(self):
        self.session_list.clear()
        base = self.cfg["output_dir"]
        if not os.path.isdir(base):
            return
        for name in sorted(os.listdir(base), reverse=True):
            path = os.path.join(base, name)
            if os.path.isdir(path) and any(
                f.startswith("cam_") for f in os.listdir(path)
            ):
                self.session_list.addItem(name)

    def _session_dir(self):
        items = self.session_list.selectedItems()
        if not items:
            return None
        return os.path.join(self.cfg["output_dir"], items[0].text())

    def _load_selected(self):
        session_dir = self._session_dir()
        if not session_dir or self.worker:
            return
        cameras = []
        meta_path = os.path.join(session_dir, "meta.yaml")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                cameras = (yaml.safe_load(f) or {}).get("cameras", [])
        if not cameras:
            cameras = sorted(
                os.path.splitext(f)[0][len("cam_"):]
                for f in os.listdir(session_dir)
                if f.startswith("cam_") and not f.endswith((".yaml", ".csv"))
            )

        rig_cfg = self.calib_cfg["rigs"]
        self.table.setRowCount(len(cameras))
        for row, label in enumerate(cameras):
            item = QTableWidgetItem(label)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, item)

            rig_combo = QComboBox()
            rig_combo.addItems(self.rig_names + [self.EXCLUDE])
            default_rig = self.calib_cfg["default_rig"]
            for rig, spec in rig_cfg.items():
                if label in (spec.get("cameras") or []):
                    default_rig = rig
            rig_combo.setCurrentText(default_rig)
            self.table.setCellWidget(row, 1, rig_combo)

            model_combo = QComboBox()
            model_combo.addItems(MODELS)
            model_combo.setCurrentText(
                self.calib_cfg["models"].get(label,
                                             self.calib_cfg["default_model"]))
            self.table.setCellWidget(row, 2, model_combo)

            ref_check = QCheckBox()
            ref_check.setChecked(
                label == (rig_cfg.get(default_rig, {}) or {}).get("reference"))
            self.table.setCellWidget(row, 3, ref_check)

        self.run_btn.setEnabled(True)
        self.stage_label.setText(f"ready: {os.path.basename(session_dir)}")

    def _table_assignments(self):
        """-> (rig_of, reference_of, model_of) from the current table state."""
        rig_of, model_of, refs = {}, {}, {}
        for row in range(self.table.rowCount()):
            label = self.table.item(row, 0).text()
            rig = self.table.cellWidget(row, 1).currentText()
            if rig == self.EXCLUDE:
                continue
            rig_of[label] = rig
            model_of[label] = self.table.cellWidget(row, 2).currentText()
            if self.table.cellWidget(row, 3).isChecked():
                refs.setdefault(rig, []).append(label)

        reference_of = {}
        for rig in set(rig_of.values()):
            chosen = refs.get(rig, [])
            if len(chosen) > 1:
                raise ValueError(
                    f"rig '{rig}' has {len(chosen)} reference cameras "
                    f"({chosen}) — check exactly one")
            members = [l for l, r in rig_of.items() if r == rig]
            reference_of[rig] = chosen[0] if chosen else members[0]
        return rig_of, reference_of, model_of

    # ---- run ----------------------------------------------------------------
    def _toggle_run(self):
        if self.worker:
            self.worker.stop()
            self.run_btn.setEnabled(False)
            return
        session_dir = self._session_dir()
        if not session_dir:
            return
        try:
            rig_of, reference_of, model_of = self._table_assignments()
        except ValueError as exc:
            QMessageBox.warning(self, "Rig setup", str(exc))
            return
        if not rig_of:
            QMessageBox.warning(self, "Rig setup", "All cameras are excluded.")
            return

        self.log_view.clear()
        self.worker = CalibrationWorker(
            session_dir, rig_of, reference_of, model_of, self.calib_cfg)
        self.worker.stage.connect(self._on_stage)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self.log_view.appendPlainText)
        self.worker.preview.connect(self._on_preview)
        self.worker.done.connect(self._on_done)
        self.worker.start()
        self.run_btn.setText("Abort calibration")
        self.session_list.setEnabled(False)
        self.table.setEnabled(False)

    def _on_stage(self, title):
        self.stage_label.setText(title)
        self.progress.setValue(0)

    def _on_progress(self, done, total):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)

    def _on_preview(self, qimg):
        self.preview.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def _on_done(self, ok, message):
        self.worker = None
        self.run_btn.setText("Run calibration")
        self.run_btn.setEnabled(True)
        self.session_list.setEnabled(True)
        self.table.setEnabled(True)
        self.stage_label.setText("done ✓" if ok else "failed ✗")
        self.log_view.appendPlainText(message)
        if not ok:
            QMessageBox.warning(self, "Calibration failed",
                                message.splitlines()[-1] if message else "error")

    def shutdown(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(5000)


# ---------------------------------------------------------------------------
# Sync tab
# ---------------------------------------------------------------------------
class SyncTab(QWidget):
    """Generate a synchronized, interpolated anchor_T_phone pose per camera tick.

    For each tick in timestamps.csv it averages the freshly-captured cameras'
    capture times, maps that instant onto the laptop wall clock via meta.yaml's
    clock_anchor, and interpolates the phone_pose.jsonl stream (translation lerp
    + quaternion SLERP against real timestamps, so ARCore's 20-40 Hz jitter is
    handled) to that time, expressed in the ARCore calibration-anchor frame.
    Result -> <session>/anchor_T_phone_sync.jsonl. See pose_sync.py.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.worker = None
        self._build_ui()
        self.refresh_sessions()

    def _build_ui(self):
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        left.addWidget(QLabel("Recorded sessions"))
        self.session_list = QListWidget()
        self.session_list.itemSelectionChanged.connect(self._load_selected)
        left.addWidget(self.session_list, stretch=1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_sessions)
        left.addWidget(refresh)
        left_box = QWidget()
        left_box.setLayout(left)
        left_box.setMaximumWidth(280)
        root.addWidget(left_box)

        right = QVBoxLayout()
        self.info = QLabel("select a session")
        self.info.setWordWrap(True)
        right.addWidget(self.info)

        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("Max interpolation gap (ms)"))
        self.gap_edit = QLineEdit(str(DEFAULT_MAX_GAP_MS))
        self.gap_edit.setMaximumWidth(80)
        self.gap_edit.setToolTip(
            "Ticks whose two bracketing phone samples are farther apart than "
            "this (dropped UDP packets) are still interpolated but flagged "
            "valid=false with reason 'large_gap'.")
        opt_row.addWidget(self.gap_edit)
        opt_row.addStretch(1)
        right.addLayout(opt_row)

        self.run_btn = QPushButton("Generate sync")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)
        right.addWidget(self.run_btn)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        right.addWidget(self.progress)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1000)
        right.addWidget(self.log_view, stretch=1)
        root.addLayout(right, stretch=1)

    def refresh_sessions(self):
        self.session_list.clear()
        base = self.cfg["output_dir"]
        if not os.path.isdir(base):
            return
        for name in sorted(os.listdir(base), reverse=True):
            path = os.path.join(base, name)
            if os.path.isdir(path) and any(
                f.startswith("cam_") for f in os.listdir(path)
            ):
                self.session_list.addItem(name)

    def _session_dir(self):
        items = self.session_list.selectedItems()
        if not items:
            return None
        return os.path.join(self.cfg["output_dir"], items[0].text())

    def _load_selected(self):
        session_dir = self._session_dir()
        if not session_dir or self.worker:
            return
        issues = []
        if not os.path.exists(os.path.join(session_dir, "phone_pose.jsonl")):
            issues.append("no phone_pose.jsonl (no phone was recorded)")
        if not os.path.exists(os.path.join(session_dir, "timestamps.csv")):
            issues.append("no timestamps.csv")
        anchor_ok = False
        meta_path = os.path.join(session_dir, "meta.yaml")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    anchor_ok = bool((yaml.safe_load(f) or {}).get("clock_anchor"))
            except OSError:
                pass
        if not anchor_ok:
            issues.append("meta.yaml has no clock_anchor (recorded before that feature)")

        if issues:
            self.info.setText("⚠ cannot sync this session:\n• " + "\n• ".join(issues))
            self.run_btn.setEnabled(False)
        else:
            done = [n for n in OUTPUT_NAMES
                    if os.path.exists(os.path.join(session_dir, n))]
            self.info.setText(
                f"ready: {os.path.basename(session_dir)}"
                + (f"  —  {', '.join(done)} exists, will overwrite" if done else ""))
            self.run_btn.setEnabled(True)

    def _run(self):
        if self.worker:
            return  # generation is quick; no abort needed
        session_dir = self._session_dir()
        if not session_dir:
            return
        try:
            max_gap_ms = float(self.gap_edit.text())
            if max_gap_ms <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Sync", "Max gap must be a positive number (ms).")
            return
        self.log_view.clear()
        self.progress.setValue(0)
        self.worker = SyncWorker(session_dir, max_gap_ms)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self.log_view.appendPlainText)
        self.worker.done.connect(self._on_done)
        self.worker.start()
        self.run_btn.setText("Generating…")
        self.run_btn.setEnabled(False)
        self.session_list.setEnabled(False)

    def _on_progress(self, done, total):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)

    def _on_done(self, ok, message):
        self.worker = None
        self.run_btn.setText("Generate sync")
        self.run_btn.setEnabled(True)
        self.session_list.setEnabled(True)
        self.log_view.appendPlainText(message)
        if not ok:
            QMessageBox.warning(self, "Sync failed", message)

    def shutdown(self):
        if self.worker:
            self.worker.wait(5000)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self, cfg):
        super().__init__()
        self.setWindowTitle("Multi-Camera Recorder")
        root = QVBoxLayout(self)
        tabs = QTabWidget()
        self.record_tab = RecordTab(cfg)
        self.playback_tab = PlaybackTab(cfg)
        self.calibrate_tab = CalibrateTab(cfg)
        self.sync_tab = SyncTab(cfg)
        self.record_tab.session_saved.connect(self.playback_tab.refresh_sessions)
        self.record_tab.session_saved.connect(self.calibrate_tab.refresh_sessions)
        self.record_tab.session_saved.connect(self.sync_tab.refresh_sessions)
        tabs.addTab(self.record_tab, "Record")
        tabs.addTab(self.playback_tab, "Playback")
        tabs.addTab(self.calibrate_tab, "Calibrate")
        tabs.addTab(self.sync_tab, "Sync")
        root.addWidget(tabs)

    def closeEvent(self, event):
        self.record_tab.shutdown()
        self.playback_tab.shutdown()
        self.calibrate_tab.shutdown()
        self.sync_tab.shutdown()
        event.accept()


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"
    )
    cfg = load_config(cfg_path)
    app = QApplication(sys.argv)
    win = MainWindow(cfg)
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
