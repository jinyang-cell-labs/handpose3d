#!/usr/bin/env python3
"""Lightweight PyQt GUI to view and record an MJPEG IP-camera stream.

Pulls the stream once over HTTP, splits it into JPEG frames, and feeds the
same frames to both the live preview and (optionally) a video file. Designed
for phone IP-webcam apps that expose an endpoint like:

    http://<ip>:<port>/video
"""
import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import requests
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class StreamWorker(QThread):
    """Reads an MJPEG HTTP stream, emits frames, and records to disk.

    Recording is handled inside this thread so the cv2.VideoWriter lives on a
    single thread and the GUI never blocks on disk writes.
    """

    frameReady = pyqtSignal(QImage)
    stats = pyqtSignal(dict)
    connected = pyqtSignal()
    stopped = pyqtSignal()
    error = pyqtSignal(str)
    recordingStarted = pyqtSignal(str)
    recordingStopped = pyqtSignal(str, int)

    def __init__(self, url, parent=None):
        super().__init__(parent)
        self._url = url
        self._running = True
        # Recording control (touched from the GUI thread).
        self._record_requested = False
        self._record_path = ""
        self._record_fps = 0  # 0 == auto (use measured fps)

        self._writer = None
        self._writer_path = ""
        self._recorded_frames = 0
        self._fps_times = deque(maxlen=60)

    # ---- called from the GUI thread -------------------------------------
    def request_record(self, path, fps):
        self._record_path = path
        self._record_fps = fps
        self._record_requested = True

    def request_stop_record(self):
        self._record_requested = False

    def stop(self):
        self._running = False

    # ---- runs on the worker thread --------------------------------------
    def _measured_fps(self):
        if len(self._fps_times) < 2:
            return 0.0
        span = self._fps_times[-1] - self._fps_times[0]
        return (len(self._fps_times) - 1) / span if span > 0 else 0.0

    def _open_writer(self, frame):
        h, w = frame.shape[:2]
        fps = self._record_fps if self._record_fps > 0 else round(self._measured_fps())
        if fps <= 0:
            fps = 30
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(self._record_path, fourcc, float(fps), (w, h))
        if not writer.isOpened():
            self.error.emit(f"could not open writer for {self._record_path}")
            self._record_requested = False
            return
        self._writer = writer
        self._writer_path = self._record_path
        self._recorded_frames = 0
        self.recordingStarted.emit(self._writer_path)

    def _close_writer(self):
        if self._writer is not None:
            self._writer.release()
            self.recordingStopped.emit(self._writer_path, self._recorded_frames)
            self._writer = None
            self._writer_path = ""

    def run(self):
        try:
            resp = requests.get(self._url, stream=True, timeout=(5, 30))
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"connection failed: {exc}")
            self.stopped.emit()
            return

        self.connected.emit()
        buf = b""
        last_stats = 0.0
        try:
            for chunk in resp.iter_content(chunk_size=8192):
                if not self._running:
                    break
                if not chunk:
                    continue
                buf += chunk

                # Extract every complete JPEG (SOI .. EOI) in the buffer.
                while True:
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9", start + 2)
                    if start == -1 or end == -1:
                        break
                    jpg = buf[start : end + 2]
                    buf = buf[end + 2 :]

                    frame = cv2.imdecode(
                        np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is None:
                        continue

                    self._fps_times.append(time.monotonic())

                    # Recording state transitions.
                    if self._record_requested and self._writer is None:
                        self._open_writer(frame)
                    elif not self._record_requested and self._writer is not None:
                        self._close_writer()

                    if self._writer is not None:
                        self._writer.write(frame)
                        self._recorded_frames += 1

                    # Preview (copy detaches the QImage from the numpy buffer).
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    qimg = QImage(
                        rgb.data, w, h, ch * w, QImage.Format_RGB888
                    ).copy()
                    self.frameReady.emit(qimg)

                    now = time.monotonic()
                    if now - last_stats > 0.3:
                        last_stats = now
                        size = (
                            os.path.getsize(self._writer_path)
                            if self._writer is not None
                            and os.path.exists(self._writer_path)
                            else 0
                        )
                        self.stats.emit(
                            {
                                "fps": self._measured_fps(),
                                "width": w,
                                "height": h,
                                "recording": self._writer is not None,
                                "recorded_frames": self._recorded_frames,
                                "file_bytes": size,
                            }
                        )
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"stream error: {exc}")
        finally:
            self._close_writer()
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass
            self.stopped.emit()


class RecorderWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.setWindowTitle("IP Stream Recorder")
        self._build_ui()
        self._set_state(connected=False, recording=False)

    # ---- UI -------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)

        # Connection settings.
        conn_box = QGroupBox("Stream")
        form = QFormLayout(conn_box)
        self.ip_edit = QLineEdit("192.168.0.110")
        self.port_edit = QLineEdit("8080")
        self.path_edit = QLineEdit("/video")
        form.addRow("IP address", self.ip_edit)
        form.addRow("Port", self.port_edit)
        form.addRow("Stream path", self.path_edit)
        self.url_label = QLabel()
        self.url_label.setStyleSheet("color: gray;")
        form.addRow("URL", self.url_label)
        for w in (self.ip_edit, self.port_edit, self.path_edit):
            w.textChanged.connect(self._update_url)
        self._update_url()
        root.addWidget(conn_box)

        # Recording settings.
        rec_box = QGroupBox("Recording")
        rec_form = QFormLayout(rec_box)
        out_row = QHBoxLayout()
        self.dir_edit = QLineEdit(os.path.expanduser("~/recordings"))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_dir)
        out_row.addWidget(self.dir_edit)
        out_row.addWidget(browse)
        out_wrap = QWidget()
        out_wrap.setLayout(out_row)
        rec_form.addRow("Output folder", out_wrap)

        self.prefix_edit = QLineEdit("capture")
        rec_form.addRow("Filename prefix", self.prefix_edit)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(0, 120)
        self.fps_spin.setValue(0)
        self.fps_spin.setSpecialValueText("auto (measured)")
        rec_form.addRow("Recording FPS", self.fps_spin)
        root.addWidget(rec_box)

        # Buttons.
        btn_row = QHBoxLayout()
        self.connect_btn = QPushButton("Start receiving")
        self.connect_btn.clicked.connect(self._toggle_connect)
        self.record_btn = QPushButton("Start recording")
        self.record_btn.clicked.connect(self._toggle_record)
        btn_row.addWidget(self.connect_btn)
        btn_row.addWidget(self.record_btn)
        root.addLayout(btn_row)

        # Preview.
        self.view = QLabel("no signal")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(640, 360)
        self.view.setStyleSheet("background:#111; color:#888;")
        root.addWidget(self.view, stretch=1)

        # Status line.
        self.status = QLabel("idle")
        self.status.setStyleSheet("color:#444;")
        root.addWidget(self.status)

    def _update_url(self):
        path = self.path_edit.text()
        if not path.startswith("/"):
            path = "/" + path
        self._url = f"http://{self.ip_edit.text()}:{self.port_edit.text()}{path}"
        self.url_label.setText(self._url)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Choose output folder", self.dir_edit.text()
        )
        if d:
            self.dir_edit.setText(d)

    def _set_state(self, connected, recording):
        self._connected = connected
        self._recording = recording
        self.connect_btn.setText("Stop receiving" if connected else "Start receiving")
        self.record_btn.setText("Stop recording" if recording else "Start recording")
        self.record_btn.setEnabled(connected)
        for w in (self.ip_edit, self.port_edit, self.path_edit):
            w.setEnabled(not connected)

    # ---- connection -----------------------------------------------------
    def _toggle_connect(self):
        if self._connected:
            if self.worker:
                self.worker.stop()
            self.connect_btn.setEnabled(False)
            self.status.setText("disconnecting…")
        else:
            self._update_url()
            self.worker = StreamWorker(self._url)
            self.worker.frameReady.connect(self._on_frame)
            self.worker.stats.connect(self._on_stats)
            self.worker.connected.connect(self._on_connected)
            self.worker.stopped.connect(self._on_stopped)
            self.worker.error.connect(self._on_error)
            self.worker.recordingStarted.connect(
                lambda p: self.status.setText(f"recording → {p}")
            )
            self.worker.recordingStopped.connect(
                lambda p, n: self.status.setText(f"saved {n} frames → {p}")
            )
            self.worker.start()
            self.connect_btn.setEnabled(False)
            self.status.setText(f"connecting to {self._url} …")

    def _on_connected(self):
        self.connect_btn.setEnabled(True)
        self._set_state(connected=True, recording=False)
        self.status.setText("receiving")

    def _on_stopped(self):
        self.connect_btn.setEnabled(True)
        self._set_state(connected=False, recording=False)
        self.view.setText("no signal")
        self.view.setPixmap(QPixmap())
        self.status.setText("stopped")
        self.worker = None

    def _on_error(self, msg):
        self.status.setText(f"⚠ {msg}")
        self.status.setStyleSheet("color:#b00;")

    # ---- recording ------------------------------------------------------
    def _toggle_record(self):
        if not self.worker:
            return
        if self._recording:
            self.worker.request_stop_record()
            self._set_state(connected=self._connected, recording=False)
        else:
            out_dir = self.dir_edit.text()
            try:
                os.makedirs(out_dir, exist_ok=True)
            except OSError as exc:
                self._on_error(f"cannot create folder: {exc}")
                return
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = self.prefix_edit.text().strip() or "capture"
            path = os.path.join(out_dir, f"{prefix}_{ts}.avi")
            self.worker.request_record(path, self.fps_spin.value())
            self._set_state(connected=self._connected, recording=True)

    # ---- frame / stats slots -------------------------------------------
    def _on_frame(self, qimg):
        pix = QPixmap.fromImage(qimg).scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.view.setPixmap(pix)

    def _on_stats(self, s):
        mb = s["file_bytes"] / (1024 * 1024)
        rec = (
            f" | REC {s['recorded_frames']} frames, {mb:.1f} MB"
            if s["recording"]
            else ""
        )
        self.status.setStyleSheet("color:#444;")
        self.status.setText(
            f"{s['width']}x{s['height']} @ {s['fps']:.1f} fps{rec}"
        )

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = RecorderWindow()
    win.resize(720, 640)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
