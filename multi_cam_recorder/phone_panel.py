#!/usr/bin/env python3
"""Phone ARCore pose control + recording for the multi-cam Record tab.

Embeds the ``phone_tracker`` reference client (stdlib ``PhoneControl`` plus the
JSONL record helpers) into a Qt widget so the recorder can drive the phone
app's 6DoF pose stream: connect + NTP clock-sync, start/stop the stream, remote
Calibrate / Waypoint / Clear, a live pose readout, and — driven by the main
Record button — logging the pose stream to ``<session>/phone_pose.jsonl``
alongside the camera videos (same JSONL layout and laptop-clock timestamps as
``record_streams.py``).

The phone and the head-mounted camera set are rigidly connected, so the phone
pose is recorded frame-synchronized with the cameras (start/stop together) as a
calibration reference. Only the ARCore pose stream is handled here; the phone's
IMU pipeline is intentionally left out (unused by the current flow).

Threading: every blocking control RPC (each ~0.5 s of UDP round-trips) runs on
a single serialized background worker thread, so the GUI never freezes and the
phone's one control socket is never used concurrently. The UDP pose stream is
received on its own thread. Widgets are only ever touched from the GUI thread —
background threads mutate plain state (under locks) and emit ``log_msg``; a
QTimer polls that state to refresh the display, mirroring how phone_gui.py polls
``/state``.
"""
import json
import os
import queue
import socket
import sys
import threading
import time

from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

# The phone client lives in the phone_tracker reference project next to this
# file; add it to the path so we reuse its stdlib control + record helpers
# rather than duplicating the packet format / clock-sync logic.
_PHONE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phone_tracker")
if _PHONE_DIR not in sys.path:
    sys.path.insert(0, _PHONE_DIR)

from phone_control import ControlError, PhoneControl  # noqa: E402
from record_streams import pose_record, sync_record  # noqa: E402

DEFAULT_PHONE = {"ip": "192.168.0.110", "pose_port": 9870, "enabled": True}
SYNC_PINGS = 50
POSE_JSONL = "phone_pose.jsonl"


class PoseReceiver:
    """Receives the phone's pose UDP stream on a background thread.

    A pose-only slice of phone_gui.StreamRx: each datagram is a complete pose
    state (newest wins), so there is no reassembly — we keep the latest packet
    for the live readout and hand every packet to ``on_packet`` for logging.
    """

    def __init__(self, on_packet):
        self.on_packet = on_packet
        self.sock = None
        self.thread = None
        self.stop_ev = threading.Event()
        self.port = None
        self.count = 0
        self.last = None
        self.last_mono = 0.0

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, port):
        self.stop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        sock.bind(("0.0.0.0", port))
        self.sock, self.port, self.count = sock, port, 0
        self.stop_ev = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_ev.is_set():
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed
            t_recv = time.time_ns()
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            self.count += 1
            self.last = msg
            self.last_mono = time.monotonic()
            self.on_packet(msg, t_recv)

    def stop(self):
        self.stop_ev.set()
        if self.sock:
            self.sock.close()
            self.sock = None
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None


class PhonePanel(QGroupBox):
    """Phone pose control + recording, embedded in the Record tab.

    Public API used by RecordTab:
        begin_recording(session_dir)  — start logging poses with the cameras
        end_recording()              — stop logging, record clock drift
        set_locked(bool)             — lock connection controls while cameras record
        shutdown()                   — stop threads + close sockets on app exit
    """

    log_msg = pyqtSignal(str)

    def __init__(self, cfg, parent=None):
        super().__init__("Phone pose (ARCore)", parent)
        # config.yaml is the single source of truth for the prefilled IP/port.
        self.cfg = dict(DEFAULT_PHONE)
        user = cfg.get("phone")
        if isinstance(user, dict):
            self.cfg.update({k: v for k, v in user.items() if v is not None})

        # -- control-plane state (touched only by the worker thread) --------
        self.ctrl = None
        self.phone_ip = None
        self.sync = None
        self._screen_locked = False  # believed state of the phone's screen lock

        # -- data plane -----------------------------------------------------
        self.receiver = PoseReceiver(self._on_pose)
        self._rec_started_stream = False

        # -- recording state (worker opens/closes the file; receiver writes) -
        self.rec_lock = threading.Lock()
        self.rec_file = None
        self.rec_path = None
        self.rec_offset = 0
        self.rec_sync_start = None
        self.rec_poses = 0

        # -- serialized background worker for all blocking control RPCs -----
        self._tasks = queue.Queue()
        self._busy = False
        self._locked = False
        self._worker = threading.Thread(target=self._task_loop, daemon=True)
        self._worker.start()

        self._prev_count = 0
        self._prev_t = time.monotonic()

        self._build_ui()
        self.log_msg.connect(self._append_log)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(400)

    # -- UI ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)

        row1 = QHBoxLayout()
        self.dot = QLabel("●")
        self.dot.setStyleSheet("color:#999;")
        row1.addWidget(self.dot)
        row1.addWidget(QLabel("IP"))
        self.ip_edit = QLineEdit(str(self.cfg["ip"]))
        self.ip_edit.setMaximumWidth(150)
        row1.addWidget(self.ip_edit)
        self.connect_btn = QPushButton("Connect + sync")
        self.connect_btn.clicked.connect(self._on_connect)
        row1.addWidget(self.connect_btn)
        self.sync_btn = QPushButton("Re-sync")
        self.sync_btn.clicked.connect(lambda: self._submit(self._do_sync))
        row1.addWidget(self.sync_btn)
        self.sync_info = QLabel("not connected")
        self.sync_info.setStyleSheet("color:#888;")
        row1.addWidget(self.sync_info, stretch=1)
        root.addLayout(row1)

        row2 = QHBoxLayout()
        self.pose_btn = QPushButton("Start pose")
        self.pose_btn.clicked.connect(self._on_toggle_pose)
        row2.addWidget(self.pose_btn)
        self.calib_btn = QPushButton("Calibrate")
        self.calib_btn.clicked.connect(lambda: self._cmd("calibrate"))
        row2.addWidget(self.calib_btn)
        self.wp_btn = QPushButton("Waypoint")
        self.wp_btn.clicked.connect(lambda: self._cmd("waypoint"))
        row2.addWidget(self.wp_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(lambda: self._cmd("clear_waypoints"))
        row2.addWidget(self.clear_btn)
        self.lock_btn = QPushButton("Lock screen")
        self.lock_btn.setToolTip(
            "Disable the phone's on-screen controls so a stray touch can't "
            "trigger anything while it is moved as a controller (the control "
            "channel and streams keep running).")
        self.lock_btn.clicked.connect(self._on_toggle_lock)
        row2.addWidget(self.lock_btn)
        self.include_chk = QCheckBox("Record with cameras")
        self.include_chk.setChecked(bool(self.cfg.get("enabled", True)))
        row2.addWidget(self.include_chk)
        row2.addStretch(1)
        root.addLayout(row2)

        self.pose_live = QLabel("—")
        self.pose_live.setStyleSheet("font-family:monospace; color:#2a7a3a;")
        root.addWidget(self.pose_live)

        self.status = QLabel("")
        self.status.setStyleSheet("color:#888;")
        root.addWidget(self.status)

    # -- background worker ---------------------------------------------------
    def _submit(self, fn):
        """Queue a blocking control operation for the worker thread."""
        self._busy = True
        self._tasks.put(fn)

    def _task_loop(self):
        while True:
            fn = self._tasks.get()
            if fn is None:
                return  # shutdown sentinel
            try:
                fn()
            except ControlError as e:
                self.log_msg.emit(str(e))
            except Exception as e:  # keep the worker alive on any error
                self.log_msg.emit(f"error: {e}")
            finally:
                if self._tasks.empty():
                    self._busy = False

    def _append_log(self, msg):
        self.status.setText(msg)

    # -- actions (button handlers dispatch to worker tasks) ------------------
    def _on_connect(self):
        ip = self.ip_edit.text().strip()
        if not ip:
            self.log_msg.emit("enter the phone IP")
            return
        self._submit(lambda: self._do_connect(ip))

    def _do_connect(self, ip):
        if self.ctrl:
            self.ctrl.close()
        self.ctrl = PhoneControl(ip)
        self.phone_ip = ip
        self.cfg["ip"] = ip
        self.log_msg.emit(f"connecting + syncing with {ip} ...")
        self.sync = self.ctrl.sync_clock(SYNC_PINGS)
        self.log_msg.emit(
            f"connected; clock offset {self.sync.offset_ns / 1e9:+.6f} s, "
            f"best RTT {self.sync.rtt_ns / 1e6:.2f} ms")

    def _do_sync(self):
        if not self.ctrl:
            raise ControlError("connect to the phone first")
        self.sync = self.ctrl.sync_clock(SYNC_PINGS)
        self.log_msg.emit(
            f"re-synced; offset {self.sync.offset_ns / 1e9:+.6f} s, "
            f"best RTT {self.sync.rtt_ns / 1e6:.2f} ms")

    def _on_toggle_pose(self):
        if self.receiver.running:
            self._submit(self._do_stop_pose_preview)
        else:
            port = int(self.cfg["pose_port"])
            self._submit(lambda: self._do_start_pose_preview(port))

    def _do_start_pose_preview(self, port):
        if not self.ctrl:
            raise ControlError("connect to the phone first")
        self.receiver.start(port)  # bind before the phone starts sending
        try:
            self.ctrl.start_pose(port)
        except Exception:
            self.receiver.stop()
            raise
        self.log_msg.emit(f"pose stream started -> :{port}")

    def _do_stop_pose_preview(self):
        with self.rec_lock:
            recording = self.rec_file is not None
        if recording:
            self.log_msg.emit("stop recording before stopping the stream")
            return
        try:
            if self.ctrl:
                self.ctrl.stop_pose()
        finally:
            self.receiver.stop()
        self.log_msg.emit("pose stream stopped")

    def _cmd(self, name):
        def run():
            if not self.ctrl:
                raise ControlError("connect to the phone first")
            getattr(self.ctrl, name)()
            self.log_msg.emit(name.replace("_", " ") + " ok")
        self._submit(run)

    def _on_toggle_lock(self):
        self._submit(self._do_toggle_lock)

    def _do_toggle_lock(self):
        if not self.ctrl:
            raise ControlError("connect to the phone first")
        target = not self._screen_locked
        (self.ctrl.lock if target else self.ctrl.unlock)()
        self._screen_locked = target  # only flip after the ack succeeds
        self.log_msg.emit("screen locked" if target else "screen unlocked")

    # -- recording (driven by the main Record button) -----------------------
    def begin_recording(self, session_dir):
        """Start logging poses into ``session_dir`` alongside the cameras.

        No-op (with a log line) when 'Record with cameras' is unchecked or the
        phone is not connected — the cameras still record on their own.
        """
        if not self.include_chk.isChecked():
            return
        if not self.ctrl:
            self.log_msg.emit("phone not connected — recording cameras only")
            return
        self._submit(lambda: self._do_begin_recording(session_dir))

    def _do_begin_recording(self, session_dir):
        with self.rec_lock:
            if self.rec_file is not None:
                raise ControlError("phone already recording")
        # Fresh clock sync so the log's laptop-clock timestamps are accurate.
        self.sync = self.ctrl.sync_clock(SYNC_PINGS)
        # Ensure the pose stream is up. Remember whether we started it so that
        # end-of-recording only stops streams the recorder itself started (a
        # stream left running for live preview keeps running afterwards).
        port = int(self.cfg["pose_port"])
        started_here = False
        if not self.receiver.running:
            self.receiver.start(port)
            try:
                self.ctrl.start_pose(port)
            except Exception:
                self.receiver.stop()
                raise
            started_here = True
        try:
            os.makedirs(session_dir, exist_ok=True)
            path = os.path.join(session_dir, POSE_JSONL)
            f = open(path, "w")
            f.write(json.dumps(sync_record(
                self.sync, "start",
                {"phone_ip": self.phone_ip, "streams": ["pose"]})) + "\n")
        except Exception:  # don't leave a stream we started running orphaned
            if started_here:
                self.receiver.stop()
                try:
                    self.ctrl.stop_pose()
                except Exception:
                    pass
            raise
        self._rec_started_stream = started_here
        with self.rec_lock:
            self.rec_file, self.rec_path = f, path
            self.rec_offset = self.sync.offset_ns
            self.rec_sync_start = self.sync
            self.rec_poses = 0
        self.log_msg.emit(f"recording phone pose -> {os.path.basename(path)}")

    def end_recording(self):
        # Always submit; the task no-ops if we were not recording.
        self._submit(self._do_end_recording)

    def _do_end_recording(self):
        with self.rec_lock:
            f, self.rec_file = self.rec_file, None
            n, path = self.rec_poses, self.rec_path
            sync_start, offset = self.rec_sync_start, self.rec_offset
        if not f:
            return  # was not recording
        # Stop the stream if the recorder started it (leave a preview running).
        if self._rec_started_stream:
            try:
                if self.ctrl:
                    self.ctrl.stop_pose()
            except ControlError as e:
                self.log_msg.emit(f"warning: could not stop stream: {e}")
            self.receiver.stop()
            self._rec_started_stream = False
        # End-of-recording re-sync captures clock drift over the session; the
        # file is always closed, even if the sync fails unexpectedly.
        try:
            sync2 = self.ctrl.sync_clock(SYNC_PINGS) if self.ctrl else None
            if sync2 is not None:
                elapsed_ns = sync2.t_wall_ns - sync_start.t_wall_ns
                drift_ppm = ((sync2.offset_ns - offset) / elapsed_ns * 1e6
                             if elapsed_ns else 0.0)
                f.write(json.dumps(sync_record(
                    sync2, "end", {"drift_ppm": round(drift_ppm, 3)})) + "\n")
                self.sync = sync2
                self.log_msg.emit(
                    f"clock drift {(sync2.offset_ns - offset) / 1e6:+.3f} ms "
                    f"({drift_ppm:+.2f} ppm)")
        except ControlError as e:
            self.log_msg.emit(f"warning: end-of-recording sync failed: {e}")
        finally:
            f.close()
        self.log_msg.emit(f"saved {n} poses -> {os.path.basename(path)}")

    def _on_pose(self, msg, t_recv):
        """Receiver-thread callback: log the pose if a recording is open."""
        with self.rec_lock:
            if self.rec_file is not None:
                self.rec_file.write(
                    json.dumps(pose_record(msg, t_recv, self.rec_offset)) + "\n")
                self.rec_poses += 1

    def set_locked(self, locked):
        """Lock connection controls (called while the cameras are recording)."""
        self._locked = locked

    # -- periodic UI refresh (GUI thread) ------------------------------------
    def _refresh(self):
        connected = self.ctrl is not None
        running = self.receiver.running
        with self.rec_lock:
            recording = self.rec_file is not None
            rec_poses = self.rec_poses

        self.dot.setStyleSheet("color:#2a7a3a;" if connected else "color:#999;")

        if self.sync is not None:
            age = (time.time_ns() - self.sync.t_wall_ns) / 1e9
            self.sync_info.setText(
                f"offset {self.sync.offset_ns / 1e9:+.6f} s · "
                f"RTT {self.sync.rtt_ns / 1e6:.2f} ms · {age:.0f}s ago")
        else:
            self.sync_info.setText("connected, not synced" if connected
                                   else "not connected")

        self.pose_btn.setText("Stop pose" if running else "Start pose")
        now = time.monotonic()
        dt = max(now - self._prev_t, 1e-3)
        hz = (self.receiver.count - self._prev_count) / dt
        self._prev_count, self._prev_t = self.receiver.count, now
        p = self.receiver.last
        if not running:
            self.pose_live.setText("—")
        elif p is None:
            self.pose_live.setText("waiting for packets…")
        else:
            stale = "  [stale]" if now - self.receiver.last_mono > 1.0 else ""
            anchor = " (anchor ok)" if p.get("apx") is not None else ""
            self.pose_live.setText(
                f"{p.get('state', '?')}  {hz:.1f} Hz  "
                f"p=({p.get('px', 0):+.2f} {p.get('py', 0):+.2f} "
                f"{p.get('pz', 0):+.2f})  calib #{p.get('calib', 0)}{anchor}  "
                f"wp:{len(p.get('wps', []))}{stale}")

        lock = self._busy or recording or self._locked
        self.connect_btn.setEnabled(not lock)
        self.sync_btn.setEnabled(connected and not self._busy)
        self.pose_btn.setEnabled(connected and not lock)
        self.calib_btn.setEnabled(connected and not self._busy)
        self.wp_btn.setEnabled(connected and not self._busy)
        self.clear_btn.setEnabled(connected and not self._busy)
        self.lock_btn.setText("Unlock screen" if self._screen_locked
                              else "Lock screen")
        self.lock_btn.setEnabled(connected and not self._busy)
        self.include_chk.setEnabled(not (recording or self._locked))

        if recording:
            self.setTitle(f"Phone pose (ARCore)  —  ● REC {rec_poses} poses")
        elif connected:
            self.setTitle("Phone pose (ARCore)  —  connected")
        else:
            self.setTitle("Phone pose (ARCore)")

    # -- shutdown ------------------------------------------------------------
    def shutdown(self):
        self.timer.stop()
        with self.rec_lock:
            f, self.rec_file = self.rec_file, None
        if f:
            try:
                f.close()
            except Exception:
                pass
        self.receiver.stop()
        self._tasks.put(None)  # stop the worker thread
        self._worker.join(timeout=1.0)
        if self.ctrl:
            try:
                self.ctrl.close()
            except Exception:
                pass
