#!/usr/bin/env python3
"""Single-camera MediaPipe hand-landmark node (body_cam_teleop).

Opens ONE V4L camera and runs MediaPipe HandLandmarker directly on the
captured frame in-process — the image never crosses DDS unless
``enable_reprojection`` is set. Publishes only the compact
``handpose3d_msgs/HandLandmarks`` message (21 raw-image pixels + MediaPipe's
hand-local metric model per hand); undistortion of the landmark points is
done downstream in hand_pose_node (points only — no full-frame remap).

With ``enable_reprojection: true`` (reprojection overlay) the raw frame is
additionally published on ``image_topic`` for hand_pose_node.

With ``async_inference: true`` (default) MediaPipe runs in LIVE_STREAM mode:
``detect_async`` returns immediately and the result arrives on MediaPipe's own
thread, so inference (the slowest stage) overlaps the next frame's capture
instead of serializing with it. When inference is slower than the camera,
MediaPipe's flow limiter drops frames (reported as the ``dropped`` counter;
``capture_fps`` - ``fps`` is the drop rate). ``false`` restores the
synchronous VIDEO mode (capture -> inference -> publish in one loop).

With ``enable_perf: true`` (default) per-stage wall times (capture, convert,
mediapipe, publish) are accumulated and published once a second as a JSON
std_msgs/String on ``perf_topic`` — recorded to CSV by perf_monitor_node.
In async mode ``mediapipe_ms`` is submit-to-result latency (queue +
inference) and ``tick_total_ms`` covers only the capture side of the loop.

Diagnostics: this node owns the first two gates a hand must pass before it can
reach hand_pose_node (MediaPipe detection, then the handedness label/score
filters). Each gate has its own opt-in log flag — ``log_capture``,
``log_detection``, ``log_handedness``, ``log_publish`` — and every reject line
names the gate, the measured value and the threshold that rejected it, so a
dropout traces back to one parameter. ``log_gate_summary`` adds a periodic
funnel count (how many hands each gate ate). All flags are runtime-settable
(``ros2 param set <node> log_handedness true``) so a live pipeline can be
instrumented without a restart.
"""
import json
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from handpose3d_msgs.msg import Hand, HandLandmarks

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

N_LANDMARKS = 21


class HandLandmarksNode(Node):
    def __init__(self):
        super().__init__("hand_landmarks_node")

        # --- camera ---------------------------------------------------------
        self.camera_name = self.declare_parameter("camera_name", "camera0").value
        camera_device = str(self.declare_parameter("camera_device", "0").value)
        self.frame_rate = float(self.declare_parameter("frame_rate", 30.0).value)
        self.capture_width = int(self.declare_parameter("capture_width", 1280).value)
        self.capture_height = int(self.declare_parameter("capture_height", 720).value)
        self.fourcc = str(self.declare_parameter("fourcc", "MJPG").value)

        # --- MediaPipe ------------------------------------------------------
        model_path = self.declare_parameter("model_path", "").value
        self.num_hands = int(self.declare_parameter("num_hands", 2).value)
        self.min_hand_detection_confidence = float(
            self.declare_parameter("min_hand_detection_confidence", 0.5).value)
        self.min_hand_presence_confidence = float(
            self.declare_parameter("min_hand_presence_confidence", 0.5).value)
        self.min_tracking_confidence = float(
            self.declare_parameter("min_tracking_confidence", 0.5).value)
        # "cpu" or "gpu". The GPU delegate runs on OpenGL ES (typically the
        # iGPU); its value is offloading a saturated CPU. Falls back to CPU.
        self.delegate = str(self.declare_parameter("delegate", "gpu").value).lower()
        # LIVE_STREAM (async) vs VIDEO (sync) inference — see module docstring.
        self.async_inference = bool(
            self.declare_parameter("async_inference", True).value)
        # left_only | right_only | left_and_right
        self.hand_filter_mode = str(
            self.declare_parameter("hand_filter_mode", "left_and_right").value).lower()
        self.min_handedness_confidence = float(
            self.declare_parameter("min_handedness_confidence", 0.0).value)

        # --- topics ---------------------------------------------------------
        self.landmarks_topic = self.declare_parameter(
            "landmarks_topic", "body_cam_teleop/landmarks").value
        self.enable_reprojection = bool(
            self.declare_parameter("enable_reprojection", False).value)
        self.publish_image = self.enable_reprojection
        self.image_topic = self.declare_parameter(
            "image_topic", "body_cam_teleop/image_raw").value

        # --- diagnostics (per-stage, opt-in, runtime-settable) ---------------
        # One flag per stage of the detection funnel; see module docstring.
        # Detail lines are throttled per (stage, reason, hand) rather than per
        # call site, so a Left reject never masks a Right one.
        self.log_capture = bool(self.declare_parameter("log_capture", False).value)
        self.log_detection = bool(self.declare_parameter("log_detection", False).value)
        self.log_handedness = bool(self.declare_parameter("log_handedness", False).value)
        self.log_publish = bool(self.declare_parameter("log_publish", False).value)
        self.log_gate_summary = bool(
            self.declare_parameter("log_gate_summary", False).value)
        self.log_throttle_sec = float(
            self.declare_parameter("log_throttle_sec", 2.0).value)
        self.log_summary_period_sec = float(
            self.declare_parameter("log_summary_period_sec", 5.0).value)
        self._diag_last = {}
        self._gate_keys = (
            "ticks", "no_frame", "results", "raw_hands", "rej_label",
            "rej_handedness_score", "rej_no_world", "passed_hands",
            "frames_with_hand")
        self._gate_counts = dict.fromkeys(self._gate_keys, 0)

        allowed_by_mode = {
            "left_only": {"Left"},
            "right_only": {"Right"},
            "left_and_right": {"Left", "Right"},
        }
        if self.hand_filter_mode not in allowed_by_mode:
            raise ValueError(
                f"hand_filter_mode must be one of {sorted(allowed_by_mode)}, "
                f"got '{self.hand_filter_mode}'")
        self.allowed_labels = allowed_by_mode[self.hand_filter_mode]

        if not model_path:
            model_path = os.path.join(
                get_package_share_directory("body_cam_teleop"),
                "models", "hand_landmarker.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"hand landmark model not found: {model_path}")
        # Shared between the executor thread (_tick) and MediaPipe's result
        # thread (_on_result): the in-flight frame contexts and the perf
        # accumulators. ts_ms -> (ros stamp, width, height, t_submit).
        self._pending = {}
        self._lock = threading.Lock()
        self.detector = self._make_landmarker(model_path)
        self._last_ts_ms = -1

        self.cap = self._open_capture(camera_device)

        self.landmarks_pub = self.create_publisher(
            HandLandmarks, self.landmarks_topic, 5)
        self.image_pub = None
        if self.publish_image:
            self.image_pub = self.create_publisher(
                Image, self.image_topic, qos_profile_sensor_data)

        # --- perf instrumentation (see module docstring) ----------------------
        self.enable_perf = bool(self.declare_parameter("enable_perf", True).value)
        self.perf_pub = None
        if self.enable_perf:
            perf_topic = self.declare_parameter(
                "perf_topic", "body_cam_teleop/perf").value
            self.perf_pub = self.create_publisher(String, perf_topic, 5)
            self._perf_stages = {}  # stage -> [n, sum_ms, max_ms]
            self._perf_counts = {
                "frames": 0, "no_frame": 0, "hands": 0,
                "captures": 0, "dropped": 0}
            self._perf_window_start = time.monotonic()
            self.perf_timer = self.create_timer(1.0, self._publish_perf)

        # The funnel timer runs regardless of log_gate_summary (it drains the
        # counters either way) so the flag can be flipped at runtime.
        self._gate_window_start = time.monotonic()
        self.gate_timer = self.create_timer(
            max(self.log_summary_period_sec, 0.1), self._publish_gate_summary)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.timer = self.create_timer(1.0 / self.frame_rate, self._tick)
        self.get_logger().info(
            f"hand_landmarks_node up: {self.camera_name} (dev {camera_device}) "
            f"@ {self.frame_rate:.0f} fps, delegate={self.delegate}, "
            f"async={'on' if self.async_inference else 'off'}, "
            f"filter={self.hand_filter_mode}, "
            f"image publishing={'on' if self.publish_image else 'off'} "
            f"-> {self.landmarks_topic}")
        enabled_logs = [
            name for name in ("capture", "detection", "handedness", "publish",
                              "gate_summary")
            if getattr(self, f"log_{name}")]
        self.get_logger().info(
            f"stage logging: {', '.join(enabled_logs) if enabled_logs else 'none'} "
            f"(throttle {self.log_throttle_sec:.1f}s; toggle with "
            "`ros2 param set <node> log_<stage> true`)")

    # ------------------------------------------------------------------ setup
    def _open_capture(self, source):
        dev = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open camera device '{source}'")
        # Order matters for V4L2: fourcc, then resolution.
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        # Keep at most one queued frame: with V4L's default 4-frame buffer a
        # loop that falls behind the camera reads stale frames and stamps
        # them (below) as if they were fresh — silently inflating the real
        # capture->pose latency beyond what the perf stages report.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (w, h) != (self.capture_width, self.capture_height):
            self.get_logger().warn(
                f"requested {self.capture_width}x{self.capture_height}, "
                f"got {w}x{h} (V4L2 fell back to a supported mode)")
        return cap

    def _make_landmarker(self, model_path):
        def build(delegate):
            delegate_enum = (mp_python.BaseOptions.Delegate.GPU
                             if delegate == "gpu"
                             else mp_python.BaseOptions.Delegate.CPU)
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=model_path, delegate=delegate_enum),
                running_mode=(mp_vision.RunningMode.LIVE_STREAM
                              if self.async_inference
                              else mp_vision.RunningMode.VIDEO),
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.min_hand_detection_confidence,
                min_hand_presence_confidence=self.min_hand_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                result_callback=self._on_result if self.async_inference else None,
            )
            return mp_vision.HandLandmarker.create_from_options(options)

        try:
            return build(self.delegate)
        except Exception as exc:  # noqa: BLE001
            if self.delegate == "gpu":
                self.get_logger().warn(
                    f"GPU delegate unavailable ({exc}); falling back to CPU")
                return build("cpu")
            raise

    # ----------------------------------------------------------- diagnostics
    def _on_set_parameters(self, params):
        """Let the log_* flags be toggled on a running pipeline."""
        result = SetParametersResult(successful=True)
        for p in params:
            if p.name in ("log_capture", "log_detection", "log_handedness",
                          "log_publish", "log_gate_summary"):
                setattr(self, p.name, bool(p.value))
            elif p.name == "log_throttle_sec":
                if p.value < 0.0:
                    return SetParametersResult(
                        successful=False, reason="log_throttle_sec must be >= 0")
                self.log_throttle_sec = float(p.value)
        return result

    def _diag(self, key, msg, level="info"):
        """Throttled diagnostic line, keyed by `key` instead of by call site.

        rclpy's throttle_duration_sec keys on the call site, which would let one
        reject reason (or one hand) mask another emitted from the same line.
        """
        now = time.monotonic()
        if now - self._diag_last.get(key, -1e9) < self.log_throttle_sec:
            return
        self._diag_last[key] = now
        # rclpy caches logger state per call site and rejects a severity change
        # at the same site ("Logger severity cannot be changed between calls"),
        # so each severity needs its own line here.
        if level == "warn":
            self.get_logger().warn(msg)
        else:
            self.get_logger().info(msg)

    def _gate(self, key, inc=1):
        with self._lock:
            self._gate_counts[key] += inc

    def _publish_gate_summary(self):
        """Drain the funnel counters; log them when log_gate_summary is on."""
        now = time.monotonic()
        window = max(now - self._gate_window_start, 1e-6)
        self._gate_window_start = now
        with self._lock:
            c = self._gate_counts
            self._gate_counts = dict.fromkeys(self._gate_keys, 0)
        if not self.log_gate_summary:
            return
        self.get_logger().info(
            f"[gate funnel {window:.1f}s] ticks={c['ticks']} "
            f"(no_frame={c['no_frame']}) -> results={c['results']} "
            f"-> gate1 detected={c['raw_hands']} hand(s) "
            f"-> gate2 rejected: label={c['rej_label']} "
            f"handedness_score={c['rej_handedness_score']} "
            f"no_world_landmarks={c['rej_no_world']} "
            f"-> published={c['passed_hands']} hand(s) in "
            f"{c['frames_with_hand']} frame(s)")

    # ------------------------------------------------------------------ perf
    def _perf_add(self, stage, dt_sec):
        """Accumulate one wall-time sample [s] for a pipeline stage."""
        if self.perf_pub is None:
            return
        ms = dt_sec * 1e3
        with self._lock:
            acc = self._perf_stages.setdefault(stage, [0, 0.0, 0.0])
            acc[0] += 1
            acc[1] += ms
            acc[2] = max(acc[2], ms)

    def _perf_count(self, key, inc=1):
        if self.perf_pub is None:
            return
        with self._lock:
            self._perf_counts[key] += inc

    def _publish_perf(self):
        now = time.monotonic()
        window = max(now - self._perf_window_start, 1e-6)
        self._perf_window_start = now
        with self._lock:
            stages = self._perf_stages
            counts = self._perf_counts
            self._perf_stages = {}
            self._perf_counts = dict.fromkeys(counts, 0)
        frames = counts["frames"]
        payload = {
            "node": self.get_fully_qualified_name(),
            "window_sec": round(window, 3),
            "stages": {
                stage: {
                    "n": n,
                    "mean_ms": round(total / n, 3) if n else 0.0,
                    "max_ms": round(peak, 3),
                }
                for stage, (n, total, peak) in stages.items()
            },
            "counters": {
                "fps": round(frames / window, 2),
                "capture_fps": round(counts["captures"] / window, 2),
                "dropped": counts["dropped"],
                "target_fps": self.frame_rate,
                "hands_per_frame": round(
                    counts["hands"] / frames, 2) if frames else 0.0,
                "no_frame": counts["no_frame"],
            },
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.perf_pub.publish(msg)

    # --------------------------------------------------------------- capture
    def _tick(self):
        t_tick = time.perf_counter()
        ret, frame = self.cap.read()
        t_capture = time.perf_counter()
        stamp = self.get_clock().now().to_msg()
        self._gate("ticks")
        if not ret or frame is None:
            self.get_logger().warn("no frame from camera", throttle_duration_sec=5.0)
            self._perf_count("no_frame")
            self._gate("no_frame")
            return
        self._perf_add("capture_ms", t_capture - t_tick)
        self._perf_count("captures")
        h, w = frame.shape[:2]
        if self.log_capture:
            self._diag(
                "capture",
                f"[stage capture] frame {w}x{h} in "
                f"{(t_capture - t_tick) * 1e3:.1f} ms")

        # The raw frame (reprojection input) doesn't depend on the detection
        # result — ship it before inference, not after.
        if self.image_pub is not None:
            img = Image()
            img.header.stamp = stamp
            img.header.frame_id = self.camera_name
            img.height = h
            img.width = w
            img.encoding = "bgr8"
            img.is_bigendian = 0
            img.step = w * 3
            img.data = np.ascontiguousarray(frame).tobytes()
            self.image_pub.publish(img)
            self._perf_add("publish_image_ms", time.perf_counter() - t_capture)

        t_pre = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        t_convert = time.perf_counter()
        self._perf_add("convert_ms", t_convert - t_pre)

        # VIDEO/LIVE_STREAM modes need strictly increasing timestamps.
        ts_ms = stamp.sec * 1000 + stamp.nanosec // 1_000_000
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        if self.async_inference:
            with self._lock:
                self._pending[ts_ms] = (stamp, w, h, t_convert)
                # Backstop only — the flow limiter plus the stale-prune in
                # _on_result keep this map at a few entries.
                while len(self._pending) > 64:
                    self._pending.pop(next(iter(self._pending)))
            self.detector.detect_async(mp_image, ts_ms)
        else:
            result = self.detector.detect_for_video(mp_image, ts_ms)
            self._perf_add("mediapipe_ms", time.perf_counter() - t_convert)
            self._handle_result(result, stamp, w, h)
        self._perf_add("tick_total_ms", time.perf_counter() - t_tick)

    # -------------------------------------------------- detection result path
    def _on_result(self, result, output_image, ts_ms):
        """LIVE_STREAM callback — runs on MediaPipe's thread, NOT the executor."""
        del output_image
        with self._lock:
            entry = self._pending.pop(ts_ms, None)
            # Older submissions that never got a callback were discarded by
            # MediaPipe's flow limiter (inference slower than the camera).
            stale = [k for k in self._pending if k < ts_ms]
            for k in stale:
                del self._pending[k]
        if stale:
            self._perf_count("dropped", len(stale))
        if entry is None:
            return
        stamp, w, h, t_submit = entry
        # Submit -> result latency: queue wait + inference.
        self._perf_add("mediapipe_ms", time.perf_counter() - t_submit)
        self._handle_result(result, stamp, w, h)

    def _handle_result(self, result, stamp, w, h):
        t_start = time.perf_counter()
        msg = HandLandmarks()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_name
        msg.source_topic = self.image_topic

        self._gate("results")
        n_raw = len(result.handedness) if result.hand_landmarks else 0
        self._gate("raw_hands", n_raw)

        # ---- gate 1: did MediaPipe find a hand at all? ----------------------
        if self.log_detection:
            if n_raw == 0:
                self._diag(
                    "detect_none",
                    "[gate1 detect] MediaPipe returned 0 hands "
                    f"(min_hand_detection_confidence={self.min_hand_detection_confidence:.2f}, "
                    f"min_hand_presence_confidence={self.min_hand_presence_confidence:.2f}, "
                    f"min_tracking_confidence={self.min_tracking_confidence:.2f}, "
                    f"num_hands={self.num_hands}) — lower these to detect sooner",
                    "warn")
            else:
                labels = ", ".join(
                    f"{h[0].category_name}:{h[0].score:.2f}" for h in result.handedness)
                self._diag("detect_ok", f"[gate1 detect] {n_raw} raw hand(s): {labels}")

        if result.hand_landmarks:
            world = result.hand_world_landmarks or []
            for i, handed in enumerate(result.handedness):
                label = handed[0].category_name  # "Left" / "Right"
                score = handed[0].score
                # ---- gate 2: handedness label / score / world landmarks ----
                if label not in self.allowed_labels:
                    self._gate("rej_label")
                    if self.log_handedness:
                        self._diag(
                            f"rej_label_{label}",
                            f"[gate2 label] dropped '{label}' (score {score:.2f}): "
                            f"hand_filter_mode='{self.hand_filter_mode}' allows "
                            f"{sorted(self.allowed_labels)}. NOTE MediaPipe labels "
                            "assume an UNMIRRORED image, and the label can flip "
                            "when the hand rotates palm<->back",
                            "warn")
                    continue
                if score < self.min_handedness_confidence:
                    self._gate("rej_handedness_score")
                    if self.log_handedness:
                        self._diag(
                            f"rej_score_{label}",
                            f"[gate2 score] dropped '{label}': handedness score "
                            f"{score:.3f} < min_handedness_confidence "
                            f"{self.min_handedness_confidence:.3f}",
                            "warn")
                    continue
                if i >= len(world):
                    self._gate("rej_no_world")
                    if self.log_handedness:
                        self._diag(
                            f"rej_world_{label}",
                            f"[gate2 world] dropped '{label}': no world landmarks at "
                            f"index {i} (hand_world_landmarks has {len(world)})",
                            "warn")
                    continue
                self._gate("passed_hands")
                if self.log_handedness:
                    self._diag(
                        f"pass_{label}",
                        f"[gate2 pass] '{label}' score {score:.3f} >= "
                        f"{self.min_handedness_confidence:.3f}, label allowed")
                hand = Hand()
                hand.handedness = label
                hand.score = float(score)
                hand.landmarks_image = [
                    Point(x=float(lm.x * w), y=float(lm.y * h), z=float(lm.z))
                    for lm in result.hand_landmarks[i]]
                hand.landmarks_world = [
                    Point(x=float(lm.x), y=float(lm.y), z=float(lm.z))
                    for lm in world[i]]
                msg.hands.append(hand)

        self.landmarks_pub.publish(msg)
        if msg.hands:
            self._gate("frames_with_hand")
        if self.log_publish:
            self._diag(
                "publish" if msg.hands else "publish_empty",
                f"[stage publish] {len(msg.hands)}/{n_raw} hand(s) -> "
                f"{self.landmarks_topic} "
                f"({', '.join(h.handedness for h in msg.hands) or 'none'})",
                "info" if msg.hands else "warn")
        self._perf_add("publish_landmarks_ms", time.perf_counter() - t_start)
        self._perf_count("frames")
        self._perf_count("hands", len(msg.hands))

    def shutdown(self):
        # Close the detector first: it joins MediaPipe's worker, so no result
        # callback can fire into a half-torn-down node.
        self.detector.close()
        if self.cap is not None:
            self.cap.release()


def main(args=None):
    rclpy.init(args=args)
    node = HandLandmarksNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
