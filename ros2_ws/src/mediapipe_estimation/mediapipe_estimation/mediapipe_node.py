#!/usr/bin/env python3

"""
MediaPipe hand-pose preview node.

Plays a single recorded video file from disk, runs MediaPipe's HandLandmarker
on every frame, draws the 21-landmark skeleton over the frame and republishes
the annotated result as ``sensor_msgs/Image`` on ``<annotated_topic>`` (default
``mediapipe/annotated``) for visualization in RViz.

This is a quick-look tool for judging how well MediaPipe extracts the hand pose
from a given camera / recording — no calibration, no triangulation. Everything
is driven from a YAML config so a different video, resolution or detector
setting is just a config edit + relaunch.
"""

import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Hand skeleton connections (21 landmarks), matching the MediaPipe topology.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]
N_LANDMARKS = 21
# BGR overlay colors per handedness (OpenCV order).
HAND_BGR = {"Left": (255, 150, 50), "Right": (50, 150, 255)}


class MediaPipeNode(Node):
    def __init__(self):
        super().__init__("mediapipe_node")

        # --- parameters -----------------------------------------------------
        self.declare_parameter("video_path", "")
        self.declare_parameter(
            "model_path",
            "/workspace/ros2_ws/src/handpose_estimation/models/hand_landmarker.task",
        )
        # Source resolution of the recording. Frames are resized to this when
        # `resize_to_config` is true (handy when comparing cameras that record
        # at different native resolutions).
        self.declare_parameter("frame_width", 1920)
        self.declare_parameter("frame_height", 1080)
        self.declare_parameter("resize_to_config", False)
        # Playback rate. 0.0 -> use the video's own FPS (falls back to 30).
        self.declare_parameter("fps", 0.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("num_hands", 2)
        self.declare_parameter("min_hand_detection_confidence", 0.5)
        self.declare_parameter("min_hand_presence_confidence", 0.5)
        self.declare_parameter("min_tracking_confidence", 0.5)
        self.declare_parameter("frame_id", "camera")
        self.declare_parameter("annotated_topic", "mediapipe/annotated")
        self.declare_parameter("draw_handedness", True)

        self.video_path = self.get_parameter("video_path").value
        self.model_path = self.get_parameter("model_path").value
        self.frame_width = int(self.get_parameter("frame_width").value)
        self.frame_height = int(self.get_parameter("frame_height").value)
        self.resize_to_config = bool(self.get_parameter("resize_to_config").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.num_hands = int(self.get_parameter("num_hands").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.annotated_topic = self.get_parameter("annotated_topic").value
        self.draw_handedness = bool(self.get_parameter("draw_handedness").value)

        if not self.video_path:
            raise ValueError("'video_path' parameter is required (set it in the config).")
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Hand landmark model not found at {self.model_path}. "
                "Run scripts/download_model.sh to fetch hand_landmarker.task."
            )

        # --- video source ---------------------------------------------------
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        src_fps = self.cap.get(cv2.CAP_PROP_FPS)
        fps_param = float(self.get_parameter("fps").value)
        self.fps = fps_param if fps_param > 0.0 else (src_fps if src_fps > 0 else 30.0)
        n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # --- mediapipe detector (VIDEO mode -> monotonic timestamps) --------
        self.detector = self._make_landmarker()
        self._frame_idx = 0

        # --- publisher + playback timer -------------------------------------
        self.pub = self.create_publisher(
            Image, self.annotated_topic, qos_profile_sensor_data
        )
        self.timer = self.create_timer(1.0 / self.fps, self._on_tick)

        self.get_logger().info(
            f"mediapipe_node playing '{self.video_path}' "
            f"({src_w}x{src_h}, {n_frames} frames @ {self.fps:.1f} fps, "
            f"loop={self.loop}) -> '{self.annotated_topic}'"
        )

    def _make_landmarker(self):
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=self.num_hands,
            min_hand_detection_confidence=float(
                self.get_parameter("min_hand_detection_confidence").value
            ),
            min_hand_presence_confidence=float(
                self.get_parameter("min_hand_presence_confidence").value
            ),
            min_tracking_confidence=float(
                self.get_parameter("min_tracking_confidence").value
            ),
        )
        return mp_vision.HandLandmarker.create_from_options(options)

    # --------------------------------------------------------------- playback
    def _on_tick(self):
        ok, frame_bgr = self.cap.read()
        if not ok:
            if self.loop:
                # Rewind the video position only — _frame_idx must keep growing
                # so MediaPipe's VIDEO-mode timestamps stay monotonic across loops.
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame_bgr = self.cap.read()
            if not ok:
                self.get_logger().info("End of video; stopping playback.")
                self.timer.cancel()
                return

        if self.resize_to_config:
            frame_bgr = cv2.resize(frame_bgr, (self.frame_width, self.frame_height))

        timestamp_ms = self._frame_idx * int(round(1000.0 / self.fps))
        self._frame_idx += 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        hands = self._detect_hands(frame_rgb, timestamp_ms, w, h)
        self._draw(frame_bgr, hands)
        self._publish(frame_bgr)

    def _detect_hands(self, frame_rgb, timestamp_ms, width, height):
        """Run the landmarker; return {label: (21, 2) pixel landmarks}."""
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb)
        )
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        hands = {}
        scores = {}
        if result.hand_landmarks:
            for i, (lm_list, handed) in enumerate(
                zip(result.hand_landmarks, result.handedness)
            ):
                label = handed[0].category_name  # "Left" / "Right"
                score = handed[0].score
                # Keep both hands even if MediaPipe labels them the same: key
                # duplicates so the overlay never drops a detection.
                key = label if label not in hands else f"{label}{i}"
                if label in hands and score <= scores.get(label, 0.0):
                    key = f"{label}{i}"
                hands[key] = np.array(
                    [[lm.x * width, lm.y * height] for lm in lm_list], dtype=float
                )
                scores[label] = max(scores.get(label, 0.0), score)
        return hands

    def _draw(self, frame_bgr, hands):
        for label, kpts in hands.items():
            base = label.rstrip("0123456789")
            color = HAND_BGR.get(base, (255, 255, 255))
            pts = {
                p: (int(round(kpts[p, 0])), int(round(kpts[p, 1])))
                for p in range(N_LANDMARKS)
                if not np.isnan(kpts[p, 0])
            }
            for a, b in HAND_CONNECTIONS:
                if a in pts and b in pts:
                    cv2.line(frame_bgr, pts[a], pts[b], color, 2)
            for p in pts.values():
                cv2.circle(frame_bgr, p, 4, color, -1)
            if self.draw_handedness and 0 in pts:
                cv2.putText(
                    frame_bgr, base, (pts[0][0] + 6, pts[0][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
                )

    def _publish(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        img = Image()
        img.header.stamp = self.get_clock().now().to_msg()
        img.header.frame_id = self.frame_id
        img.height = h
        img.width = w
        img.encoding = "bgr8"
        img.is_bigendian = 0
        img.step = w * 3
        img.data = np.ascontiguousarray(frame_bgr).tobytes()
        self.pub.publish(img)

    def shutdown(self):
        self.detector.close()
        self.cap.release()


def main(args=None):
    rclpy.init(args=args)
    node = MediaPipeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
