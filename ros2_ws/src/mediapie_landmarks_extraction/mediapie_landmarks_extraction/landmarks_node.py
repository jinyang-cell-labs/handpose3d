#!/usr/bin/env python3

"""
MediaPipe hand-landmark extraction node.

A deliberately basic, standalone counterpart to ``handpose_estimation``: it does
NOT triangulate or estimate 3D pose. For each configured image topic it runs
MediaPipe's HandLandmarker on the incoming frames, draws the 21 2D landmarks +
skeleton onto a copy of the image, and republishes the annotated frame.

Everything is config-driven (see config/mediapie_landmarks_extraction.yaml):

    image_topics            list of input sensor_msgs/Image topics
    annotated_topics        optional explicit 1:1 output topics; if empty the
                            output topic is <input_topic> + annotated_suffix
    annotated_suffix        suffix appended to each input topic (when no
                            explicit annotated_topics are given)
    enable_annotation       master switch for publishing annotated images
    model_path / num_hands / min_*_confidence / running_mode   MediaPipe config

One detector is created per input topic so VIDEO-mode timestamps stay
independent across streams.
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

# Hand skeleton connections (21 landmarks), formerly
# mp.solutions.hands.HAND_CONNECTIONS.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]
N_LANDMARKS = 21

# BGR colors for the 2D annotated overlay (OpenCV order), per handedness.
HAND_BGR = {"Left": (255, 150, 50), "Right": (50, 150, 255)}
DEFAULT_BGR = (255, 255, 255)


class LandmarksNode(Node):
    def __init__(self):
        super().__init__("mediapie_landmarks_node")

        # --- parameters -----------------------------------------------------
        self.declare_parameter("image_topics", ["camera0/image_raw"])
        # Explicit 1:1 output topics. Leave empty to derive them from the input
        # topic names + annotated_suffix.
        self.declare_parameter("annotated_topics", [""])
        self.declare_parameter("annotated_suffix", "/landmarks/annotated")
        self.declare_parameter("enable_annotation", True)

        # MediaPipe HandLandmarker configuration.
        self.declare_parameter(
            "model_path",
            "/workspace/ros2_ws/src/mediapie_landmarks_extraction/models/"
            "hand_landmarker.task",
        )
        self.declare_parameter("num_hands", 2)
        self.declare_parameter("min_hand_detection_confidence", 0.5)
        self.declare_parameter("min_hand_presence_confidence", 0.5)
        self.declare_parameter("min_tracking_confidence", 0.5)
        # "video" -> detect_for_video (per-stream monotonic timestamps);
        # "image" -> detect (stateless per frame).
        self.declare_parameter("running_mode", "video")

        # Overlay drawing.
        self.declare_parameter("line_thickness", 2)
        self.declare_parameter("point_radius", 3)

        self.image_topics = [t for t in self.get_parameter("image_topics").value if t]
        if not self.image_topics:
            raise ValueError("image_topics must list at least one input topic")

        annotated = [t for t in self.get_parameter("annotated_topics").value if t]
        suffix = self.get_parameter("annotated_suffix").value
        if annotated:
            if len(annotated) != len(self.image_topics):
                raise ValueError(
                    "annotated_topics, when set, must be 1:1 with image_topics "
                    f"({len(annotated)} vs {len(self.image_topics)})"
                )
            self.annotated_topics = annotated
        else:
            self.annotated_topics = [t + suffix for t in self.image_topics]

        self.enable_annotation = bool(self.get_parameter("enable_annotation").value)
        self.model_path = self.get_parameter("model_path").value
        self.num_hands = int(self.get_parameter("num_hands").value)
        self.running_mode = str(self.get_parameter("running_mode").value).lower()
        self.line_thickness = int(self.get_parameter("line_thickness").value)
        self.point_radius = int(self.get_parameter("point_radius").value)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Hand landmark model not found at {self.model_path}. "
                "Run scripts/download_model.sh to fetch hand_landmarker.task."
            )

        # --- detectors / publishers / subscriptions, one per topic ---------
        # A separate detector per stream keeps VIDEO-mode timestamps independent.
        self.detectors = [self._make_landmarker() for _ in self.image_topics]
        self._frame_idx = [0] * len(self.image_topics)

        self.annotated_pubs = []
        self.subs = []
        for i, in_topic in enumerate(self.image_topics):
            pub = None
            if self.enable_annotation:
                pub = self.create_publisher(
                    Image, self.annotated_topics[i], qos_profile_sensor_data
                )
            self.annotated_pubs.append(pub)
            self.subs.append(
                self.create_subscription(
                    Image,
                    in_topic,
                    lambda msg, idx=i: self._on_image(msg, idx),
                    qos_profile_sensor_data,
                )
            )

        pairs = ", ".join(
            f"{i} -> {o}" if self.enable_annotation else f"{i} (annotation off)"
            for i, o in zip(self.image_topics, self.annotated_topics)
        )
        self.get_logger().info(
            f"mediapie_landmarks_node ready ({self.running_mode} mode, "
            f"num_hands={self.num_hands}): {pairs}"
        )

    # ------------------------------------------------------------------ setup
    def _make_landmarker(self):
        mode = (
            mp_vision.RunningMode.IMAGE
            if self.running_mode == "image"
            else mp_vision.RunningMode.VIDEO
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.model_path),
            running_mode=mode,
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

    # --------------------------------------------------------------- callback
    def _on_image(self, msg, idx):
        frame_bgr = self._decode_to_bgr(msg)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]

        hands = self._detect_hands(self.detectors[idx], frame_rgb, idx, w, h)

        n_hands = len(hands)
        self.get_logger().info(
            f"[{self.image_topics[idx]}] detected {n_hands} hand(s): "
            f"{sorted(hands)}",
            throttle_duration_sec=5.0,
        )

        if self.enable_annotation and self.annotated_pubs[idx] is not None:
            self._publish_annotated(idx, frame_bgr, hands, msg.header)

    def _detect_hands(self, detector, frame_rgb, idx, width, height):
        """Run the landmarker; return {label: (21, 2) pixels}.

        Label is MediaPipe's handedness category ("Left"/"Right"); if the same
        label is reported twice the higher-confidence detection wins.
        """
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        if self.running_mode == "image":
            result = detector.detect(mp_image)
        else:
            timestamp_ms = self._frame_idx[idx] * 33  # monotonic for VIDEO mode
            self._frame_idx[idx] += 1
            result = detector.detect_for_video(mp_image, timestamp_ms)

        hands, scores = {}, {}
        if result.hand_landmarks:
            for lm_list, handed in zip(result.hand_landmarks, result.handedness):
                label = handed[0].category_name  # "Left" / "Right"
                score = handed[0].score
                if label in hands and score <= scores[label]:
                    continue
                hands[label] = np.array(
                    [[lm.x * width, lm.y * height] for lm in lm_list], dtype=float
                )
                scores[label] = score
        return hands

    def _publish_annotated(self, idx, frame_bgr, hands, header):
        frame = frame_bgr.copy()
        for label, kpts in hands.items():
            color = HAND_BGR.get(label, DEFAULT_BGR)
            pts = {
                p: (int(round(kpts[p, 0])), int(round(kpts[p, 1])))
                for p in range(N_LANDMARKS)
                if not np.isnan(kpts[p, 0])
            }
            for a, b in HAND_CONNECTIONS:
                if a in pts and b in pts:
                    cv2.line(frame, pts[a], pts[b], color, self.line_thickness)
            for p in pts.values():
                cv2.circle(frame, p, self.point_radius, color, -1)

        h, w = frame.shape[:2]
        img = Image()
        img.header = header
        img.height = h
        img.width = w
        img.encoding = "bgr8"
        img.is_bigendian = 0
        img.step = w * 3
        img.data = np.ascontiguousarray(frame).tobytes()
        self.annotated_pubs[idx].publish(img)

    def _decode_to_bgr(self, msg):
        """Decode a sensor_msgs/Image to a contiguous bgr8 ndarray.

        Honors msg.encoding (rgb8/bgr8/rgba8/bgra8/mono8) and msg.step (row
        stride / padding) so rgb8 sources aren't R/B-swapped and padded rows
        aren't sheared.
        """
        enc = (msg.encoding or "bgr8").lower()
        channels = {
            "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8uc1": 1,
        }.get(enc, 3)

        buf = np.frombuffer(msg.data, dtype=np.uint8)
        step = msg.step if msg.step else msg.width * channels
        arr = buf[: step * msg.height].reshape(msg.height, step)
        arr = arr[:, : msg.width * channels].reshape(msg.height, msg.width, channels)

        if enc == "rgb8":
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif enc == "rgba8":
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif enc == "bgra8":
            bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif enc in ("mono8", "8uc1"):
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:  # bgr8 or unknown 3-channel
            bgr = arr[:, :, :3]
        return np.ascontiguousarray(bgr)

    def shutdown(self):
        for d in self.detectors:
            d.close()


def main(args=None):
    rclpy.init(args=args)
    node = LandmarksNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
