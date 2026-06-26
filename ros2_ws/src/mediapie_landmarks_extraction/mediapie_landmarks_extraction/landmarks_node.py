#!/usr/bin/env python3

"""
MediaPipe hand-landmark extraction node.

A deliberately basic, standalone counterpart to ``handpose_estimation``: it does
NOT triangulate. For each configured image topic it runs MediaPipe's
HandLandmarker on the incoming frames, draws the 21 2D landmarks + skeleton onto
a copy of the image, and republishes the annotated frame.

Optionally (``enable_3d_estimation``) it also publishes MediaPipe's own
``hand_world_landmarks`` as a ``visualization_msgs/MarkerArray`` for RViz. These
are the model's single-view metric (metres) 3D estimate in a HAND-LOCAL frame
(origin ~ the hand's geometric center) — NO camera_info / calibration is
involved, so the hands carry shape but no absolute world placement. Each
(camera, hand) skeleton is laid out at a distinct offset so they don't overlap.

Optionally (``enable_undistortion``) it undistorts each incoming frame using
the matching ``camera_info`` intrinsics (plumb_bob K/D only) BEFORE running the
detector, so the 2D landmarks and annotated image are both in the undistorted
pinhole image. This is pure lens undistortion (R=identity, output intrinsics =
K) — it does NOT rectify, so no stereo R/P (which depend on extrinsics) is
needed. Frames are dropped until the camera_info for that stream has arrived and
its undistort map is built.

Everything is config-driven (see config/mediapie_landmarks_extraction.yaml):

    image_topics            list of input sensor_msgs/Image topics
    enable_undistortion     undistort frames using camera_info K/D first
    camera_info_topics      optional 1:1 camera_info topics; if empty derived
                            as <image_topic dirname>/camera_info
    annotated_topics        optional explicit 1:1 output topics; if empty the
                            output topic is <input_topic> + annotated_suffix
    annotated_suffix        suffix appended to each input topic (when no
                            explicit annotated_topics are given)
    enable_annotation       master switch for publishing annotated images
    enable_3d_estimation    publish hand_world_landmarks as RViz markers
    enable_landmark_msg     publish handpose3d_msgs/HandLandmarks (2D+3D data)
    model_path / num_hands / min_*_confidence / running_mode   MediaPipe config

One detector is created per input topic so VIDEO-mode timestamps stay
independent across streams.
"""

import json
import os
import time

import cv2
import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from handpose3d_msgs.msg import Hand, HandLandmarks

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

# MediaPipe 21-landmark hand model, in index order. Recorded in the log meta so
# downstream evaluation can name joints without hard-coding the order.
JOINT_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# BGR colors for the 2D annotated overlay (OpenCV order), per handedness.
HAND_BGR = {"Left": (255, 150, 50), "Right": (50, 150, 255)}
DEFAULT_BGR = (255, 255, 255)
# RViz marker colors (RGBA) per handedness for the 3D world-landmark skeleton.
HAND_RGBA = {
    "Left": ColorRGBA(r=0.2, g=0.6, b=1.0, a=1.0),   # blue
    "Right": ColorRGBA(r=1.0, g=0.5, b=0.2, a=1.0),  # orange
}
DEFAULT_RGBA = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
# Stable (joints, bones) marker ids per hand so updates replace in place.
HAND_MARKER_IDS = {"Left": (0, 1), "Right": (2, 3)}


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

        # Undistortion: undistort each frame with its camera_info intrinsics
        # (plumb_bob K/D only) before detection so landmarks + annotation are in
        # the undistorted pinhole image. Pure lens undistortion (R=identity,
        # output intrinsics = K); no stereo rectification, so no R/P is used.
        # camera_info_topics, when empty, are derived as
        # <image_topic dirname>/camera_info.
        self.declare_parameter("enable_undistortion", False)
        self.declare_parameter("camera_info_topics", [""])

        # Data: publish landmarks (2D image + 3D world), handedness and score
        # as handpose3d_msgs/HandLandmarks, one topic per input image topic
        # (<input_topic> + landmarks_suffix).
        self.declare_parameter("enable_landmark_msg", True)
        self.declare_parameter("landmarks_suffix", "/landmarks/hands")

        # 3D: publish MediaPipe hand_world_landmarks (metres, hand-local frame)
        # as a MarkerArray for RViz. No camera_info / calibration involved.
        self.declare_parameter("enable_3d_estimation", False)
        self.declare_parameter("markers_3d_topic", "landmarks/markers_3d")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("joint_size", 0.01)      # m (sphere diameter)
        self.declare_parameter("line_width", 0.004)     # m (bone thickness)
        # Layout offsets so multiple cameras/hands don't render on top of
        # each other (world landmarks are all centered at the hand origin).
        self.declare_parameter("camera_spacing", 0.4)   # m between cameras (x)
        self.declare_parameter("hand_spacing", 0.25)    # m between L/R (y)

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
        # Post-detection filtering on MediaPipe's handedness label/score.
        self.declare_parameter("hand_filter_mode", "left_and_right")
        self.declare_parameter("min_handedness_confidence", 0.0)
        # "video" -> detect_for_video (per-stream monotonic timestamps);
        # "image" -> detect (stateless per frame).
        self.declare_parameter("running_mode", "video")

        # Overlay drawing.
        self.declare_parameter("line_thickness", 2)
        self.declare_parameter("point_radius", 3)

        # --- session logging (service-driven) ------------------------------
        # Master switch: create the start_log/stop_log services and capture the
        # rig calibration (intrinsics from camera_info, extrinsics from
        # extrinsics_file) so a recording can be started on demand. Each take is
        # one self-contained JSONL file: a "meta" header line with the
        # per-camera intrinsics/extrinsics, then one "frame" record per
        # processed image carrying the 2D + world hand landmarks.
        self.declare_parameter("enable_logging", True)
        self.declare_parameter("log_dir", "/workspace/ros2_ws/logs")
        # Extrinsics (T_world_cam 4x4 per camera) from calibration_multi_cam,
        # embedded in the log meta. Missing file -> extrinsics logged as null.
        self.declare_parameter(
            "extrinsics_file",
            "/workspace/ros2_ws/src/calibration_multi_cam/config/extrinsics.yaml",
        )
        # Per-stream camera name used as the log key and to look up extrinsics.
        # Empty ([""]) -> derived from each image topic's namespace
        # (camera0/image_raw -> camera0).
        self.declare_parameter("log_camera_names", [""])

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

        self.enable_undistortion = bool(
            self.get_parameter("enable_undistortion").value
        )
        self.enable_logging = bool(self.get_parameter("enable_logging").value)
        # camera_info is needed for undistortion (build remap) and/or logging
        # (record intrinsics). Subscribe once and share it between both.
        self.need_camera_info = self.enable_undistortion or self.enable_logging
        if self.need_camera_info:
            ci = [t for t in self.get_parameter("camera_info_topics").value if t]
            if ci:
                if len(ci) != len(self.image_topics):
                    raise ValueError(
                        "camera_info_topics, when set, must be 1:1 with "
                        f"image_topics ({len(ci)} vs {len(self.image_topics)})"
                    )
                self.camera_info_topics = ci
            else:
                self.camera_info_topics = [
                    self._derive_camera_info_topic(t) for t in self.image_topics
                ]
        else:
            self.camera_info_topics = []
        # Per-stream (map1, map2) for cv2.remap and the (w, h) they were built
        # for; populated lazily when each camera_info arrives.
        self.undistort_maps = [None] * len(self.image_topics)
        self._undistort_size = [None] * len(self.image_topics)
        # Per-stream intrinsics captured from camera_info (for the log meta):
        # {K, distortion, model, resolution}.
        self.intrinsics = [None] * len(self.image_topics)

        # --- logging config / state ----------------------------------------
        self.log_dir = self.get_parameter("log_dir").value
        self.extrinsics_file = self.get_parameter("extrinsics_file").value
        names = [n for n in self.get_parameter("log_camera_names").value if n]
        if names:
            if len(names) != len(self.image_topics):
                raise ValueError(
                    "log_camera_names, when set, must be 1:1 with image_topics "
                    f"({len(names)} vs {len(self.image_topics)})"
                )
            self.log_camera_names = names
        else:
            self.log_camera_names = [
                self._derive_camera_name(t) for t in self.image_topics
            ]
        self._log_file = None     # open file handle while a take is recording
        self._log_path = None
        self._log_count = 0       # frame records written this take

        self.enable_annotation = bool(self.get_parameter("enable_annotation").value)
        self.enable_landmark_msg = bool(
            self.get_parameter("enable_landmark_msg").value
        )
        self.landmark_topics = [
            t + self.get_parameter("landmarks_suffix").value
            for t in self.image_topics
        ]
        self.enable_3d_estimation = bool(
            self.get_parameter("enable_3d_estimation").value
        )
        self.world_frame = self.get_parameter("world_frame").value
        self.joint_size = float(self.get_parameter("joint_size").value)
        self.line_width = float(self.get_parameter("line_width").value)
        self.camera_spacing = float(self.get_parameter("camera_spacing").value)
        self.hand_spacing = float(self.get_parameter("hand_spacing").value)
        self.model_path = self.get_parameter("model_path").value
        self.num_hands = int(self.get_parameter("num_hands").value)
        # Handedness filtering: mode selects which labels to keep, threshold
        # drops detections whose handedness score is too low.
        self.hand_filter_mode = str(
            self.get_parameter("hand_filter_mode").value
        ).lower()
        allowed_by_mode = {
            "left_only": {"Left"},
            "right_only": {"Right"},
            "left_and_right": {"Left", "Right"},
        }
        if self.hand_filter_mode not in allowed_by_mode:
            raise ValueError(
                "hand_filter_mode must be one of "
                f"{sorted(allowed_by_mode)}, got '{self.hand_filter_mode}'"
            )
        self.allowed_labels = allowed_by_mode[self.hand_filter_mode]
        self.min_handedness_confidence = float(
            self.get_parameter("min_handedness_confidence").value
        )
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
        self.landmark_pubs = []
        self.subs = []
        for i, in_topic in enumerate(self.image_topics):
            pub = None
            if self.enable_annotation:
                pub = self.create_publisher(
                    Image, self.annotated_topics[i], qos_profile_sensor_data
                )
            self.annotated_pubs.append(pub)

            lm_pub = None
            if self.enable_landmark_msg:
                lm_pub = self.create_publisher(
                    HandLandmarks, self.landmark_topics[i], 10
                )
            self.landmark_pubs.append(lm_pub)

            self.subs.append(
                self.create_subscription(
                    Image,
                    in_topic,
                    lambda msg, idx=i: self._on_image(msg, idx),
                    qos_profile_sensor_data,
                )
            )

        # --- camera_info subscriptions for undistortion (optional) ---------
        # The calibration publisher latches camera_info (TRANSIENT_LOCAL, depth
        # 1): it is sent once and never republished. Match that QoS so this
        # (late-joining) subscriber actually receives the cached sample; a
        # VOLATILE subscriber would connect but never get it.
        latching_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.info_subs = []
        if self.need_camera_info:
            for i, ci_topic in enumerate(self.camera_info_topics):
                self.info_subs.append(
                    self.create_subscription(
                        CameraInfo,
                        ci_topic,
                        lambda msg, idx=i: self._on_camera_info(msg, idx),
                        latching_qos,
                    )
                )

        # --- logging services ----------------------------------------------
        self.start_log_srv = None
        self.stop_log_srv = None
        if self.enable_logging:
            self.start_log_srv = self.create_service(
                Trigger, "~/start_log", self._on_start_log
            )
            self.stop_log_srv = self.create_service(
                Trigger, "~/stop_log", self._on_stop_log
            )

        # --- 3D world-landmark markers (optional) --------------------------
        self.markers_pub = None
        self.static_tf_broadcaster = None
        if self.enable_3d_estimation:
            self.markers_pub = self.create_publisher(
                MarkerArray, self.get_parameter("markers_3d_topic").value, 10
            )
            # world_world_landmarks have no absolute placement; publish a single
            # identity transform so `world_frame` exists in TF and RViz can use
            # it as the fixed frame.
            self.static_tf_broadcaster = StaticTransformBroadcaster(self)
            self._broadcast_world_frame()

        pairs = ", ".join(
            f"{i} -> {o}" if self.enable_annotation else f"{i} (annotation off)"
            for i, o in zip(self.image_topics, self.annotated_topics)
        )
        self.get_logger().info(
            f"mediapie_landmarks_node ready ({self.running_mode} mode, "
            f"num_hands={self.num_hands}, filter={self.hand_filter_mode}"
            f"@>={self.min_handedness_confidence}, "
            f"undistort={self.enable_undistortion}, "
            f"3d={self.enable_3d_estimation}, "
            f"landmark_msg={self.enable_landmark_msg}, "
            f"logging={self.enable_logging}): {pairs}"
        )
        if self.enable_logging:
            self.get_logger().info(
                "logging ready: call ~/start_log then ~/stop_log "
                f"(std_srvs/Trigger); files go to {self.log_dir}"
            )

    @staticmethod
    def _derive_camera_info_topic(image_topic):
        """Map an image topic to its conventional camera_info sibling.

        ``camera0/image_raw`` -> ``camera0/camera_info``.
        """
        base = image_topic.rsplit("/", 1)[0] if "/" in image_topic else ""
        return f"{base}/camera_info" if base else "camera_info"

    @staticmethod
    def _derive_camera_name(image_topic):
        """Camera name = the image topic's namespace.

        ``camera0/image_raw`` -> ``camera0``; ``/ns/camera0/image_raw`` ->
        ``camera0``. Used as the log key and to look up extrinsics by name.
        """
        base = image_topic.rsplit("/", 1)[0] if "/" in image_topic else image_topic
        return base.strip("/").rsplit("/", 1)[-1] or image_topic

    def _on_camera_info(self, msg, idx):
        """Capture intrinsics for stream ``idx`` and (if on) build its undistort map.

        Intrinsics (K, distortion, model, resolution) are stored once for the
        log meta. For undistortion: pure lens undistortion using plumb_bob K/D
        only (rectification R = identity, output intrinsics kept at K; no stereo
        R/P, which would depend on extrinsics). The map is built once and only
        rebuilt if the reported image size changes.
        """
        size = (msg.width, msg.height)
        if self.intrinsics[idx] is None:
            self.intrinsics[idx] = {
                "K": [float(v) for v in msg.k],
                "distortion": [float(v) for v in msg.d],
                "model": (msg.distortion_model or "plumb_bob"),
                "resolution": [int(msg.width), int(msg.height)],
            }
        if not self.enable_undistortion:
            return
        if (
            self.undistort_maps[idx] is not None
            and self._undistort_size[idx] == size
        ):
            return
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.d, dtype=np.float64)
        # R=None -> identity (no rectification); new camera matrix = K so the
        # undistorted image keeps the same focal length / principal point.
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, None, K, size, cv2.CV_16SC2
        )
        self.undistort_maps[idx] = (map1, map2)
        self._undistort_size[idx] = size
        self.get_logger().info(
            f"[{self.image_topics[idx]}] undistort map built from "
            f"{self.camera_info_topics[idx]} ({size[0]}x{size[1]})"
        )

    def _broadcast_world_frame(self):
        """Register `world_frame` in TF via one identity transform.

        Markers are published in `world_frame`; RViz needs the fixed frame to
        exist in the TF tree, so emit an identity world_frame -> *_origin.
        """
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.world_frame
        tf.child_frame_id = f"{self.world_frame}_origin"
        tf.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform([tf])

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

        if self.enable_undistortion:
            maps = self.undistort_maps[idx]
            if maps is None:
                self.get_logger().warn(
                    f"[{self.image_topics[idx]}] waiting for camera_info on "
                    f"{self.camera_info_topics[idx]} before undistorting; "
                    "dropping frame",
                    throttle_duration_sec=5.0,
                )
                return
            frame_bgr = cv2.remap(
                frame_bgr, maps[0], maps[1], cv2.INTER_LINEAR
            )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]

        hands, world_hands, scores = self._detect_hands(
            self.detectors[idx], frame_rgb, idx, w, h
        )

        n_hands = len(hands)
        self.get_logger().info(
            f"[{self.image_topics[idx]}] detected {n_hands} hand(s): "
            f"{sorted(hands)}",
            throttle_duration_sec=5.0,
        )

        if self.enable_annotation and self.annotated_pubs[idx] is not None:
            self._publish_annotated(idx, frame_bgr, hands, msg.header)

        if self.enable_landmark_msg and self.landmark_pubs[idx] is not None:
            self._publish_landmarks(idx, hands, world_hands, scores, msg.header)

        if self.enable_3d_estimation and self.markers_pub is not None:
            self._publish_world_markers(idx, world_hands, msg.header.stamp)

        if self._log_file is not None:
            self._log_frame(idx, hands, world_hands, scores, msg.header.stamp)

    def _detect_hands(self, detector, frame_rgb, idx, width, height):
        """Run the landmarker; return (hands, world_hands, scores).

        - hands:       {label: (21, 3)} image landmarks — x,y in pixels, z the
                       model's relative depth.
        - world_hands: {label: (21, 3)} hand_world_landmarks in metres
                       (hand-local frame); populated only when 3D or the
                       landmark message is enabled.
        - scores:      {label: float} per-hand handedness confidence.

        Label is MediaPipe's handedness category ("Left"/"Right"); if the same
        label is reported twice the higher-confidence detection wins.

        Detections are filtered by ``hand_filter_mode`` (which labels to keep)
        and ``min_handedness_confidence`` (minimum handedness score) before being
        returned.
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

        want_world = self.enable_3d_estimation or self.enable_landmark_msg
        hands, world_hands, scores = {}, {}, {}
        if result.hand_landmarks:
            world = result.hand_world_landmarks or []
            for h, handed in enumerate(result.handedness):
                label = handed[0].category_name  # "Left" / "Right"
                score = handed[0].score
                # Drop hands excluded by mode or below the handedness threshold.
                if label not in self.allowed_labels:
                    continue
                if score < self.min_handedness_confidence:
                    continue
                if label in hands and score <= scores[label]:
                    continue
                lm_list = result.hand_landmarks[h]
                hands[label] = np.array(
                    [[lm.x * width, lm.y * height, lm.z] for lm in lm_list],
                    dtype=float,
                )
                scores[label] = score
                if want_world and h < len(world):
                    world_hands[label] = np.array(
                        [[lm.x, lm.y, lm.z] for lm in world[h]], dtype=float
                    )
        return hands, world_hands, scores

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

    def _publish_landmarks(self, idx, hands, world_hands, scores, header):
        """Publish one frame's detections as handpose3d_msgs/HandLandmarks.

        Image landmarks carry x,y pixels + z relative depth; world landmarks
        (metres, hand-local) are included when available, else left empty.
        """
        msg = HandLandmarks()
        msg.header = header
        msg.source_topic = self.image_topics[idx]
        for label, kpts in hands.items():
            hand = Hand()
            hand.handedness = label
            hand.score = float(scores.get(label, 0.0))
            hand.landmarks_image = [
                Point(x=float(kpts[p, 0]), y=float(kpts[p, 1]), z=float(kpts[p, 2]))
                for p in range(N_LANDMARKS)
            ]
            world = world_hands.get(label)
            if world is not None:
                hand.landmarks_world = [
                    Point(x=float(world[p, 0]), y=float(world[p, 1]),
                          z=float(world[p, 2]))
                    for p in range(N_LANDMARKS)
                ]
            msg.hands.append(hand)
        self.landmark_pubs[idx].publish(msg)

    def _publish_world_markers(self, idx, world_hands, stamp):
        """Publish one camera's hand_world_landmarks as RViz markers.

        Each hand's 21 metric points are centered at the hand origin, so the
        whole skeleton is shifted by a per-(camera, hand) offset to keep
        multiple hands/cameras from overlapping. Both Left and Right are always
        published (empty when absent) so a vanished hand clears in RViz.
        """
        marker_array = MarkerArray()
        cam_off = idx * self.camera_spacing
        for h, label in enumerate(("Left", "Right")):
            pts3d = world_hands.get(label)
            color = HAND_RGBA.get(label, DEFAULT_RGBA)
            joint_id, bone_id = HAND_MARKER_IDS[label]
            # Left to the left, Right to the right; cameras spread along x.
            off_x = cam_off
            off_y = (-1.0 if label == "Left" else 1.0) * (self.hand_spacing / 2.0)

            joints = Marker()
            joints.header.frame_id = self.world_frame
            joints.header.stamp = stamp
            joints.ns = f"cam{idx}_{label.lower()}_joints"
            joints.id = joint_id
            joints.type = Marker.SPHERE_LIST
            joints.action = Marker.ADD
            joints.scale.x = joints.scale.y = joints.scale.z = self.joint_size
            joints.color = color
            joints.lifetime = Duration(sec=0, nanosec=500_000_000)
            joints.pose.position.x = off_x
            joints.pose.position.y = off_y
            joints.pose.orientation.w = 1.0

            bones = Marker()
            bones.header.frame_id = self.world_frame
            bones.header.stamp = stamp
            bones.ns = f"cam{idx}_{label.lower()}_bones"
            bones.id = bone_id
            bones.type = Marker.LINE_LIST
            bones.action = Marker.ADD
            bones.scale.x = self.line_width
            bones.color = color
            bones.lifetime = Duration(sec=0, nanosec=500_000_000)
            bones.pose.position.x = off_x
            bones.pose.position.y = off_y
            bones.pose.orientation.w = 1.0

            if pts3d is not None:
                def to_point(i, _p=pts3d):
                    return Point(x=float(_p[i, 0]), y=float(_p[i, 1]),
                                 z=float(_p[i, 2]))

                for p in range(N_LANDMARKS):
                    joints.points.append(to_point(p))
                for a, b in HAND_CONNECTIONS:
                    bones.points.append(to_point(a))
                    bones.points.append(to_point(b))

            marker_array.markers.append(joints)
            marker_array.markers.append(bones)
        self.markers_pub.publish(marker_array)

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

    # ------------------------------------------------------------- logging
    def _on_start_log(self, request, response):
        """Begin a recording take: open a JSONL file and write the meta header."""
        if self._log_file is not None:
            response.success = False
            response.message = f"already logging to {self._log_path}"
            return response
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.log_dir, f"handpose_log_{stamp}.jsonl")
            meta = self._build_meta()
            f = open(path, "w")
            f.write(json.dumps(meta) + "\n")
            f.flush()
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f"failed to start log: {exc}"
            self.get_logger().error(response.message)
            return response
        self._log_file = f
        self._log_path = path
        self._log_count = 0
        missing = [
            self.log_camera_names[i]
            for i in range(len(self.image_topics))
            if self.intrinsics[i] is None
        ]
        warn = (
            f" (WARNING: no camera_info yet for {missing}, "
            "intrinsics logged as null)"
            if missing
            else ""
        )
        response.success = True
        response.message = f"logging to {path}{warn}"
        self.get_logger().info(response.message)
        return response

    def _on_stop_log(self, request, response):
        """Finalize the current take: flush and close the JSONL file."""
        if self._log_file is None:
            response.success = False
            response.message = "not currently logging"
            return response
        path, count = self._log_path, self._log_count
        try:
            self._log_file.flush()
            self._log_file.close()
        finally:
            self._log_file = None
            self._log_path = None
        response.success = True
        response.message = f"wrote {count} frame records to {path}"
        self.get_logger().info(response.message)
        return response

    def _build_meta(self):
        """Assemble the session meta header (calibration + schema)."""
        extrinsics, world_frame = self._load_extrinsics_for_log()
        cameras = {}
        for i, img_topic in enumerate(self.image_topics):
            name = self.log_camera_names[i]
            cameras[name] = {
                "image_topic": img_topic,
                "landmark_topic": self.landmark_topics[i],
                "intrinsics": self.intrinsics[i],          # None until camera_info
                "T_world_cam": extrinsics.get(name),       # None if file missing
            }
        return {
            "type": "meta",
            "schema_version": 1,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "world_frame": world_frame,
            # True when landmarks_image is in the undistorted pinhole image
            # (enable_undistortion), so consumers know which intrinsics apply.
            "landmarks_undistorted": self.enable_undistortion,
            "num_hands": self.num_hands,
            "joint_names": JOINT_NAMES,
            "cameras": cameras,
        }

    def _load_extrinsics_for_log(self):
        """Load ``T_world_cam`` per camera + world_frame from extrinsics_file.

        Returns ``({name: 4x4 list}, world_frame)``; on a missing/unreadable file
        returns ``({}, "")`` with a warning rather than failing the take.
        """
        path = self.extrinsics_file
        if not path or not os.path.isfile(path):
            self.get_logger().warn(
                f"extrinsics_file '{path}' not found; logging extrinsics as null"
            )
            return {}, ""
        try:
            with open(path, "r") as fh:
                data = yaml.safe_load(fh)
            world_frame = data.get("world_frame", "")
            ext = {
                name: [[float(v) for v in row] for row in c["T_world_cam"]]
                for name, c in data.get("cameras", {}).items()
            }
            return ext, world_frame
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"failed to read extrinsics_file: {exc}")
            return {}, ""

    def _log_frame(self, idx, hands, world_hands, scores, stamp):
        """Append one processed frame's detections as a JSONL record.

        Written even with zero hands so the timeline records detection gaps.
        landmarks_image is (21, 3) [x_px, y_px, z_rel]; landmarks_world is
        (21, 3) metres (hand-local) or null when 3D was unavailable.
        """
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        hand_records = []
        for label, kpts in hands.items():
            world = world_hands.get(label)
            hand_records.append({
                "handedness": label,
                "score": float(scores.get(label, 0.0)),
                "landmarks_image": kpts.tolist(),
                "landmarks_world": world.tolist() if world is not None else None,
            })
        record = {
            "type": "frame",
            "camera": self.log_camera_names[idx],
            "stamp_ns": stamp_ns,
            "hands": hand_records,
        }
        try:
            self._log_file.write(json.dumps(record) + "\n")
            self._log_count += 1
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"log write failed, stopping take: {exc}",
                throttle_duration_sec=5.0,
            )
            try:
                self._log_file.close()
            finally:
                self._log_file = None
                self._log_path = None

    def shutdown(self):
        if self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
                self.get_logger().info(
                    f"closed log {self._log_path} ({self._log_count} records)"
                )
            except Exception:  # noqa: BLE001
                pass
            self._log_file = None
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
