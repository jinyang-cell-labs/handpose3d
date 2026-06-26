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

Optionally (``enable_rectification``) it undistorts/rectifies each incoming
frame using the matching ``camera_info`` (plumb_bob K/D/R/P) BEFORE running the
detector, so the 2D landmarks and annotated image are both in the rectified
(pinhole) image. Frames are dropped until the camera_info for that stream has
arrived and its rectify map is built.

Everything is config-driven (see config/mediapie_landmarks_extraction.yaml):

    image_topics            list of input sensor_msgs/Image topics
    enable_rectification    undistort/rectify frames using camera_info first
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

import os

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
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

        # Rectification: undistort/rectify each frame with its camera_info
        # (plumb_bob K/D/R/P) before detection so landmarks + annotation are in
        # the rectified pinhole image. camera_info_topics, when empty, are
        # derived as <image_topic dirname>/camera_info.
        self.declare_parameter("enable_rectification", False)
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

        self.enable_rectification = bool(
            self.get_parameter("enable_rectification").value
        )
        if self.enable_rectification:
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
        self.rectify_maps = [None] * len(self.image_topics)
        self._rectify_size = [None] * len(self.image_topics)

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

        # --- camera_info subscriptions for rectification (optional) --------
        self.info_subs = []
        if self.enable_rectification:
            for i, ci_topic in enumerate(self.camera_info_topics):
                self.info_subs.append(
                    self.create_subscription(
                        CameraInfo,
                        ci_topic,
                        lambda msg, idx=i: self._on_camera_info(msg, idx),
                        qos_profile_sensor_data,
                    )
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
            f"rectify={self.enable_rectification}, "
            f"3d={self.enable_3d_estimation}, "
            f"landmark_msg={self.enable_landmark_msg}): {pairs}"
        )

    @staticmethod
    def _derive_camera_info_topic(image_topic):
        """Map an image topic to its conventional camera_info sibling.

        ``camera0/image_raw`` -> ``camera0/camera_info``.
        """
        base = image_topic.rsplit("/", 1)[0] if "/" in image_topic else ""
        return f"{base}/camera_info" if base else "camera_info"

    def _on_camera_info(self, msg, idx):
        """Build the undistort/rectify map for stream ``idx`` from camera_info.

        Uses plumb_bob K/D, the rectification R and the projection P (the
        rectified pinhole intrinsics are P[:3, :3]). The map is built once and
        only rebuilt if the reported image size changes.
        """
        size = (msg.width, msg.height)
        if self.rectify_maps[idx] is not None and self._rectify_size[idx] == size:
            return
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.d, dtype=np.float64)
        R = np.array(msg.r, dtype=np.float64).reshape(3, 3)
        new_K = np.array(msg.p, dtype=np.float64).reshape(3, 4)[:, :3]
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, R, new_K, size, cv2.CV_16SC2
        )
        self.rectify_maps[idx] = (map1, map2)
        self._rectify_size[idx] = size
        self.get_logger().info(
            f"[{self.image_topics[idx]}] rectify map built from "
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

        if self.enable_rectification:
            maps = self.rectify_maps[idx]
            if maps is None:
                self.get_logger().warn(
                    f"[{self.image_topics[idx]}] waiting for camera_info on "
                    f"{self.camera_info_topics[idx]} before rectifying; "
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
