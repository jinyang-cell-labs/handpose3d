#!/usr/bin/env python3

"""
Hand pose estimation node.

Subscribes to two calibrated camera streams published by ``vision_interfaces``:

    <name>/image_raw    sensor_msgs/Image      (bgr8)
    <name>/camera_info  sensor_msgs/CameraInfo

Runs MediaPipe's HandLandmarker on each view to get 2D hand keypoints, then
triangulates the 21 landmarks to 3D via the Direct Linear Transform (intrinsics
come from the camera_info topics, stereo extrinsics from the node's config).

The 3D skeleton is published as a ``visualization_msgs/MarkerArray`` in the
world frame for visualization in RViz. The annotated 2D views are optionally
republished as ``<name>/handpose/annotated``.
"""

import os

import cv2
import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from handpose_estimation.triangulation import (
    dlt,
    make_projection_matrix,
    rotation_matrix_to_quaternion,
)

# Hand skeleton connections (21 landmarks), formerly mp.solutions.hands.HAND_CONNECTIONS
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]
N_LANDMARKS = 21


class HandPoseNode(Node):
    def __init__(self):
        super().__init__("handpose_node")

        # --- parameters -----------------------------------------------------
        self.declare_parameter("camera_names", ["camera0", "camera1"])
        self.declare_parameter(
            "model_path",
            "/workspace/ros2_ws/src/handpose_estimation/models/hand_landmarker.task",
        )
        self.declare_parameter(
            "extrinsics_file",
            "/workspace/ros2_ws/src/handpose_estimation/config/extrinsics.yaml",
        )
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("num_hands", 1)
        self.declare_parameter("min_hand_detection_confidence", 0.5)
        self.declare_parameter("min_hand_presence_confidence", 0.5)
        self.declare_parameter("min_tracking_confidence", 0.5)
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("sync_queue_size", 10)
        # Calibration world units -> metres for RViz-friendly marker sizes.
        self.declare_parameter("scale", 0.05)
        self.declare_parameter("joint_size", 0.02)
        self.declare_parameter("line_width", 0.01)
        self.declare_parameter("publish_annotated", True)
        # Publish each camera's pose (from extrinsics) as TF + a frustum marker.
        self.declare_parameter("publish_camera_pose", True)
        self.declare_parameter("camera_marker_size", 0.08)

        self.camera_names = list(self.get_parameter("camera_names").value)
        if len(self.camera_names) != 2:
            raise ValueError("handpose_node requires exactly 2 camera_names")
        self.model_path = self.get_parameter("model_path").value
        self.extrinsics_file = self.get_parameter("extrinsics_file").value
        self.world_frame = self.get_parameter("world_frame").value
        self.num_hands = int(self.get_parameter("num_hands").value)
        self.scale = float(self.get_parameter("scale").value)
        self.joint_size = float(self.get_parameter("joint_size").value)
        self.line_width = float(self.get_parameter("line_width").value)
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self.publish_camera_pose = bool(
            self.get_parameter("publish_camera_pose").value
        )
        self.camera_marker_size = float(
            self.get_parameter("camera_marker_size").value
        )

        # --- mediapipe detectors (one per camera so VIDEO timestamps stay
        # independent) ------------------------------------------------------
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Hand landmark model not found at {self.model_path}. "
                "Run scripts/download_model.sh to fetch hand_landmarker.task."
            )
        self.detectors = [self._make_landmarker() for _ in self.camera_names]
        self._frame_idx = 0

        # --- calibration state ---------------------------------------------
        self.extrinsics = self._load_extrinsics(self.extrinsics_file)
        # Intrinsics arrive on the camera_info topics; projection matrices are
        # built lazily once both K matrices are known.
        self.K = {name: None for name in self.camera_names}
        self.P = {name: None for name in self.camera_names}

        # --- subscriptions & publishers ------------------------------------
        self.info_subs = []
        for name in self.camera_names:
            self.info_subs.append(
                self.create_subscription(
                    CameraInfo,
                    f"{name}/camera_info",
                    lambda msg, n=name: self._on_camera_info(msg, n),
                    qos_profile_sensor_data,
                )
            )

        image_subs = [
            Subscriber(self, Image, f"{name}/image_raw", qos_profile=qos_profile_sensor_data)
            for name in self.camera_names
        ]
        self.sync = ApproximateTimeSynchronizer(
            image_subs,
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop").value),
        )
        self.sync.registerCallback(self._on_images)

        self.marker_pub = self.create_publisher(MarkerArray, "handpose/markers", 10)
        self.annotated_pubs = {}
        if self.publish_annotated:
            for name in self.camera_names:
                self.annotated_pubs[name] = self.create_publisher(
                    Image, f"{name}/handpose/annotated", qos_profile_sensor_data
                )

        # --- camera poses ---------------------------------------------------
        # Broadcast each camera's pose as static TF (this also gives RViz the
        # 'world' frame to use as fixed frame) and a latched frustum marker so
        # late-joining RViz still receives it.
        if self.publish_camera_pose:
            self.static_tf_broadcaster = StaticTransformBroadcaster(self)
            latching_qos = QoSProfile(
                depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
            )
            self.camera_marker_pub = self.create_publisher(
                MarkerArray, "handpose/cameras", latching_qos
            )
            self._broadcast_camera_poses()
            self._publish_camera_markers()

        self.get_logger().info(
            f"handpose_node ready: cameras={self.camera_names}, "
            f"world_frame='{self.world_frame}', waiting for camera_info + images..."
        )

    # ------------------------------------------------------------------ setup
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

    def _load_extrinsics(self, path):
        """Load per-camera world->camera rotation/translation from YAML."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Extrinsics file not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        cameras = data["cameras"]
        ext = {}
        for name in self.camera_names:
            if name not in cameras:
                raise KeyError(f"No extrinsics for camera '{name}' in {path}")
            R = np.array(cameras[name]["rotation"], dtype=float).reshape(3, 3)
            t = np.array(cameras[name]["translation"], dtype=float).reshape(3)
            ext[name] = (R, t)
        return ext

    # ------------------------------------------------------------ camera poses
    def _broadcast_camera_poses(self):
        """Publish world->camera static transforms from the extrinsics.

        Extrinsics are world->camera (X_cam = R X_world + t), so the camera's
        pose in the world is the inverse: orientation R^T, centre -R^T t. The
        centre is scaled by `scale` to share the hand markers' metric space.
        """
        stamp = self.get_clock().now().to_msg()
        transforms = []
        for name in self.camera_names:
            R, t = self.extrinsics[name]
            R_wc = R.T
            center = (-R_wc @ t) * self.scale
            q = rotation_matrix_to_quaternion(R_wc)

            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.world_frame
            tf.child_frame_id = name
            tf.transform.translation.x = float(center[0])
            tf.transform.translation.y = float(center[1])
            tf.transform.translation.z = float(center[2])
            tf.transform.rotation.x = float(q[0])
            tf.transform.rotation.y = float(q[1])
            tf.transform.rotation.z = float(q[2])
            tf.transform.rotation.w = float(q[3])
            transforms.append(tf)
        self.static_tf_broadcaster.sendTransform(transforms)
        self.get_logger().info(
            f"Broadcast camera poses to TF: {self.camera_names}"
        )

    def _publish_camera_markers(self):
        """Draw a small frustum per camera in its own (optical) frame."""
        d = self.camera_marker_size
        w, h = d * 0.6, d * 0.45
        # Optical-frame convention: x right, y down, z forward.
        corners = [(-w, -h, d), (w, -h, d), (w, h, d), (-w, h, d)]

        array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, name in enumerate(self.camera_names):
            m = Marker()
            m.header.frame_id = name
            m.header.stamp = stamp
            m.ns = "camera_frustum"
            m.id = i
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = max(d * 0.02, 0.002)
            m.color = ColorRGBA(r=0.2, g=0.8, b=1.0, a=1.0)
            m.pose.orientation.w = 1.0

            apex = Point(x=0.0, y=0.0, z=0.0)
            cpts = [Point(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in corners]
            for cp in cpts:  # apex -> each corner
                m.points.append(apex)
                m.points.append(cp)
            for j in range(4):  # rectangle around the far plane
                m.points.append(cpts[j])
                m.points.append(cpts[(j + 1) % 4])
            array.markers.append(m)
        self.camera_marker_pub.publish(array)

    # --------------------------------------------------------------- callbacks
    def _on_camera_info(self, msg, name):
        if self.K[name] is not None:
            return  # intrinsics are static; capture once
        self.K[name] = np.array(msg.k, dtype=float).reshape(3, 3)
        R, t = self.extrinsics[name]
        self.P[name] = make_projection_matrix(self.K[name], R, t)
        self.get_logger().info(f"Built projection matrix for {name}")

    def _on_images(self, *msgs):
        # All projection matrices must be ready before we can triangulate.
        if any(self.P[name] is None for name in self.camera_names):
            self.get_logger().warn(
                "Waiting for camera_info on all cameras...",
                throttle_duration_sec=5.0,
            )
            return

        timestamp_ms = self._frame_idx * 33  # monotonically increasing for VIDEO mode
        self._frame_idx += 1

        keypoints_2d = []  # per-camera (21, 2) pixel arrays (NaN where missing)
        for i, (name, msg) in enumerate(zip(self.camera_names, msgs)):
            frame_bgr = self._decode_to_bgr(msg)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb)
            )
            result = self.detectors[i].detect_for_video(mp_image, timestamp_ms)

            kpts = np.full((N_LANDMARKS, 2), np.nan)
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                for p in range(N_LANDMARKS):
                    kpts[p, 0] = hand[p].x * msg.width
                    kpts[p, 1] = hand[p].y * msg.height
            keypoints_2d.append(kpts)

            if self.publish_annotated and name in self.annotated_pubs:
                self._publish_annotated(name, frame_bgr.copy(), kpts, msg.header)

        # --- triangulate ----------------------------------------------------
        P0 = self.P[self.camera_names[0]]
        P1 = self.P[self.camera_names[1]]
        kp0, kp1 = keypoints_2d[0], keypoints_2d[1]

        points_3d = np.full((N_LANDMARKS, 3), np.nan)
        for p in range(N_LANDMARKS):
            if np.isnan(kp0[p, 0]) or np.isnan(kp1[p, 0]):
                continue
            points_3d[p] = dlt(P0, P1, kp0[p], kp1[p])

        self._publish_markers(points_3d, msgs[0].header.stamp)

    def _decode_to_bgr(self, msg):
        """Decode a sensor_msgs/Image to a contiguous bgr8 ndarray.

        Honors msg.encoding (rgb8/bgr8/rgba8/bgra8/mono8) and msg.step (row
        stride / padding). The previous code hard-assumed bgr8 with no padding,
        which silently swaps R/B (poor MediaPipe detection) for rgb8 sources or
        shears the image when rows are padded.
        """
        enc = (msg.encoding or "bgr8").lower()
        channels = {
            "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8uc1": 1,
        }.get(enc, 3)

        buf = np.frombuffer(msg.data, dtype=np.uint8)
        step = msg.step if msg.step else msg.width * channels
        # Reshape by stride, then drop any trailing row padding.
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

    # ------------------------------------------------------------- publishing
    def _publish_markers(self, points_3d, stamp):
        marker_array = MarkerArray()

        # Joints (spheres)
        joints = Marker()
        joints.header.frame_id = self.world_frame
        joints.header.stamp = stamp
        joints.ns = "hand_joints"
        joints.id = 0
        joints.type = Marker.SPHERE_LIST
        joints.action = Marker.ADD
        joints.scale.x = joints.scale.y = joints.scale.z = self.joint_size
        joints.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)
        joints.lifetime = Duration(sec=0, nanosec=200_000_000)
        joints.pose.orientation.w = 1.0

        # Bones (line list)
        bones = Marker()
        bones.header.frame_id = self.world_frame
        bones.header.stamp = stamp
        bones.ns = "hand_bones"
        bones.id = 1
        bones.type = Marker.LINE_LIST
        bones.action = Marker.ADD
        bones.scale.x = self.line_width
        bones.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        bones.lifetime = Duration(sec=0, nanosec=200_000_000)
        bones.pose.orientation.w = 1.0

        def to_point(idx):
            x, y, z = points_3d[idx] * self.scale
            return Point(x=float(x), y=float(y), z=float(z))

        valid = ~np.isnan(points_3d[:, 0])
        for p in range(N_LANDMARKS):
            if valid[p]:
                joints.points.append(to_point(p))
        for a, b in HAND_CONNECTIONS:
            if valid[a] and valid[b]:
                bones.points.append(to_point(a))
                bones.points.append(to_point(b))

        marker_array.markers.append(joints)
        marker_array.markers.append(bones)
        self.marker_pub.publish(marker_array)

    def _publish_annotated(self, name, frame_bgr, kpts, header):
        h, w = frame_bgr.shape[:2]
        # Build pixel points only for detected landmarks. When no hand is found
        # (e.g. the first frames after a video loop), kpts is all-NaN — skip
        # those so we never feed NaN to int()/cv2.
        pts = {
            p: (int(round(kpts[p, 0])), int(round(kpts[p, 1])))
            for p in range(N_LANDMARKS)
            if not np.isnan(kpts[p, 0])
        }
        for a, b in HAND_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame_bgr, pts[a], pts[b], (255, 255, 255), 2)
        for p in pts.values():
            cv2.circle(frame_bgr, p, 3, (0, 0, 255), -1)

        img = Image()
        img.header = header
        img.height = h
        img.width = w
        img.encoding = "bgr8"
        img.is_bigendian = 0
        img.step = w * 3
        img.data = np.ascontiguousarray(frame_bgr).tobytes()
        self.annotated_pubs[name].publish(img)

    def shutdown(self):
        for d in self.detectors:
            d.close()


def main(args=None):
    rclpy.init(args=args)
    node = HandPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
