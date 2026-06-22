#!/usr/bin/env python3

"""
HOT3D (Project Aria) dataset publisher.

Replays a HOT3D clip (e.g. ``clip-001849``) as ROS 2 camera streams, mirroring
the topic/format contract of ``vision_interfaces`` / ``cityu_data_interface`` so
any downstream consumer can subscribe uniformly:

    <name>/image_raw    sensor_msgs/Image      (mono8 or bgr8)
    <name>/camera_info  sensor_msgs/CameraInfo (FISHEYE624 intrinsics)

plus, per frame, the camera pose in the world frame on /tf:

    world -> <frame_id>   geometry_msgs/TransformStamped

A clip folder holds, per frame index ``NNNNNN``::

    NNNNNN.image_1201-1.jpg   left  SLAM fisheye, 640x480 grayscale
    NNNNNN.image_1201-2.jpg   right SLAM fisheye, 640x480 grayscale
    NNNNNN.image_214-1.jpg    middle RGB camera, 1408x1408
    NNNNNN.cameras.json       per-camera intrinsics (static) + extrinsics
    NNNNNN.{hands,objects,hand_crops}.json   annotations (not published here)

The Aria cameras use Meta's FISHEYE624 model (15 params: focal, cx, cy, 6 radial,
2 tangential, 4 thin-prism). No standard ROS distortion model captures this, so
camera_info is published faithfully with ``distortion_model = "FISHEYE624"`` and
the 12 distortion coefficients in ``D``; ``K``/``P`` carry focal + principal point.

To keep RAM flat regardless of clip length, frames are NOT preloaded -- each timer
tick reads only the current frame of each published stream from disk via
``cv2.imread``. All streams sampled on a tick share the *same* timestamp so a
downstream ApproximateTimeSynchronizer pairs them.
"""

import glob
import json
import os
import re

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster


class Hot3DataPublisherNode(Node):
    def __init__(self):
        super().__init__("hot3_data_publisher_node")

        # --- parameters -----------------------------------------------------
        # Root folder holding clip subfolders. The repo is bind-mounted at
        # /workspace inside the container.
        self.declare_parameter("dataset_root", "/workspace/ros2_ws/recordings")
        # Which clip subfolder to replay (e.g. "clip-001849").
        self.declare_parameter("clip", "clip-001849")
        # HOT3D stream labels to publish, the topic namespace for each, the ROS
        # image encoding, and the TF frame_id. Defaults publish all three Aria
        # cameras: the two SLAM fisheye (mono) + the middle RGB camera.
        self.declare_parameter("stream_labels", ["1201-1", "1201-2", "214-1"])
        self.declare_parameter("camera_names", ["camera0", "camera1", "camera2"])
        self.declare_parameter("encodings", ["mono8", "mono8", "bgr8"])
        self.declare_parameter(
            "frame_ids", ["camera_1201-1", "camera_1201-2", "camera_214-1"]
        )
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("frame_rate", 30.0)
        self.declare_parameter("loop", True)
        # Playback window: skip to start_frame, then play at most num_frames
        # frames (num_frames < 0 means "to the end").
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("num_frames", -1)

        dataset_root = str(self.get_parameter("dataset_root").value)
        clip = str(self.get_parameter("clip").value)
        self.stream_labels = list(self.get_parameter("stream_labels").value)
        self.camera_names = list(self.get_parameter("camera_names").value)
        self.encodings = list(self.get_parameter("encodings").value)
        self.frame_ids = list(self.get_parameter("frame_ids").value)
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        frame_rate = float(self.get_parameter("frame_rate").value)
        self.loop = bool(self.get_parameter("loop").value)
        start_frame = int(self.get_parameter("start_frame").value)
        num_frames = int(self.get_parameter("num_frames").value)

        n = len(self.stream_labels)
        for label, value in (
            ("camera_names", self.camera_names),
            ("encodings", self.encodings),
            ("frame_ids", self.frame_ids),
        ):
            if len(value) != n:
                raise ValueError(
                    f"{label} ({len(value)}) must match stream_labels ({n})."
                )

        self.clip_dir = os.path.join(dataset_root, clip)
        if not os.path.isdir(self.clip_dir):
            raise FileNotFoundError(f"Clip folder not found: {self.clip_dir}")

        # Discover frame indices from the first stream; all streams share the
        # same NNNNNN indexing. Apply the playback window.
        self.frame_indices = self._discover_frames(self.stream_labels[0])
        if not self.frame_indices:
            raise FileNotFoundError(
                f"No '*.image_{self.stream_labels[0]}.jpg' frames in "
                f"{self.clip_dir}"
            )
        self.frame_indices = [f for f in self.frame_indices if f >= start_frame]
        if num_frames >= 0:
            self.frame_indices = self.frame_indices[:num_frames]
        if not self.frame_indices:
            raise ValueError(
                f"Playback window is empty (start_frame={start_frame}, "
                f"num_frames={num_frames})."
            )
        self.cursor = 0

        # --- publishers -----------------------------------------------------
        self.image_pubs = []
        self.info_pubs = []
        for name in self.camera_names:
            self.image_pubs.append(
                self.create_publisher(
                    Image, f"{name}/image_raw", qos_profile_sensor_data
                )
            )
            self.info_pubs.append(
                self.create_publisher(
                    CameraInfo, f"{name}/camera_info", qos_profile_sensor_data
                )
            )
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.timer = self.create_timer(1.0 / frame_rate, self._tick)
        self.get_logger().info(
            f"Replaying HOT3D clip '{clip}' ({len(self.frame_indices)} frames) "
            f"as streams {list(zip(self.camera_names, self.stream_labels))} "
            f"at {frame_rate:.1f} fps (loop={self.loop}, tf={self.publish_tf})"
        )

    def _discover_frames(self, label):
        """Sorted list of integer frame indices available for ``label``."""
        pattern = os.path.join(self.clip_dir, f"*.image_{label}.jpg")
        rx = re.compile(rf"(\d+)\.image_{re.escape(label)}\.jpg$")
        indices = []
        for path in glob.glob(pattern):
            m = rx.search(os.path.basename(path))
            if m:
                indices.append(int(m.group(1)))
        return sorted(indices)

    def _image_path(self, frame_idx, label):
        return os.path.join(self.clip_dir, f"{frame_idx:06d}.image_{label}.jpg")

    def _cameras_path(self, frame_idx):
        return os.path.join(self.clip_dir, f"{frame_idx:06d}.cameras.json")

    @staticmethod
    def _build_camera_info(calib):
        """sensor_msgs/CameraInfo from a HOT3D FISHEYE624 calibration block."""
        params = calib["projection_params"]
        f, cx, cy = params[0], params[1], params[2]
        distortion = params[3:]  # 6 radial + 2 tangential + 4 thin-prism

        info = CameraInfo()
        info.width = int(calib["image_width"])
        info.height = int(calib["image_height"])
        info.distortion_model = "FISHEYE624"
        info.d = [float(x) for x in distortion]
        info.k = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [f, 0.0, cx, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _tick(self):
        if self.cursor >= len(self.frame_indices):
            if not self.loop:
                self.get_logger().info(
                    "Reached end of clip; stopping playback.",
                    throttle_duration_sec=5.0,
                )
                self.timer.cancel()
                return
            self.cursor = 0

        frame_idx = self.frame_indices[self.cursor]
        self.cursor += 1

        # One shared timestamp for all streams this tick so downstream
        # time-synchronization pairs the frames.
        stamp = self.get_clock().now().to_msg()

        # Per-frame intrinsics + extrinsics live in one sidecar JSON.
        cameras = None
        cameras_path = self._cameras_path(frame_idx)
        try:
            with open(cameras_path) as fh:
                cameras = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            self.get_logger().warn(
                f"frame {frame_idx}: cannot read {cameras_path}: {exc}",
                throttle_duration_sec=5.0,
            )

        for i, name in enumerate(self.camera_names):
            label = self.stream_labels[i]
            encoding = self.encodings[i]
            frame_id = self.frame_ids[i]

            read_flag = (
                cv2.IMREAD_GRAYSCALE if encoding == "mono8" else cv2.IMREAD_COLOR
            )
            path = self._image_path(frame_idx, label)
            frame = cv2.imread(path, read_flag)
            if frame is None:
                self.get_logger().warn(
                    f"{name}: missing/unreadable frame {path}",
                    throttle_duration_sec=5.0,
                )
                continue
            frame = np.ascontiguousarray(frame)
            h, w = frame.shape[:2]

            img = Image()
            img.header.stamp = stamp
            img.header.frame_id = frame_id
            img.height = h
            img.width = w
            img.encoding = encoding
            img.is_bigendian = 0
            img.step = w if encoding == "mono8" else w * 3
            img.data = frame.tobytes()
            self.image_pubs[i].publish(img)

            cam = cameras.get(label) if cameras else None
            if cam is None:
                continue

            info = self._build_camera_info(cam["calibration"])
            info.header.stamp = stamp
            info.header.frame_id = frame_id
            self.info_pubs[i].publish(info)

            if self.tf_broadcaster is not None:
                self._publish_tf(cam["T_world_from_camera"], frame_id, stamp)

    def _publish_tf(self, pose, frame_id, stamp):
        """Broadcast world -> frame_id from a {quaternion_wxyz, translation_xyz}."""
        qw, qx, qy, qz = pose["quaternion_wxyz"]
        tx, ty, tz = pose["translation_xyz"]

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.world_frame
        tf.child_frame_id = frame_id
        tf.transform.translation.x = float(tx)
        tf.transform.translation.y = float(ty)
        tf.transform.translation.z = float(tz)
        tf.transform.rotation.w = float(qw)
        tf.transform.rotation.x = float(qx)
        tf.transform.rotation.y = float(qy)
        tf.transform.rotation.z = float(qz)
        self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = Hot3DataPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
