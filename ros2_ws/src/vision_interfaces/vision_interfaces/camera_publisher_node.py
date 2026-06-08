#!/usr/bin/env python3

"""
Camera publisher node for vision_interfaces.

Opens one OpenCV ``VideoCapture`` per camera — backed either by a real device
(webcam / V4L index) or by a video file — and publishes, for every camera:

    <name>/image_raw    sensor_msgs/Image      (bgr8)
    <name>/camera_info  sensor_msgs/CameraInfo

This mirrors the topic/format contract of the camera_s3 driver so any
downstream consumer (e.g. handpose_estimation) can subscribe uniformly,
regardless of whether the frames come from a live camera or a recorded clip.

All cameras are sampled on a single timer tick and stamped with the *same*
timestamp, so a downstream ApproximateTimeSynchronizer pairs them cleanly.
"""

import os

import cv2
import numpy as np
import rclpy
from camera_info_manager import CameraInfoManager
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__("camera_publisher_node")

        # --- parameters -----------------------------------------------------
        # "video" reads from files in `video_paths`; "camera" opens the V4L
        # device indices/paths in `camera_devices`.
        self.declare_parameter("source_type", "video")
        self.declare_parameter(
            "video_paths",
            ["/workspace/media/cam0_test.mp4", "/workspace/media/cam1_test.mp4"],
        )
        self.declare_parameter("camera_devices", ["0", "1"])
        self.declare_parameter("camera_names", ["camera0", "camera1"])
        self.declare_parameter(
            "camera_info_urls",
            [
                "package://vision_interfaces/config/camera_info/camera0.yaml",
                "package://vision_interfaces/config/camera_info/camera1.yaml",
            ],
        )
        self.declare_parameter("frame_rate", 30.0)
        # The original handpose3d calibration was done on a centre-cropped
        # square frame. Center-crop each frame to a square then (optionally)
        # resize so the published image matches the camera_info intrinsics.
        self.declare_parameter("crop_square", True)
        self.declare_parameter("output_size", 720)
        # Capture resolution requested from live devices (ignored for files).
        self.declare_parameter("capture_width", 1280)
        self.declare_parameter("capture_height", 720)
        # Loop video files when they reach EOF (no effect for live cameras).
        self.declare_parameter("loop", True)

        self.source_type = self.get_parameter("source_type").value
        video_paths = list(self.get_parameter("video_paths").value)
        camera_devices = list(self.get_parameter("camera_devices").value)
        self.camera_names = list(self.get_parameter("camera_names").value)
        camera_info_urls = list(self.get_parameter("camera_info_urls").value)
        frame_rate = float(self.get_parameter("frame_rate").value)
        self.crop_square = bool(self.get_parameter("crop_square").value)
        self.output_size = int(self.get_parameter("output_size").value)
        self.capture_width = int(self.get_parameter("capture_width").value)
        self.capture_height = int(self.get_parameter("capture_height").value)
        self.loop = bool(self.get_parameter("loop").value)

        # Pick the per-camera source list according to source_type.
        if self.source_type == "video":
            sources = video_paths
        elif self.source_type == "camera":
            sources = camera_devices
        else:
            raise ValueError(
                f"source_type must be 'video' or 'camera', got '{self.source_type}'"
            )

        n = len(self.camera_names)
        if len(camera_info_urls) == 1 and n > 1:
            camera_info_urls = camera_info_urls * n
        if len(sources) != n:
            raise ValueError(
                f"Number of sources ({len(sources)}) does not match number of "
                f"camera_names ({n})."
            )

        # --- open captures + publishers ------------------------------------
        self.captures = []
        self.image_pubs = []
        self.info_pubs = []
        self.camera_info_managers = []

        for i, name in enumerate(self.camera_names):
            cap = self._open_capture(sources[i])
            self.captures.append(cap)

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
            cim = CameraInfoManager(
                self, cname=name, url=camera_info_urls[i], namespace=name
            )
            cim.loadCameraInfo()
            self.camera_info_managers.append(cim)

        self.timer = self.create_timer(1.0 / frame_rate, self._tick)
        self.get_logger().info(
            f"Publishing {n} camera stream(s) {self.camera_names} "
            f"from {self.source_type} sources at {frame_rate:.1f} fps"
        )

    def _open_capture(self, source):
        if self.source_type == "video":
            if not os.path.exists(source):
                self.get_logger().error(f"Video file not found: {source}")
            cap = cv2.VideoCapture(source)
        else:
            # Numeric strings -> device index, otherwise treat as a device path.
            dev = int(source) if str(source).isdigit() else source
            cap = cv2.VideoCapture(dev)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        if not cap.isOpened():
            self.get_logger().error(f"Failed to open capture source: {source}")
        return cap

    def _process(self, frame):
        """Center-crop to a square and resize to output_size (matches calibration)."""
        if self.crop_square:
            h, w = frame.shape[:2]
            side = min(h, w)
            y0 = (h - side) // 2
            x0 = (w - side) // 2
            frame = frame[y0 : y0 + side, x0 : x0 + side]
            if self.output_size and side != self.output_size:
                frame = cv2.resize(
                    frame, (self.output_size, self.output_size),
                    interpolation=cv2.INTER_AREA,
                )
        return np.ascontiguousarray(frame)

    def _tick(self):
        # One shared timestamp for all cameras this tick so downstream
        # time-synchronization pairs the frames.
        stamp = self.get_clock().now().to_msg()

        for i, name in enumerate(self.camera_names):
            cap = self.captures[i]
            ret, frame = cap.read()
            if not ret:
                if self.source_type == "video" and self.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret:
                    self.get_logger().warn(
                        f"{name}: no frame (end of stream?)", throttle_duration_sec=5.0
                    )
                    continue

            frame = self._process(frame)
            h, w = frame.shape[:2]

            img = Image()
            img.header.stamp = stamp
            img.header.frame_id = name
            img.height = h
            img.width = w
            img.encoding = "bgr8"
            img.is_bigendian = 0
            img.step = w * 3
            img.data = frame.tobytes()
            self.image_pubs[i].publish(img)

            info = self.camera_info_managers[i].getCameraInfo()
            info.header.stamp = stamp
            info.header.frame_id = name
            self.info_pubs[i].publish(info)

    def shutdown(self):
        for cap in self.captures:
            cap.release()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
