#!/usr/bin/env python3

"""
CityU stereo hand pose dataset publisher.

Replays the CityU "stereo hand pose benchmark" (Zhang et al., ICIP 2017) as ROS
2 camera streams, mirroring the topic/format contract of ``vision_interfaces``
so any downstream consumer (e.g. handpose_estimation) can subscribe uniformly:

    <name>/image_raw    sensor_msgs/Image      (bgr8)
    <name>/camera_info  sensor_msgs/CameraInfo

A sequence folder (e.g. ``B1Counting``) holds 6000 PNGs: 1500 frames x 4 image
streams (``BB_left_``, ``BB_right_``, ``SK_color_``, ``SK_depth_``). To keep RAM
flat regardless of sequence length, frames are NOT preloaded -- each timer tick
reads only the current frame of each published stream from disk via
``cv2.imread`` (one ~900 KB image per stream in memory at a time).

All published streams are sampled on a single timer tick and stamped with the
*same* timestamp, so a downstream ApproximateTimeSynchronizer pairs them.
"""

import glob
import os
import re

import cv2
import numpy as np
import rclpy
from camera_info_manager import CameraInfoManager
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class CityuDataPublisherNode(Node):
    def __init__(self):
        super().__init__("cityu_data_publisher_node")

        # --- parameters -----------------------------------------------------
        # Root of the extracted dataset (the folder containing images/ and
        # labels/). Default assumes the repo bind-mount at /workspace.
        self.declare_parameter(
            "dataset_root",
            "/workspace/ros2_ws/data_set/stereo hand pose data set",
        )
        # Which sequence folder under images/ to replay (e.g. "B1Counting").
        self.declare_parameter("sequence", "B1Counting")
        # Which image streams to publish, the topic namespace for each, and the
        # camera_info to attach. The defaults publish the Bumblebee2 rectified
        # stereo pair as camera0/camera1 so this is drop-in for the stereo
        # handpose pipeline.
        self.declare_parameter("image_prefixes", ["BB_left", "BB_right"])
        self.declare_parameter("camera_names", ["camera0", "camera1"])
        self.declare_parameter(
            "camera_info_urls",
            [
                "package://cityu_data_interface/config/camera_info/bb_left.yaml",
                "package://cityu_data_interface/config/camera_info/bb_right.yaml",
            ],
        )
        self.declare_parameter("frame_rate", 30.0)
        # Loop back to the first frame at end of sequence.
        self.declare_parameter("loop", True)
        # Playback window over the sequence: skip to start_frame, then play at
        # most num_frames frames (num_frames < 0 means "to the end").
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("num_frames", -1)

        self.dataset_root = str(self.get_parameter("dataset_root").value)
        self.sequence = str(self.get_parameter("sequence").value)
        self.image_prefixes = list(self.get_parameter("image_prefixes").value)
        self.camera_names = list(self.get_parameter("camera_names").value)
        camera_info_urls = list(self.get_parameter("camera_info_urls").value)
        frame_rate = float(self.get_parameter("frame_rate").value)
        self.loop = bool(self.get_parameter("loop").value)
        start_frame = int(self.get_parameter("start_frame").value)
        num_frames = int(self.get_parameter("num_frames").value)

        n = len(self.camera_names)
        if len(self.image_prefixes) != n:
            raise ValueError(
                f"image_prefixes ({len(self.image_prefixes)}) must match "
                f"camera_names ({n})."
            )
        if len(camera_info_urls) == 1 and n > 1:
            camera_info_urls = camera_info_urls * n
        if len(camera_info_urls) != n:
            raise ValueError(
                f"camera_info_urls ({len(camera_info_urls)}) must match "
                f"camera_names ({n})."
            )

        self.image_dir = os.path.join(self.dataset_root, "images", self.sequence)
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Sequence folder not found: {self.image_dir}")

        # Discover the available frame indices from the first stream and apply
        # the playback window. All streams share the same 0..N-1 indexing.
        self.frame_indices = self._discover_frames(self.image_prefixes[0])
        if not self.frame_indices:
            raise FileNotFoundError(
                f"No '{self.image_prefixes[0]}_*.png' frames in {self.image_dir}"
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

        # --- publishers + camera_info --------------------------------------
        self.image_pubs = []
        self.info_pubs = []
        self.camera_info_managers = []
        for i, name in enumerate(self.camera_names):
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
            f"Replaying CityU sequence '{self.sequence}' "
            f"({len(self.frame_indices)} frames) as streams "
            f"{list(zip(self.camera_names, self.image_prefixes))} "
            f"at {frame_rate:.1f} fps (loop={self.loop})"
        )

    def _discover_frames(self, prefix):
        """Sorted list of integer frame indices available for ``prefix``."""
        pattern = os.path.join(self.image_dir, f"{prefix}_*.png")
        rx = re.compile(rf"{re.escape(prefix)}_(\d+)\.png$")
        indices = []
        for path in glob.glob(pattern):
            m = rx.search(os.path.basename(path))
            if m:
                indices.append(int(m.group(1)))
        return sorted(indices)

    def _frame_path(self, prefix, frame_idx):
        return os.path.join(self.image_dir, f"{prefix}_{frame_idx}.png")

    def _tick(self):
        if self.cursor >= len(self.frame_indices):
            if not self.loop:
                self.get_logger().info(
                    "Reached end of sequence; stopping playback.",
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

        for i, name in enumerate(self.camera_names):
            path = self._frame_path(self.image_prefixes[i], frame_idx)
            # Lazy, one-frame-at-a-time read keeps RAM flat over long sequences.
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
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


def main(args=None):
    rclpy.init(args=args)
    node = CityuDataPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
