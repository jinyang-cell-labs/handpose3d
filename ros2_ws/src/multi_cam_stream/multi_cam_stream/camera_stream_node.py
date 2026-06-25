#!/usr/bin/env python3
"""Multi-USB-camera publisher.

Opens one OpenCV ``VideoCapture`` per V4L device and publishes, per camera:

    <name>/image_raw    sensor_msgs/Image   (bgr8)

All cameras are grabbed on a single timer tick and stamped with the *same*
timestamp, so a downstream ApproximateTimeSynchronizer (e.g.
calibration_multi_cam) pairs the frames cleanly.

No camera_info, no file replay — this is just the live USB frontend.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

import cv2
import numpy as np


def _fourcc_str(val):
    code = int(val)
    return "".join(chr((code >> (8 * j)) & 0xFF) for j in range(4))


class CameraStreamNode(Node):
    def __init__(self):
        super().__init__("camera_stream_node")

        # --- parameters ----------------------------------------------------
        self.camera_names = list(
            self.declare_parameter("camera_names", ["cam0", "cam1"]).value
        )
        camera_devices = list(
            self.declare_parameter("camera_devices", ["0", "1"]).value
        )
        frame_rate = float(self.declare_parameter("frame_rate", 30.0).value)
        self.capture_width = int(self.declare_parameter("capture_width", 1280).value)
        self.capture_height = int(self.declare_parameter("capture_height", 720).value)
        self.fourcc = str(self.declare_parameter("fourcc", "MJPG").value)

        if len(camera_devices) != len(self.camera_names):
            raise ValueError(
                f"camera_devices ({len(camera_devices)}) must match "
                f"camera_names ({len(self.camera_names)})."
            )
        if frame_rate <= 0.0:
            self.get_logger().warn(f"frame_rate must be > 0, got {frame_rate}; using 30.0")
            frame_rate = 30.0

        # --- open captures + publishers ------------------------------------
        self.captures = []
        self.image_pubs = []
        for name, dev in zip(self.camera_names, camera_devices):
            self.captures.append(self._open_capture(name, dev))
            self.image_pubs.append(
                self.create_publisher(Image, f"{name}/image_raw", qos_profile_sensor_data)
            )

        self.timer = self.create_timer(1.0 / frame_rate, self._tick)
        self.get_logger().info(
            f"Publishing {len(self.camera_names)} USB camera stream(s) "
            f"{self.camera_names} at {frame_rate:.1f} fps"
        )

    def _open_capture(self, name, source):
        # Numeric strings -> device index, otherwise treat as a device path.
        dev = int(source) if str(source).isdigit() else source
        cap = cv2.VideoCapture(dev)

        self.get_logger().info(
            f"[{name} dev={source}] BEFORE set: backend={cap.getBackendName()} "
            f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
            f"fourcc={_fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))} "
            f"fps={cap.get(cv2.CAP_PROP_FPS):.1f}"
        )

        # Order matters for V4L2: fourcc, then resolution.
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(
            f"[{name} dev={source}] AFTER set: {actual_w}x{actual_h} "
            f"fourcc={_fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))} "
            f"fps={cap.get(cv2.CAP_PROP_FPS):.1f}"
        )
        if (actual_w, actual_h) != (self.capture_width, self.capture_height):
            self.get_logger().warn(
                f"[{name} dev={source}] RESOLUTION MISMATCH: requested "
                f"{self.capture_width}x{self.capture_height}, got {actual_w}x{actual_h} "
                f"(V4L2 fell back to a supported mode)"
            )
        if not cap.isOpened():
            self.get_logger().error(f"[{name} dev={source}] failed to open device")
        return cap

    def _tick(self):
        # Shared timestamp for all cameras so downstream sync pairs the frames.
        stamp = self.get_clock().now().to_msg()

        for name, cap, pub in zip(self.camera_names, self.captures, self.image_pubs):
            ret, frame = cap.read()
            if not ret or frame is None:
                self.get_logger().warn(
                    f"{name}: no frame", throttle_duration_sec=5.0
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
            pub.publish(img)

    def shutdown(self):
        for cap in self.captures:
            cap.release()


def main(args=None):
    rclpy.init(args=args)
    node = CameraStreamNode()
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
