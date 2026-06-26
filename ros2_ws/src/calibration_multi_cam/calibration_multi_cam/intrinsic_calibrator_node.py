"""Intrinsic calibration node (stage 1 of 2).

    ros2 launch calibration_multi_cam intrinsic.launch.py

Subscribes to every camera *independently* (no synchronization needed) and lets
each camera accumulate full-frame views of the AprilGrid on its own. Move the
board so it fills each camera's frame at several distances/angles. Then:

    ros2 service call /calibration_intrinsic/calibrate std_srvs/srv/Trigger {}

runs cv2.calibrateCamera per camera (4-param radtan) and writes `intrinsics_file`.
"""
from __future__ import annotations

import os

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from calibration_multi_cam.intrinsics import calibrate_intrinsics
from calibration_multi_cam.target import AprilGridTarget
from calibration_multi_cam.view_buffer import MaximinViewBuffer, corner_features


class IntrinsicCalibratorNode(Node):
    def __init__(self):
        super().__init__("calibration_intrinsic")

        self.camera_names = list(
            self.declare_parameter("camera_names", ["camera0", "camera1"]).value
        )
        topics = {}
        for cam in self.camera_names:
            topics[cam] = self.declare_parameter(f"{cam}.topic", f"/{cam}/image_raw").value

        target_params = {
            "type": self.declare_parameter("target.type", "aprilgrid").value,
            "family": self.declare_parameter("target.family", "36h11").value,
            "tag_cols": self.declare_parameter("target.tag_cols", 6).value,
            "tag_rows": self.declare_parameter("target.tag_rows", 6).value,
            "tag_size": self.declare_parameter("target.tag_size", 0.03).value,
            "tag_spacing": self.declare_parameter("target.tag_spacing", 0.333).value,
            "border_bits": self.declare_parameter("target.border_bits", 2).value,
        }
        self.target = AprilGridTarget.from_params(target_params)

        self.min_corners = int(self.declare_parameter("min_corners_per_camera", 8).value)
        self.novelty_px = float(self.declare_parameter("novelty_min_pixel_motion", 12.0).value)
        self.min_views = int(self.declare_parameter("min_views_per_camera", 20).value)
        # keep-most-informative cap per camera (0 = unlimited)
        self.max_views = int(self.declare_parameter("max_views_per_camera", 80).value)
        status_period = float(self.declare_parameter("status_period_sec", 3.0).value)
        self.intrinsics_file = self.declare_parameter(
            "intrinsics_file",
            "/workspace/ros2_ws/src/calibration_multi_cam/config/intrinsics.yaml",
        ).value

        self.bridge = CvBridge()
        # per camera: keep-most-informative buffer of (pids, pixels) views
        self.obs = {n: MaximinViewBuffer(self.max_views) for n in self.camera_names}
        self.resolution = {}                                   # name -> (w, h)
        self._last_corners = {}                                # name -> {pid: (x, y)}
        self._frames = {n: 0 for n in self.camera_names}       # images received
        self._last_detected = {n: 0 for n in self.camera_names}  # corners last frame

        self.subs = []
        for cam in self.camera_names:
            self.subs.append(self.create_subscription(
                Image, topics[cam],
                lambda msg, n=cam: self._on_image(msg, n),
                qos_profile_sensor_data,
            ))
        self.calibrate_srv = self.create_service(Trigger, "~/calibrate", self._on_calibrate)
        self.status_timer = self.create_timer(status_period, self._log_status)
        self.get_logger().info(
            f"Intrinsic calibrator up. cameras={self.camera_names}, target={self.target}. "
            f"Fill each camera's frame with the board; call ~/calibrate when done."
        )

    def _on_image(self, msg, name):
        self._frames[name] += 1
        if name not in self.resolution:
            self.resolution[name] = (int(msg.width), int(msg.height))
            self.get_logger().info(
                f"{name}: first image received ({msg.width}x{msg.height}, "
                f"encoding={msg.encoding})"
            )
        try:
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"{name}: image conversion failed: {exc}")
            return
        pids, pts = self.target.detect(gray)
        self._last_detected[name] = int(pids.size)
        if pids.size < self.min_corners:
            return
        if not self._is_novel(name, pids, pts):
            return
        w, h = self.resolution[name]
        feat = corner_features(pids, pts, w, h)
        self.obs[name].add((pids, pts), feat)
        self._last_corners[name] = {int(i): tuple(p) for i, p in zip(pids, pts)}

    def _is_novel(self, name, pids, pts):
        prev = self._last_corners.get(name)
        if prev is None:
            return True
        shared = [(p, prev[int(i)]) for i, p in zip(pids, pts) if int(i) in prev]
        if not shared:
            return True
        disp = np.mean([np.hypot(p[0] - q[0], p[1] - q[1]) for p, q in shared])
        return disp >= self.novelty_px

    def _log_status(self):
        counts = {n: len(self.obs[n]) for n in self.camera_names}
        ready = all(c >= self.min_views for c in counts.values())
        frames = {n: self._frames[n] for n in self.camera_names}
        detected = {n: self._last_detected[n] for n in self.camera_names}
        self.get_logger().info(
            f"views per camera={counts}/{self.min_views} | ready={ready} | "
            f"frames={frames} | corners_last_frame={detected} "
            f"(need >={self.min_corners})"
        )

    def _on_calibrate(self, request, response):
        result = {"cameras": {}}
        msgs = []
        ok_all = True
        for name in self.camera_names:
            n = len(self.obs[name])
            if n < self.min_views:
                ok_all = False
                msgs.append(f"{name}: only {n}/{self.min_views} views (skipped)")
                continue
            if name not in self.resolution:
                ok_all = False
                msgs.append(f"{name}: no resolution seen")
                continue
            try:
                r = calibrate_intrinsics(self.target.object_points, self.obs[name].items,
                                         self.resolution[name])
            except Exception as exc:  # noqa: BLE001
                ok_all = False
                msgs.append(f"{name}: calibration failed ({exc})")
                continue
            result["cameras"][name] = r
            rej = r.pop("num_rejected", 0)   # diagnostic only; keep out of the file
            r.pop("num_corners", None)       # ditto
            msgs.append(
                f"{name}: rms={r['reproj_rms']:.3f}px "
                f"({r['num_views']} views, {rej} corners rejected)"
            )
            pv = r.pop("per_view_rms", [])  # diagnostic only; keep out of the file
            if pv:
                worst = ", ".join(f"{e:.2f}" for e in pv[:5])
                self.get_logger().info(
                    f"{name}: per-view rms median={np.median(pv):.3f} "
                    f"min={pv[-1]:.3f} max={pv[0]:.3f} | worst5=[{worst}]"
                )

        if not result["cameras"]:
            response.success = False
            response.message = "No cameras calibrated. " + "; ".join(msgs)
            self.get_logger().error(response.message)
            return response

        try:
            os.makedirs(os.path.dirname(self.intrinsics_file), exist_ok=True)
            with open(self.intrinsics_file, "w") as fh:
                yaml.safe_dump(result, fh, default_flow_style=None, sort_keys=False)
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f"Failed to write {self.intrinsics_file}: {exc}"
            self.get_logger().error(response.message)
            return response

        summary = f"Wrote intrinsics for {list(result['cameras'])} to {self.intrinsics_file}. " \
                  + "; ".join(msgs)
        self.get_logger().info(summary)
        response.success = ok_all
        response.message = summary
        return response


def main(args=None):
    rclpy.init(args=args)
    node = IntrinsicCalibratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
