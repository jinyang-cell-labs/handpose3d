"""Extrinsic calibration node (stage 2 of 2).

    ros2 launch calibration_multi_cam extrinsic.launch.py

Loads `intrinsics_file` (from the intrinsic stage), subscribes to all cameras
and time-synchronizes them, and accumulates views where >= 2 cameras see the
AprilGrid simultaneously. Then:

    ros2 service call /calibration_extrinsic/calibrate std_srvs/srv/Trigger {}

runs PnP -> covisibility-graph chaining -> bundle adjustment (intrinsics fixed,
camera0 = world) and writes `extrinsics_file`.
"""
from __future__ import annotations

import os

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from calibration_multi_cam import se3
from calibration_multi_cam.bundle_adjust import bundle_adjust, per_camera_rms
from calibration_multi_cam.extrinsics import init_extrinsics
from calibration_multi_cam.observations import ObservationDatabase
from calibration_multi_cam.target import AprilGridTarget


class ExtrinsicCalibratorNode(Node):
    def __init__(self):
        super().__init__("calibration_extrinsic")

        self.camera_names = list(
            self.declare_parameter("camera_names", ["camera0", "camera1"]).value
        )
        topics = [self.declare_parameter(f"{c}.topic", f"/{c}/image_raw").value
                  for c in self.camera_names]

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

        self.sync_slop = float(self.declare_parameter("sync_slop", 0.02).value)
        self.sync_queue = int(self.declare_parameter("sync_queue_size", 20).value)
        self.min_corners = int(self.declare_parameter("min_corners_per_camera", 8).value)
        self.min_cams = int(self.declare_parameter("min_cameras_per_view", 2).value)
        self.novelty_px = float(self.declare_parameter("novelty_min_pixel_motion", 12.0).value)
        self.min_views = int(self.declare_parameter("min_views", 30).value)
        # keep-most-informative cap on retained synchronized views (0 = unlimited)
        self.max_views = int(self.declare_parameter("max_views", 150).value)
        status_period = float(self.declare_parameter("status_period_sec", 3.0).value)
        self.robust_loss = self.declare_parameter("robust_loss", "huber").value
        self.loss_scale = float(self.declare_parameter("robust_loss_scale", 1.0).value)
        self.world_frame = self.declare_parameter("world_frame", "camera0").value
        self.intrinsics_file = self.declare_parameter(
            "intrinsics_file",
            "/workspace/ros2_ws/src/calibration_multi_cam/config/intrinsics.yaml",
        ).value
        self.extrinsics_file = self.declare_parameter(
            "extrinsics_file",
            "/workspace/ros2_ws/src/calibration_multi_cam/config/extrinsics.yaml",
        ).value

        # ---- load intrinsics (required) ----------------------------------
        self.intrinsics = self._load_intrinsics()

        self.bridge = CvBridge()
        # per-camera resolutions (from intrinsics) normalize the diversity features
        resolutions = {n: self.intrinsics[n].get("resolution") for n in self.camera_names}
        self.obsdb = ObservationDatabase(self.camera_names, capacity=self.max_views,
                                         resolutions=resolutions)
        self._last_corners = {}

        self.subs = [Subscriber(self, Image, t, qos_profile=qos_profile_sensor_data)
                     for t in topics]
        self.sync = ApproximateTimeSynchronizer(
            self.subs, queue_size=self.sync_queue, slop=self.sync_slop)
        self.sync.registerCallback(self._on_images)

        self.calibrate_srv = self.create_service(Trigger, "~/calibrate", self._on_calibrate)
        self.status_timer = self.create_timer(status_period, self._log_status)
        self.get_logger().info(
            f"Extrinsic calibrator up. cameras={self.camera_names}. Loaded intrinsics "
            f"for {list(self.intrinsics)}. Move the board across overlapping views; "
            f"call ~/calibrate when the rig is connected."
        )

    def _load_intrinsics(self):
        if not os.path.isfile(self.intrinsics_file):
            raise RuntimeError(
                f"intrinsics_file not found: {self.intrinsics_file}. Run the intrinsic "
                f"calibration stage first."
            )
        with open(self.intrinsics_file, "r") as fh:
            data = yaml.safe_load(fh)
        cams = data.get("cameras", {})
        missing = [n for n in self.camera_names if n not in cams]
        if missing:
            raise RuntimeError(f"intrinsics_file is missing cameras: {missing}")
        return cams

    def _on_images(self, *msgs):
        detections = {}
        for cam, msg in zip(self.camera_names, msgs):
            try:
                gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"{cam}: image conversion failed: {exc}")
                continue
            pids, pts = self.target.detect(gray)
            if pids.size >= self.min_corners:
                detections[cam] = (pids, pts)

        if len(detections) < self.min_cams or not self._is_novel(detections):
            return
        stamp = float(msgs[0].header.stamp.sec) + float(msgs[0].header.stamp.nanosec) * 1e-9
        self.obsdb.add_view(stamp, detections)
        for cam, (pids, pts) in detections.items():
            self._last_corners[cam] = {int(i): tuple(p) for i, p in zip(pids, pts)}

    def _is_novel(self, detections):
        for cam, (pids, pts) in detections.items():
            prev = self._last_corners.get(cam)
            if prev is None:
                return True
            shared = [(p, prev[int(i)]) for i, p in zip(pids, pts) if int(i) in prev]
            if not shared:
                return True
            if np.mean([np.hypot(p[0] - q[0], p[1] - q[1]) for p, q in shared]) >= self.novelty_px:
                return True
        return False

    def _log_status(self):
        n = self.obsdb.num_views
        pairs = self.obsdb.pair_coobservation_count()
        connected = self.obsdb.is_connected() if n else False
        pair_str = ", ".join(f"{a}-{b}:{c}" for (a, b), c in sorted(pairs.items())) or "none"
        self.get_logger().info(
            f"views={n}/{self.min_views} | pairs[{pair_str}] | rig_connected={connected}")
        if n and not connected:
            self.get_logger().warn(
                "Cameras not yet linked by shared views; extrinsics can't be chained to camera0.")

    def _on_calibrate(self, request, response):
        if self.obsdb.num_views == 0:
            response.success = False
            response.message = "No views collected yet."
            return response
        try:
            cam_world, board_world, obs_struct, info = init_extrinsics(
                self.obsdb.views, self.camera_names, self.intrinsics,
                self.target.object_points, min_corners=self.min_corners)
            cam_world, board_world, ba_info = bundle_adjust(
                cam_world, board_world, obs_struct, self.camera_names,
                self.intrinsics, self.target.object_points,
                robust_loss=self.robust_loss, loss_scale=self.loss_scale)
            cam_rms = per_camera_rms(cam_world, board_world, obs_struct,
                                     self.camera_names, self.intrinsics,
                                     self.target.object_points)
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f"Extrinsic solve failed: {exc}"
            self.get_logger().error(response.message)
            return response

        # T_world_cam (pose of camera in world) = inverse of T_cam_world
        out = {"world_frame": self.world_frame, "cameras": {}}
        for i, name in enumerate(self.camera_names):
            T_world_cam = se3.invert_T(cam_world[i])
            out["cameras"][name] = {"T_world_cam": T_world_cam.tolist()}
        try:
            os.makedirs(os.path.dirname(self.extrinsics_file), exist_ok=True)
            with open(self.extrinsics_file, "w") as fh:
                yaml.safe_dump(out, fh, default_flow_style=None, sort_keys=False)
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f"Failed to write {self.extrinsics_file}: {exc}"
            self.get_logger().error(response.message)
            return response

        summary = (
            f"Extrinsics written to {self.extrinsics_file}. "
            f"BA rms {ba_info['rms_before']:.3f}->{ba_info['rms_after']:.3f}px over "
            f"{info['num_views_used']} views; per-camera rms="
            f"{ {k: round(v, 3) for k, v in cam_rms.items()} }; tree={info['tree_edges']}")
        self.get_logger().info(summary)
        response.success = True
        response.message = summary
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ExtrinsicCalibratorNode()
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
