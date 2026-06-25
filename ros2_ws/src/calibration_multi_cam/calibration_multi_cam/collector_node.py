"""Collector node: subscribe to N camera topics, synchronize, detect the
AprilGrid, and accumulate target views for calibration.

    ros2 launch calibration_multi_cam calibrate.launch.py

Topics (per camera <c>, from the central YAML): subscribes to <c>.topic
(sensor_msgs/Image). Cameras stream continuously; a *view* is accepted only
when >= `min_cameras_per_view` cameras detect the target and the board has
moved >= `novelty_min_pixel_motion` pixels since the last accepted view (a
lightweight stand-in for kalibr's information-gain view selection).

Services:
    ~/calibrate  (std_srvs/Trigger)  -> persist collected views to
                                        `observations_file` and report readiness.

NOTE: this node performs collection + persistence. The solver that turns the
saved views into `result_file` (intrinsics + extrinsics) is a separate module
wired in next; for now `~/calibrate` saves the dataset and reports a summary.
"""
from __future__ import annotations

import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from calibration_multi_cam.observations import ObservationDatabase
from calibration_multi_cam.target import AprilGridTarget


class CollectorNode(Node):
    def __init__(self):
        super().__init__("calibration_collector")

        # ---- parameters ---------------------------------------------------
        self.camera_names = list(
            self.declare_parameter("camera_names", ["cam0", "cam1"]).value
        )
        self.cam_cfg = {}
        topics = []
        for cam in self.camera_names:
            topic = self.declare_parameter(f"{cam}.topic", f"/{cam}/image_raw").value
            model = self.declare_parameter(f"{cam}.model", "pinhole-radtan").value
            opt_in = self.declare_parameter(f"{cam}.optimize_intrinsics", True).value
            info_topic = self.declare_parameter(f"{cam}.camera_info_topic", "").value
            self.cam_cfg[cam] = {
                "topic": topic, "model": model,
                "optimize_intrinsics": bool(opt_in), "camera_info_topic": info_topic,
            }
            topics.append(topic)

        target_params = {
            "type": self.declare_parameter("target.type", "aprilgrid").value,
            "family": self.declare_parameter("target.family", "36h11").value,
            "tag_cols": self.declare_parameter("target.tag_cols", 6).value,
            "tag_rows": self.declare_parameter("target.tag_rows", 6).value,
            "tag_size": self.declare_parameter("target.tag_size", 0.03).value,
            "tag_spacing": self.declare_parameter("target.tag_spacing", 0.333).value,
        }
        self.target = AprilGridTarget.from_params(target_params)
        self.target_params = target_params

        self.sync_slop = float(self.declare_parameter("sync_slop", 0.02).value)
        self.sync_queue = int(self.declare_parameter("sync_queue_size", 20).value)
        self.min_corners = int(self.declare_parameter("min_corners_per_camera", 8).value)
        self.min_cams = int(self.declare_parameter("min_cameras_per_view", 2).value)
        self.novelty_px = float(self.declare_parameter("novelty_min_pixel_motion", 12.0).value)
        self.min_views = int(self.declare_parameter("min_views", 30).value)
        status_period = float(self.declare_parameter("status_period_sec", 3.0).value)
        self.world_frame = self.declare_parameter("world_frame", "cam0").value
        self.observations_file = self.declare_parameter(
            "observations_file",
            "/workspace/ros2_ws/src/calibration_multi_cam/config/observations.npz",
        ).value

        # ---- state --------------------------------------------------------
        self.bridge = CvBridge()
        self.obsdb = ObservationDatabase(self.camera_names)
        self.resolutions = {}                 # cam -> (width, height)
        self._last_corners = {}               # cam -> {point_id: (x, y)} of last accepted view

        # ---- subscriptions + synchronizer --------------------------------
        self.subs = [
            Subscriber(self, Image, t, qos_profile=qos_profile_sensor_data)
            for t in topics
        ]
        self.sync = ApproximateTimeSynchronizer(
            self.subs, queue_size=self.sync_queue, slop=self.sync_slop
        )
        self.sync.registerCallback(self._on_images)

        # ---- services + status timer -------------------------------------
        self.calibrate_srv = self.create_service(Trigger, "~/calibrate", self._on_calibrate)
        self.status_timer = self.create_timer(status_period, self._log_status)

        self.get_logger().info(
            f"Collector up. cameras={self.camera_names}, target={self.target}, "
            f"sync_slop={self.sync_slop}s. Move the board through the shared FoV; "
            f"call ~/calibrate when coverage looks good."
        )

    # ------------------------------------------------------------------ #
    # Synchronized image callback
    # ------------------------------------------------------------------ #
    def _on_images(self, *msgs):
        detections = {}
        for cam, msg in zip(self.camera_names, msgs):
            if cam not in self.resolutions:
                self.resolutions[cam] = (int(msg.width), int(msg.height))
            try:
                gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"{cam}: image conversion failed: {exc}")
                continue
            pids, pts = self.target.detect(gray)
            if pids.size >= self.min_corners:
                detections[cam] = (pids, pts)

        if len(detections) < self.min_cams:
            return  # not enough cameras see the target in this instant

        if not self._is_novel(detections):
            return  # too similar to the last accepted view

        stamp = self._stamp_seconds(msgs[0])
        self.obsdb.add_view(stamp, detections)
        for cam, (pids, pts) in detections.items():
            self._last_corners[cam] = {int(i): tuple(p) for i, p in zip(pids, pts)}

    def _is_novel(self, detections):
        """Accept if any camera is new or its corners moved >= novelty threshold."""
        for cam, (pids, pts) in detections.items():
            prev = self._last_corners.get(cam)
            if prev is None:
                return True
            shared = [(p, prev[int(i)]) for i, p in zip(pids, pts) if int(i) in prev]
            if not shared:
                return True
            disp = np.mean([np.hypot(p[0] - q[0], p[1] - q[1]) for p, q in shared])
            if disp >= self.novelty_px:
                return True
        return False

    @staticmethod
    def _stamp_seconds(msg):
        s = msg.header.stamp
        return float(s.sec) + float(s.nanosec) * 1e-9

    # ------------------------------------------------------------------ #
    # Status + calibrate trigger
    # ------------------------------------------------------------------ #
    def _log_status(self):
        n = self.obsdb.num_views
        per_cam = self.obsdb.per_camera_view_count()
        pairs = self.obsdb.pair_coobservation_count()
        connected = self.obsdb.is_connected() if n else False
        pair_str = ", ".join(f"{a}-{b}:{c}" for (a, b), c in sorted(pairs.items())) or "none"
        self.get_logger().info(
            f"views={n}/{self.min_views} | per-camera={per_cam} | "
            f"pairs[{pair_str}] | rig_connected={connected}"
        )
        if n and not connected:
            self.get_logger().warn(
                "Cameras are NOT yet linked by shared target views. Every adjacent "
                "pair needs simultaneous views of the board, or the extrinsics "
                "cannot be chained to cam0."
            )

    def _on_calibrate(self, request, response):
        n = self.obsdb.num_views
        if n == 0:
            response.success = False
            response.message = "No views collected yet."
            return response

        meta = {
            "camera_names": self.camera_names,
            "camera_config": self.cam_cfg,
            "target": self.target_params,
            "resolutions": self.resolutions,
            "world_frame": self.world_frame,
        }
        try:
            self.obsdb.save(self.observations_file, meta=meta)
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f"Failed to save observations: {exc}"
            self.get_logger().error(response.message)
            return response

        connected = self.obsdb.is_connected()
        per_cam = self.obsdb.per_camera_view_count()
        msg = (
            f"Saved {n} views to {self.observations_file} "
            f"(per-camera={per_cam}, rig_connected={connected}). "
            "Run the solver to produce result_file."
        )
        self.get_logger().info(msg)
        if not connected:
            self.get_logger().warn(
                "Rig is NOT connected; the solver will not be able to chain all "
                "extrinsics to cam0. Collect more overlapping views."
            )
        response.success = connected
        response.message = msg
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CollectorNode()
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
