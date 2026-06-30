"""Board pose node: single-camera AprilGrid pose estimation + visualization.

    ros2 launch calibration_multi_cam board_pose.launch.py

Subscribes to ONE camera's image stream (``board_pose.camera``, default the
first entry in ``camera_names``), detects the AprilGrid, and runs solvePnP using
that camera's *already-calibrated* intrinsics (``intrinsics_file``). On each
frame with a valid detection it:

  * broadcasts a dynamic TF  ``<camera> -> <board_frame>``  (the board pose in
    the camera frame, ``T_cam_target``), and
  * republishes the input image with the board's coordinate axes (and detected
    corners) drawn on it (``board_pose.image_topic``), so the detection can be
    eye-aligned in RViz alongside the TF.

Intrinsics must already exist for the selected camera (run intrinsic.launch.py
first). Uses the same central ``calibration.yaml`` as the other nodes; only the
parameters this node declares are read.

It also offers a ``~/save_board_pose`` (std_srvs/srv/Trigger) service: calling
it latches the most recent valid ``T_cam_board`` to ``board_pose_file`` (YAML),
which the publisher node reads back to place the board (and the operator_body
hanging off it) into the static TF tree.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from calibration_multi_cam import se3
from calibration_multi_cam.extrinsics import estimate_target_pose
from calibration_multi_cam.intrinsics import K_from_intrinsics, dist_array
from calibration_multi_cam.publisher_node import _mat_to_quat
from calibration_multi_cam.target import AprilGridTarget


class BoardPoseNode(Node):
    def __init__(self):
        super().__init__("calibration_board_pose")

        camera_names = list(
            self.declare_parameter("camera_names", ["camera0"]).value
        )
        # Which camera to track. Empty -> the world/first camera.
        cam = self.declare_parameter("board_pose.camera", "").value
        self.camera = cam if cam else (camera_names[0] if camera_names else "camera0")
        self.topic = self.declare_parameter(
            f"{self.camera}.topic", f"/{self.camera}/image_raw"
        ).value

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

        # PnP needs >= 4 corners; reuse the shared collection threshold as a
        # sensible floor for a *stable* pose.
        self.min_corners = max(4, int(self.declare_parameter("min_corners_per_camera", 8).value))
        self.intrinsics_file = self.declare_parameter(
            "intrinsics_file",
            "/workspace/ros2_ws/src/calibration_multi_cam/config/intrinsics.yaml",
        ).value
        # The rig extrinsics connect the tracked camera back to the world frame
        # (camera0). Without them, a TF <camera_i> -> board is published but RViz
        # (fixed frame = world) has no path to it unless camera_i *is* the world.
        self.extrinsics_file = self.declare_parameter(
            "extrinsics_file",
            "/workspace/ros2_ws/src/calibration_multi_cam/config/extrinsics.yaml",
        ).value
        self.world_frame_override = self.declare_parameter("world_frame", "").value
        self.board_frame = self.declare_parameter("board_pose.board_frame", "calib_board").value
        # Where ``save_board_pose`` writes the latched T_cam_board (read back by
        # the publisher node to put the board into the static TF tree).
        self.board_pose_file = self.declare_parameter(
            "board_pose_file",
            "/workspace/ros2_ws/src/calibration_multi_cam/config/board_pose.yaml",
        ).value
        self.axis_length = float(self.declare_parameter("board_pose.axis_length", 0.1).value)
        image_topic = self.declare_parameter("board_pose.image_topic", "").value
        self.image_topic = image_topic or f"/{self.camera}/board_pose/image_axes"
        status_period = float(self.declare_parameter("status_period_sec", 3.0).value)

        self.bridge = CvBridge()
        self.K = None
        self.dist = None

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.image_pub = self.create_publisher(Image, self.image_topic, 10)
        self.sub = self.create_subscription(
            Image, self.topic, self._on_image, qos_profile_sensor_data
        )
        self.save_srv = self.create_service(
            Trigger, "~/save_board_pose", self._on_save_board_pose
        )
        self.status_timer = self.create_timer(status_period, self._log_status)

        self._frames = 0                 # images received
        self._last_detected = 0          # corners in the most recent frame
        self._last_pose_ok = False       # pose recovered in the most recent frame
        self._last_T = None              # most recent valid T_cam_board (4x4)

        self.world_frame = self._publish_rig_tf()

        self.get_logger().info(
            f"Board pose tracker up. camera={self.camera} (frame_id), "
            f"topic={self.topic}, target={self.target}.\n"
            f"  TF: {self.camera} -> {self.board_frame} | annotated image: {self.image_topic}\n"
            f"  world frame: {self.world_frame} | intrinsics: {self.intrinsics_file}"
        )

    # ------------------------------------------------------------------ #
    def _publish_rig_tf(self):
        """Latch the static world -> camera_i rig transforms from extrinsics_file.

        This connects the tracked camera (and the board hanging off it) to the
        world frame, so RViz can display the board no matter which camera is
        selected. Returns the world frame to use as RViz's fixed frame: the rig
        world if extrinsics are available, otherwise the tracked camera itself
        (a sane fallback when only intrinsics have been calibrated).
        """
        if not os.path.isfile(self.extrinsics_file):
            self.get_logger().warning(
                f"extrinsics_file not found: {self.extrinsics_file}. Only the "
                f"{self.camera} -> {self.board_frame} TF will be published; set "
                f"RViz's fixed frame to '{self.camera}' to see the board."
            )
            return self.camera
        try:
            with open(self.extrinsics_file, "r") as fh:
                extr = yaml.safe_load(fh) or {}
            world_frame = self.world_frame_override or extr.get("world_frame", "camera0")
            transforms = []
            now = self.get_clock().now().to_msg()
            for cam, c in extr.get("cameras", {}).items():
                if cam == world_frame:
                    continue  # world camera pose is identity; no self-TF
                T = np.asarray(c["T_world_cam"], dtype=np.float64)
                transforms.append(
                    self._make_tf(world_frame, cam, T[:3, 3], _mat_to_quat(T[:3, :3]), now)
                )
            if transforms:
                self.static_tf.sendTransform(transforms)
            self.get_logger().info(
                f"Published rig TF ({world_frame} -> {len(transforms)} cameras)."
            )
            return world_frame
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"Failed to publish rig TF from {self.extrinsics_file}: {exc}. "
                f"Set RViz's fixed frame to '{self.camera}'."
            )
            return self.camera

    @staticmethod
    def _make_tf(parent, child, p, quat, stamp):
        qx, qy, qz, qw = quat
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = parent
        tf.child_frame_id = child
        tf.transform.translation.x = float(p[0])
        tf.transform.translation.y = float(p[1])
        tf.transform.translation.z = float(p[2])
        tf.transform.rotation.x = float(qx)
        tf.transform.rotation.y = float(qy)
        tf.transform.rotation.z = float(qz)
        tf.transform.rotation.w = float(qw)
        return tf

    # ------------------------------------------------------------------ #
    def _ensure_intrinsics(self):
        """Lazily load K + distortion for the selected camera; returns ok."""
        if self.K is not None:
            return True
        if not os.path.isfile(self.intrinsics_file):
            self.get_logger().warning(
                f"intrinsics_file not found yet: {self.intrinsics_file} "
                f"(run intrinsic.launch.py first)",
                throttle_duration_sec=5.0,
            )
            return False
        try:
            with open(self.intrinsics_file, "r") as fh:
                data = yaml.safe_load(fh) or {}
            cam = data.get("cameras", {}).get(self.camera)
            if cam is None:
                self.get_logger().error(
                    f"camera '{self.camera}' not in {self.intrinsics_file}; "
                    f"available: {list(data.get('cameras', {}))}"
                )
                return False
            self.K = K_from_intrinsics(cam["intrinsics"])
            self.dist = dist_array(cam.get("distortion", []))
            self.get_logger().info(f"Loaded intrinsics for {self.camera}.")
            return True
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to read intrinsics: {exc}")
            return False

    def _on_image(self, msg):
        self._frames += 1
        if not self._ensure_intrinsics():
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f"image conversion failed: {exc}",
                                      throttle_duration_sec=5.0)
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        pids, pts = self.target.detect(gray)
        self._last_detected = int(pids.size)
        self._last_pose_ok = False

        # Always draw the detected corners so the detection is visible even when
        # the pose can't be solved.
        for p in pts:
            cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 3, (0, 255, 0), -1)

        if pids.size >= self.min_corners:
            T = estimate_target_pose(self.target.object_points, pids, pts, self.K, self.dist)
            if T is not None:
                rvec, tvec = se3.T_to_rt(T)
                cv2.drawFrameAxes(img, self.K, self.dist, rvec, tvec, self.axis_length, 3)
                self._broadcast_tf(T, msg.header.stamp)
                self._last_pose_ok = True
                self._last_T = T

        out = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.camera
        self.image_pub.publish(out)

    def _broadcast_tf(self, T_cam_target, stamp):
        tf = self._make_tf(
            self.camera, self.board_frame,
            T_cam_target[:3, 3], _mat_to_quat(T_cam_target[:3, :3]), stamp,
        )
        self.tf_broadcaster.sendTransform(tf)

    def _on_save_board_pose(self, request, response):
        """Latch the most recent T_cam_board to ``board_pose_file`` (YAML)."""
        if self._last_T is None:
            response.success = False
            response.message = (
                "No board pose available yet; point the board at "
                f"{self.camera} until pose=OK, then call again."
            )
            self.get_logger().warning(response.message)
            return response
        out = {
            "camera_frame": self.camera,
            "board_frame": self.board_frame,
            "T_cam_board": np.asarray(self._last_T, dtype=np.float64).tolist(),
        }
        try:
            os.makedirs(os.path.dirname(self.board_pose_file) or ".", exist_ok=True)
            with open(self.board_pose_file, "w") as fh:
                yaml.safe_dump(out, fh, default_flow_style=None, sort_keys=False)
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f"Failed to write {self.board_pose_file}: {exc}"
            self.get_logger().error(response.message)
            return response
        response.success = True
        response.message = (
            f"Saved T_cam_board ({self.camera} -> {self.board_frame}) to "
            f"{self.board_pose_file}"
        )
        self.get_logger().info(response.message)
        return response

    def _log_status(self):
        self.get_logger().info(
            f"{self.camera}: frames={self._frames} | "
            f"corners_last_frame={self._last_detected} (need >={self.min_corners}) | "
            f"pose={'OK' if self._last_pose_ok else 'no'}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = BoardPoseNode()
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
