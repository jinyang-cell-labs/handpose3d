"""Publisher node: load the two calibration files and publish them.

    ros2 launch calibration_multi_cam publish.launch.py

Loads `intrinsics_file` (stage 1) and `extrinsics_file` (stage 2) and publishes:
  * intrinsics-only sensor_msgs/CameraInfo on `<camera>/camera_info`
    (K + distortion only; R and P are left empty/zero), latched.
  * the rig extrinsics as static TF (world -> camera) and a
    geometry_msgs/PoseArray on `~/extrinsics`, latched.
  * if `board_pose_file` exists, the saved board pose as static TF
    (<camera> -> <board_frame>), and the operator_body offset as static TF
    (<board_frame> -> operator_body), completing the chain
    world -> camera -> board -> operator_body so RViz shows the full tree.

The world frame is the first camera (`extrinsics_file: world_frame`, default
"camera0"); that camera's pose is identity, so no TF is emitted for it.

File schemas
------------
intrinsics_file::
    cameras:
      camera0: {model, resolution: [w,h], intrinsics: [fx,fy,cx,cy], distortion: [k1,k2,p1,p2]}
extrinsics_file::
    world_frame: camera0
    cameras:
      camera0: {T_world_cam: 4x4}     # identity for the world camera
      camera1: {T_world_cam: 4x4}
"""
from __future__ import annotations

import os

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo
from tf2_ros import StaticTransformBroadcaster

from calibration_multi_cam import se3


def _mat_to_quat(R):
    """Rotation matrix (3x3) -> quaternion (x, y, z, w)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (x, y, z, w)


class PublisherNode(Node):
    def __init__(self):
        super().__init__("calibration_publisher")

        self.intrinsics_file = self.declare_parameter(
            "intrinsics_file",
            "/workspace/ros2_ws/src/calibration_multi_cam/config/intrinsics.yaml",
        ).value
        self.extrinsics_file = self.declare_parameter(
            "extrinsics_file",
            "/workspace/ros2_ws/src/calibration_multi_cam/config/extrinsics.yaml",
        ).value
        self.world_frame_override = self.declare_parameter("world_frame", "").value
        self.publish_camera_info = bool(self.declare_parameter("publish_camera_info", True).value)
        self.publish_tf = bool(self.declare_parameter("publish_tf", True).value)
        self.publish_pose = bool(self.declare_parameter("publish_pose", True).value)
        self.info_template = self.declare_parameter(
            "camera_info_topic_template", "{camera}/camera_info").value

        # ---- board pose + operator_body (extends the static TF tree) -------
        self.board_pose_file = self.declare_parameter(
            "board_pose_file",
            "/workspace/ros2_ws/src/calibration_multi_cam/config/board_pose.yaml",
        ).value
        self.board_frame = self.declare_parameter(
            "board_pose.board_frame", "calib_board").value
        self.publish_operator_body = bool(
            self.declare_parameter("publish_operator_body", True).value)
        self.operator_body_frame = self.declare_parameter(
            "operator_body.frame", "operator_body").value
        self.operator_body_position = list(
            self.declare_parameter("operator_body.position", [0.0, 0.0, 0.0]).value)
        self.operator_body_rotation = list(
            self.declare_parameter("operator_body.rotation", [0.0, 0.0, 0.0]).value)

        self.latching_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.static_tf = StaticTransformBroadcaster(self)
        self.info_pubs = {}
        self.pose_pub = self.create_publisher(PoseArray, "~/extrinsics", self.latching_qos)

        self._published = False
        self.timer = self.create_timer(1.0, self._try_publish)
        self.get_logger().info(
            f"Publisher up; waiting for\n  intrinsics: {self.intrinsics_file}"
            f"\n  extrinsics: {self.extrinsics_file}")

    def _try_publish(self):
        if self._published:
            return
        if not (os.path.isfile(self.intrinsics_file) and os.path.isfile(self.extrinsics_file)):
            return
        try:
            with open(self.intrinsics_file, "r") as fh:
                intrinsics = yaml.safe_load(fh)
            with open(self.extrinsics_file, "r") as fh:
                extrinsics = yaml.safe_load(fh)
            self._publish(intrinsics, extrinsics)
            self._published = True
            self.get_logger().info("Published calibration (CameraInfo + TF/Pose).")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to publish calibration: {exc}")

    def _publish(self, intrinsics, extrinsics):
        now = self.get_clock().now().to_msg()
        intr_cams = intrinsics.get("cameras", {})
        extr_cams = extrinsics.get("cameras", {})
        world_frame = self.world_frame_override or extrinsics.get("world_frame", "camera0")

        # ---- intrinsics-only CameraInfo (no R, no P) ----------------------
        if self.publish_camera_info:
            for cam, c in intr_cams.items():
                info = self._make_camera_info(cam, c, now)
                if cam not in self.info_pubs:
                    topic = self.info_template.format(camera=cam)
                    self.info_pubs[cam] = self.create_publisher(
                        CameraInfo, topic, self.latching_qos)
                self.info_pubs[cam].publish(info)

        # ---- extrinsics: static TF + PoseArray ----------------------------
        transforms = []
        pose_array = PoseArray()
        pose_array.header.stamp = now
        pose_array.header.frame_id = world_frame
        for cam, c in extr_cams.items():
            T = np.asarray(c["T_world_cam"], dtype=np.float64)
            p = T[:3, 3]
            qx, qy, qz, qw = _mat_to_quat(T[:3, :3])
            if self.publish_tf and cam != world_frame:
                tf = TransformStamped()
                tf.header.stamp = now
                tf.header.frame_id = world_frame
                tf.child_frame_id = cam
                tf.transform.translation.x = float(p[0])
                tf.transform.translation.y = float(p[1])
                tf.transform.translation.z = float(p[2])
                tf.transform.rotation.x = float(qx)
                tf.transform.rotation.y = float(qy)
                tf.transform.rotation.z = float(qz)
                tf.transform.rotation.w = float(qw)
                transforms.append(tf)
            if self.publish_pose:
                pose = Pose()
                pose.position.x = float(p[0])
                pose.position.y = float(p[1])
                pose.position.z = float(p[2])
                pose.orientation.x = float(qx)
                pose.orientation.y = float(qy)
                pose.orientation.z = float(qz)
                pose.orientation.w = float(qw)
                pose_array.poses.append(pose)

        if self.publish_tf:
            transforms.extend(self._board_and_operator_transforms(now))
        if self.publish_tf and transforms:
            self.static_tf.sendTransform(transforms)
        if self.publish_pose:
            self.pose_pub.publish(pose_array)

    def _board_and_operator_transforms(self, now):
        """Static TFs that hang the board (and operator_body) off the rig.

          <camera_frame> -> <board_frame>     from board_pose_file (if present)
          <board_frame>  -> <operator_body>   from operator_body.* params

        The board TF is skipped (with a warning) when board_pose_file does not
        exist yet — save it first via /calibration_board_pose/save_board_pose.
        """
        transforms = []
        board_frame = self.board_frame
        if os.path.isfile(self.board_pose_file):
            try:
                with open(self.board_pose_file, "r") as fh:
                    bp = yaml.safe_load(fh) or {}
                camera_frame = bp["camera_frame"]
                board_frame = bp.get("board_frame", self.board_frame)
                T = np.asarray(bp["T_cam_board"], dtype=np.float64)
                transforms.append(self._make_tf(
                    camera_frame, board_frame, T[:3, 3], _mat_to_quat(T[:3, :3]), now))
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(
                    f"Failed to read board pose from {self.board_pose_file}: {exc}")
        else:
            self.get_logger().warning(
                f"board_pose_file not found: {self.board_pose_file}; the board "
                "TF (and so operator_body) will be disconnected. Save it with "
                "/calibration_board_pose/save_board_pose first.")

        if self.publish_operator_body:
            R = se3.euler_deg_to_R(*self.operator_body_rotation)
            p = np.asarray(self.operator_body_position, dtype=np.float64)
            transforms.append(self._make_tf(
                board_frame, self.operator_body_frame, p, _mat_to_quat(R), now))
        return transforms

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

    def _make_camera_info(self, cam, c, stamp):
        fx, fy, cx, cy = [float(v) for v in c["intrinsics"]]
        dist = [float(v) for v in c.get("distortion", [])]
        w, h = (int(v) for v in c["resolution"])
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = cam
        info.width = w
        info.height = h
        info.distortion_model = "plumb_bob"
        # plumb_bob expects [k1, k2, t1, t2, k3]; radtan gives [k1, k2, p1, p2].
        info.d = (dist + [0.0])[:5] if len(dist) == 4 else dist
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        # R and P intentionally left as zero arrays: intrinsics only.
        return info


def main(args=None):
    rclpy.init(args=args)
    node = PublisherNode()
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
