"""Publisher node: load a calibration result and publish it.

    ros2 launch calibration_multi_cam publish.launch.py

Publishes, per the original spec:
  * intrinsics-only sensor_msgs/CameraInfo on `<camera>/camera_info`
    (K + distortion only; R and P are left empty/zero), latched.
  * the rig extrinsics as static TF (world -> camera) and as a
    geometry_msgs/PoseArray on `~/extrinsics`, latched.

The world frame is aligned with the first camera (`world_frame`, default
"cam0"); that camera's pose is identity, so no TF is emitted for it.

Expected `result_file` schema (written by the solver)::

    world_frame: cam0
    cameras:
      cam0:
        model: pinhole-radtan
        resolution: [w, h]
        intrinsics: [fx, fy, cx, cy]
        distortion: [k1, k2, p1, p2]
        T_world_cam:                # 4x4 pose of the camera in the world frame
          - [1,0,0,0]
          - [0,1,0,0]
          - [0,0,1,0]
          - [0,0,0,1]
      cam1: {...}
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

        self.result_file = self.declare_parameter(
            "result_file",
            "/workspace/ros2_ws/src/calibration_multi_cam/config/calibration_result.yaml",
        ).value
        self.world_frame_override = self.declare_parameter("world_frame", "").value
        self.publish_camera_info = bool(self.declare_parameter("publish_camera_info", True).value)
        self.publish_tf = bool(self.declare_parameter("publish_tf", True).value)
        self.publish_pose = bool(self.declare_parameter("publish_pose", True).value)
        self.info_template = self.declare_parameter(
            "camera_info_topic_template", "{camera}/camera_info"
        ).value

        self.latching_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.static_tf = StaticTransformBroadcaster(self)
        self.info_pubs = {}
        self.pose_pub = self.create_publisher(PoseArray, "~/extrinsics", self.latching_qos)

        self._published = False
        self.timer = self.create_timer(1.0, self._try_publish)
        self.get_logger().info(
            f"Publisher up; waiting for result_file: {self.result_file}"
        )

    def _try_publish(self):
        if self._published:
            return
        if not os.path.isfile(self.result_file):
            return
        try:
            with open(self.result_file, "r") as fh:
                result = yaml.safe_load(fh)
            self._publish(result)
            self._published = True
            self.get_logger().info("Published calibration (CameraInfo + TF/Pose).")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to publish result_file: {exc}")

    def _publish(self, result):
        world_frame = self.world_frame_override or result.get("world_frame", "cam0")
        cameras = result.get("cameras", {})
        now = self.get_clock().now().to_msg()

        transforms = []
        pose_array = PoseArray()
        pose_array.header.stamp = now
        pose_array.header.frame_id = world_frame

        for cam, c in cameras.items():
            T = np.asarray(c["T_world_cam"], dtype=np.float64)
            R, p = T[:3, :3], T[:3, 3]
            qx, qy, qz, qw = _mat_to_quat(R)

            # ---- intrinsics-only CameraInfo (no R, no P) ------------------
            if self.publish_camera_info:
                info = self._make_camera_info(cam, c, now)
                if cam not in self.info_pubs:
                    topic = self.info_template.format(camera=cam)
                    self.info_pubs[cam] = self.create_publisher(
                        CameraInfo, topic, self.latching_qos
                    )
                self.info_pubs[cam].publish(info)

            # ---- extrinsics: TF (skip the world camera itself) -----------
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

            # ---- extrinsics: Pose ----------------------------------------
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

        if self.publish_tf and transforms:
            self.static_tf.sendTransform(transforms)
        if self.publish_pose:
            self.pose_pub.publish(pose_array)

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
