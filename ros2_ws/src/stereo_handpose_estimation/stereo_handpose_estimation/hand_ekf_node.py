#!/usr/bin/env python3

"""
EKF smoothing of stereo hand-centroid positions.

Subscribes to the ``PoseWithCovarianceStamped`` hand poses published by
``stereo_handpose_node`` (``stereo_handpose/hand_left`` / ``_right``). Each
carries a triangulated 3D centroid plus its measurement covariance ``R`` (the
top-left 3x3 of the 6x6 block). One constant-velocity Kalman filter per hand
turns that noisy stream into a smoother, lower-jitter centroid track, which is
republished as ``PoseWithCovarianceStamped`` (+ an optional RViz marker).

Why the covariance matters
--------------------------
The state is ``[position, velocity]`` with a constant-velocity motion model.
Both the motion and the measurement (we observe position directly, ``H=[I|0]``)
are linear, so this exact linear Kalman filter *is* the EKF for this problem
(constant Jacobians, no linearisation). The win is that each measurement brings
its own ``R``: the Kalman gain ``K = P Hᵀ (H P Hᵀ + R)⁻¹`` automatically
down-weights noisy axes, so the large stereo depth variance gets smoothed hard
while the tight lateral axes are tracked closely. That adaptive weighting is
exactly what the upstream covariance enables.

Robustness
----------
- Per-measurement validity gate: drop measurements whose position sigma exceeds
  ``max_measurement_sigma`` (e.g. the 1e6 "unknown" covariance emitted when
  triangulation was degenerate).
- Mahalanobis gating: reject measurements too far from the prediction
  (chi-square, 3 DOF); the prediction is published instead.
- Track reset: if no measurement arrives for ``reset_timeout`` seconds (hand
  left the view), the next measurement re-seeds the filter.

Inputs
------
    hand_topics (x2)   geometry_msgs/PoseWithCovarianceStamped

Outputs
-------
    output_topics (x2)         geometry_msgs/PoseWithCovarianceStamped
    markers_topic              visualization_msgs/MarkerArray  (filtered centroid)
"""

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from stereo_handpose_estimation.kalman import ConstantVelocityKF

# Filtered-centroid marker color (green) + stable per-hand marker ids.
FILTERED_COLOR = ColorRGBA(r=0.2, g=1.0, b=0.4, a=1.0)
HAND_MARKER_IDS = {"Left": 0, "Right": 1}


class HandEKFNode(Node):
    def __init__(self):
        super().__init__("hand_ekf_node")

        # --- parameters -----------------------------------------------------
        self.declare_parameter(
            "hand_topics",
            ["stereo_handpose/hand_left", "stereo_handpose/hand_right"],
        )
        self.declare_parameter(
            "output_topics",
            ["stereo_handpose/hand_left/filtered",
             "stereo_handpose/hand_right/filtered"],
        )
        self.declare_parameter("hand_labels", ["Left", "Right"])
        self.declare_parameter("world_frame", "world")

        # Process-noise acceleration density (white-noise-accel model). Larger
        # -> more responsive / less smoothing; smaller -> smoother / laggier.
        # ~0.5 balances jitter reduction against lag on brisk hand motion; drop
        # toward 0.1 for a mostly-static hand, raise for fast gestures.
        self.declare_parameter("process_noise_accel", 0.5)
        # Initial velocity std (m/s) for the filter's P0.
        self.declare_parameter("init_velocity_sigma", 1.0)
        # Multiply the incoming measurement covariance R by this (>1 trusts the
        # measurement less, smooths more).
        self.declare_parameter("measurement_cov_scale", 1.0)
        # Floor on per-axis measurement std (m): clamps R's diagonal so an
        # over-confident measurement can't make the filter chase noise.
        self.declare_parameter("measurement_sigma_floor", 0.0005)
        # Drop a measurement whose largest per-axis std exceeds this (m). Catches
        # the "unknown" 1e6 covariance and grossly uncertain triangulations.
        self.declare_parameter("max_measurement_sigma", 0.5)
        # Mahalanobis gate (chi-square, 3 DOF): reject measurements farther than
        # this from the prediction. 7.81=0.95, 11.34=0.99, 16.27=0.999. <=0 off.
        self.declare_parameter("mahalanobis_gate", 16.27)
        # Re-seed the filter if no accepted measurement for this long (s).
        self.declare_parameter("reset_timeout", 0.5)

        self.declare_parameter("publish_markers", True)
        self.declare_parameter("markers_topic", "stereo_handpose/filtered_markers")
        self.declare_parameter("marker_size", 0.03)

        self.hand_topics = list(self.get_parameter("hand_topics").value)
        self.output_topics = list(self.get_parameter("output_topics").value)
        self.hand_labels = list(self.get_parameter("hand_labels").value)
        n = len(self.hand_topics)
        if not (len(self.output_topics) == n and len(self.hand_labels) == n):
            raise ValueError(
                "hand_topics, output_topics and hand_labels must be the same "
                f"length (got {n}, {len(self.output_topics)}, "
                f"{len(self.hand_labels)})"
            )
        self.world_frame = self.get_parameter("world_frame").value
        self.q_accel = float(self.get_parameter("process_noise_accel").value)
        self.init_velocity_sigma = float(
            self.get_parameter("init_velocity_sigma").value
        )
        self.measurement_cov_scale = float(
            self.get_parameter("measurement_cov_scale").value
        )
        self.measurement_sigma_floor = float(
            self.get_parameter("measurement_sigma_floor").value
        )
        self.max_measurement_sigma = float(
            self.get_parameter("max_measurement_sigma").value
        )
        self.mahalanobis_gate = float(
            self.get_parameter("mahalanobis_gate").value
        )
        self.reset_timeout = float(self.get_parameter("reset_timeout").value)
        self.publish_markers = bool(self.get_parameter("publish_markers").value)
        self.marker_size = float(self.get_parameter("marker_size").value)

        # --- per-hand filter state -----------------------------------------
        self.filters = {
            label: ConstantVelocityKF(self.q_accel, self.init_velocity_sigma)
            for label in self.hand_labels
        }
        self.last_t = {label: None for label in self.hand_labels}

        # --- publishers & subscriptions ------------------------------------
        self.out_pubs = {
            label: self.create_publisher(PoseWithCovarianceStamped, topic, 10)
            for label, topic in zip(self.hand_labels, self.output_topics)
        }
        self.marker_pub = None
        if self.publish_markers:
            self.marker_pub = self.create_publisher(
                MarkerArray, self.get_parameter("markers_topic").value, 10
            )

        self.subs = []
        for label, topic in zip(self.hand_labels, self.hand_topics):
            self.subs.append(
                self.create_subscription(
                    PoseWithCovarianceStamped,
                    topic,
                    lambda msg, lbl=label: self._on_pose(msg, lbl),
                    10,
                )
            )

        self.get_logger().info(
            f"hand_ekf_node ready: filtering {self.hand_labels}, "
            f"q_accel={self.q_accel}, gate={self.mahalanobis_gate}, "
            f"reset_timeout={self.reset_timeout}s"
        )

    # --------------------------------------------------------------- callback
    def _on_pose(self, msg, label):
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9

        p = msg.pose.pose.position
        z = np.array([p.x, p.y, p.z], dtype=float)
        cov6 = np.array(msg.pose.covariance, dtype=float).reshape(6, 6)
        R = cov6[:3, :3] * self.measurement_cov_scale

        # Validity: finite, and not the "unknown"/grossly uncertain covariance.
        max_sigma = np.sqrt(max(float(np.max(np.diag(R))), 0.0))
        if (not np.all(np.isfinite(z)) or not np.all(np.isfinite(R))
                or max_sigma > self.max_measurement_sigma):
            self.get_logger().warn(
                f"[{label}] dropping uninformative measurement "
                f"(max sigma {max_sigma:.3f} m)",
                throttle_duration_sec=5.0,
            )
            return

        # Floor the diagonal so an over-confident R can't make us chase noise.
        floor_var = self.measurement_sigma_floor ** 2
        diag = np.clip(np.diag(R), floor_var, None)
        np.fill_diagonal(R, diag)

        kf = self.filters[label]
        last = self.last_t[label]
        rejected = False

        if not kf.initialized or last is None:
            kf.initialize(z, R)
            self.last_t[label] = t
        else:
            dt = t - last
            if dt <= 0.0:
                return  # duplicate / out-of-order stamp
            if dt > self.reset_timeout:
                kf.initialize(z, R)  # gap in tracking -> re-seed
            else:
                kf.predict(dt)
                maha2 = kf.innovation_mahalanobis2(z, R)
                if self.mahalanobis_gate > 0.0 and maha2 > self.mahalanobis_gate:
                    rejected = True  # outlier: keep the prediction
                else:
                    kf.update(z, R)
            self.last_t[label] = t

        self._publish_filtered(label, kf, msg.pose.pose.orientation, stamp)
        if self.marker_pub is not None:
            self._publish_marker(label, kf.position, stamp)

        sigma = np.sqrt(np.clip(np.diag(kf.position_covariance), 0.0, None))
        speed = float(np.linalg.norm(kf.velocity))
        self.get_logger().info(
            f"[{label}] filt sigma(x,y,z)=("
            f"{sigma[0] * 1000:.1f},{sigma[1] * 1000:.1f},"
            f"{sigma[2] * 1000:.1f})mm speed={speed:.2f}m/s"
            f"{' [meas rejected]' if rejected else ''}",
            throttle_duration_sec=5.0,
        )

    # ------------------------------------------------------------- publishing
    def _publish_filtered(self, label, kf, orientation, stamp):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.world_frame
        msg.header.stamp = stamp
        pos = kf.position
        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        msg.pose.pose.orientation = orientation  # pass the input through

        cov6 = np.zeros((6, 6), dtype=float)
        cov6[:3, :3] = kf.position_covariance
        cov6[3, 3] = cov6[4, 4] = cov6[5, 5] = 1e6  # orientation not estimated
        msg.pose.covariance = cov6.flatten().tolist()
        self.out_pubs[label].publish(msg)

    def _publish_marker(self, label, pos, stamp):
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = stamp
        m.ns = f"filtered_{label.lower()}"
        m.id = HAND_MARKER_IDS.get(label, abs(hash(label)) % 1000)
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = self.marker_size
        m.color = FILTERED_COLOR
        m.lifetime = Duration(sec=0, nanosec=300_000_000)
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.position.z = float(pos[2])
        m.pose.orientation.w = 1.0
        array = MarkerArray()
        array.markers.append(m)
        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = HandEKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
