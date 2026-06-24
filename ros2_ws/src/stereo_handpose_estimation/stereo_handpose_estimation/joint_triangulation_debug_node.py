#!/usr/bin/env python3

"""Per-joint stereo-triangulation debug node.

Why
---
``stereo_handpose_node`` triangulates only the hand *centroid* (one well-averaged
point per hand), which is stable but hides which individual landmarks are noisy.
When the recovered depth wobbles, the culprit is usually one or two landmarks
whose left/right correspondence is poor in a given frame.

This node triangulates **all 21 landmarks independently** (landmark ``i`` in cam0
vs landmark ``i`` in cam1 -- the same joint by MediaPipe's index convention) and
streams, per hand, the per-joint world coordinates and the per-joint
reprojection residual as ``sensor_msgs/JointState``. JointState carries parallel
``name[]`` / ``position[]`` arrays that PlotJuggler and Foxglove split into one
labelled time series per joint -- so you can watch each joint's ``z`` over time
and immediately see which one jumps.

The reprojection **residual** (``joint_residual``) is the real diagnostic: a
joint whose residual spikes is a bad stereo correspondence in that frame, which
explains the depth jump far better than ``z`` alone. Plot ``joint_z`` and
``joint_residual`` stacked and look for the joint whose residual correlates with
the wobble.

It reuses ``stereo_handpose_node``'s calibration handling (stereo-from-
camera_info, or extrinsics.yaml + raw K + DLT) so the world frame and metric
scale match the production node exactly. It publishes no markers and touches no
production code -- run it alongside the real node, then kill it.

Topics
------
    <lm topic> x2                       handpose3d_msgs/HandLandmarks  (in)
    <camera_info> x2                    sensor_msgs/CameraInfo         (in)
    stereo_handpose/debug/<hand>/joint_x          sensor_msgs/JointState (out)
    stereo_handpose/debug/<hand>/joint_y          sensor_msgs/JointState (out)
    stereo_handpose/debug/<hand>/joint_z          sensor_msgs/JointState (out)
    stereo_handpose/debug/<hand>/joint_residual   sensor_msgs/JointState (out)
        <hand> in {left, right}; residual is the mean reprojection error (px).
"""

import os

import cv2
import numpy as np
import rclpy
import yaml
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, JointState

from handpose3d_msgs.msg import HandLandmarks

from stereo_handpose_estimation.triangulation import dlt, make_projection_matrix

N_LANDMARKS = 21
HAND_LABELS = ("Left", "Right")

# MediaPipe 21-landmark hand model, in index order, so PlotJuggler legends read
# as joint names instead of lm00..lm20.
JOINT_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


class JointTriangulationDebugNode(Node):
    def __init__(self):
        super().__init__("joint_triangulation_debug_node")

        # --- parameters (calibration subset mirrors stereo_handpose_node) ----
        self.declare_parameter("camera_names", ["camera0", "camera1"])
        self.declare_parameter(
            "landmark_topics",
            ["camera0/image_raw/landmarks/hands",
             "camera1/image_raw/landmarks/hands"],
        )
        self.declare_parameter(
            "camera_info_topics", ["camera0/camera_info", "camera1/camera_info"]
        )
        self.declare_parameter(
            "extrinsics_file",
            "/workspace/ros2_ws/src/stereo_handpose_estimation/config/"
            "extrinsics.yaml",
        )
        self.declare_parameter("use_camera_info_extrinsics", False)
        # Set true when landmarks come from ALREADY-rectified images
        # (mediapie_landmarks_extraction enable_rectification=true): the 2D
        # points are already in the rectified (P) frame and are fed straight to
        # DLT, skipping the per-point undistort/rectify. Only affects STEREO mode.
        self.declare_parameter("enable_rectification", False)
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("min_score", 0.5)
        self.declare_parameter("scale", 0.05)
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("sync_queue_size", 10)

        self.camera_names = list(self.get_parameter("camera_names").value)
        if len(self.camera_names) != 2:
            raise ValueError("joint_triangulation_debug_node requires 2 cameras")
        self.landmark_topics = list(self.get_parameter("landmark_topics").value)
        self.camera_info_topics = list(
            self.get_parameter("camera_info_topics").value
        )
        if len(self.landmark_topics) != 2 or len(self.camera_info_topics) != 2:
            raise ValueError("landmark_topics and camera_info_topics need 2 each")
        self.extrinsics_file = self.get_parameter("extrinsics_file").value
        self.use_camera_info_extrinsics = bool(
            self.get_parameter("use_camera_info_extrinsics").value
        )
        self.enable_rectification = bool(
            self.get_parameter("enable_rectification").value
        )
        self.world_frame = self.get_parameter("world_frame").value
        self.min_score = float(self.get_parameter("min_score").value)
        self.scale = float(self.get_parameter("scale").value)

        # --- calibration state ----------------------------------------------
        self.extrinsics = (
            None
            if self.use_camera_info_extrinsics
            else self._load_extrinsics(self.extrinsics_file)
        )
        self.calib = {name: None for name in self.camera_names}
        self.P_ext = {name: None for name in self.camera_names}
        self.mode = None
        self.ready = False
        self.effective_scale = self.scale

        # --- subscriptions ---------------------------------------------------
        self.info_subs = []
        for name, topic in zip(self.camera_names, self.camera_info_topics):
            self.info_subs.append(
                self.create_subscription(
                    CameraInfo,
                    topic,
                    lambda msg, n=name: self._on_camera_info(msg, n),
                    qos_profile_sensor_data,
                )
            )

        lm_subs = [Subscriber(self, HandLandmarks, t) for t in self.landmark_topics]
        self.sync = ApproximateTimeSynchronizer(
            lm_subs,
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop").value),
        )
        self.sync.registerCallback(self._on_landmarks)

        # --- publishers: per hand, one JointState per field -----------------
        self.pubs = {}
        for label in HAND_LABELS:
            hand = label.lower()
            self.pubs[label] = {
                field: self.create_publisher(
                    JointState, f"stereo_handpose/debug/{hand}/joint_{field}", 10
                )
                for field in ("x", "y", "z", "residual")
            }

        self.get_logger().info(
            f"joint_triangulation_debug_node ready: cameras={self.camera_names}, "
            "waiting for camera_info..."
        )

    # ------------------------------------------------------------------ setup
    def _load_extrinsics(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Extrinsics file not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        cameras = data["cameras"]
        ext = {}
        for name in self.camera_names:
            if name not in cameras:
                raise KeyError(f"No extrinsics for camera '{name}' in {path}")
            R = np.array(cameras[name]["rotation"], dtype=float).reshape(3, 3)
            t = np.array(cameras[name]["translation"], dtype=float).reshape(3)
            ext[name] = (R, t)
        return ext

    def _on_camera_info(self, msg, name):
        if self.calib[name] is not None:
            return  # calibration is static; capture once
        d = np.array(msg.d, dtype=float).ravel()
        if d.size == 0:
            d = np.zeros(5)
        self.calib[name] = {
            "k": np.array(msg.k, dtype=float).reshape(3, 3),
            "d": d,
            "r": np.array(msg.r, dtype=float).reshape(3, 3),
            "p": np.array(msg.p, dtype=float).reshape(3, 4),
            "model": (msg.distortion_model or "plumb_bob").lower(),
        }
        self.get_logger().info(f"Captured calibration for {name}")
        if all(self.calib[n] is not None for n in self.camera_names):
            self._on_calibration_ready()

    def _on_calibration_ready(self):
        if self.use_camera_info_extrinsics:
            self.mode = "stereo"
            self.effective_scale = 1.0  # rectified P is already metric
            P1 = self.calib[self.camera_names[1]]["p"]
            baseline = max(abs(P1[0, 3]), abs(P1[1, 3]))
            if baseline <= 1e-9:
                self.get_logger().warn(
                    "use_camera_info_extrinsics=true but camera_info P has no "
                    "baseline: triangulation will be degenerate."
                )
        else:
            self.mode = "extrinsics"
            self.effective_scale = self.scale
            for name in self.camera_names:
                R, t = self.extrinsics[name]
                self.P_ext[name] = make_projection_matrix(
                    self.calib[name]["k"], R, t
                )
        self.ready = True
        self.get_logger().info(f"Triangulation mode: {self.mode.upper()} ready.")

    # --------------------------------------------------------------- callbacks
    def _on_landmarks(self, msg0, msg1):
        if not self.ready:
            self.get_logger().warn(
                "Waiting for camera_info on all cameras...",
                throttle_duration_sec=5.0,
            )
            return

        hands0 = self._index_hands(msg0)
        hands1 = self._index_hands(msg1)
        stamp = msg0.header.stamp

        for label in HAND_LABELS:
            h0 = hands0.get(label)
            h1 = hands1.get(label)
            if h0 is None or h1 is None:
                continue  # need the hand in both views to triangulate

            xs, ys, zs, residuals = [], [], [], []
            for i in range(N_LANDMARKS):
                X, residual = self._triangulate_point(h0["pts"][i], h1["pts"][i])
                if np.all(np.isfinite(X)):
                    X = X * self.effective_scale
                    xs.append(float(X[0]))
                    ys.append(float(X[1]))
                    zs.append(float(X[2]))
                else:
                    xs.append(float("nan"))
                    ys.append(float("nan"))
                    zs.append(float("nan"))
                residuals.append(float(residual))

            self._publish(label, "x", xs, stamp)
            self._publish(label, "y", ys, stamp)
            self._publish(label, "z", zs, stamp)
            self._publish(label, "residual", residuals, stamp)

    # ------------------------------------------------------------- core maths
    def _index_hands(self, msg):
        """Index a HandLandmarks msg by handedness.

        Returns {label: {pts: (21, 2) image landmarks, score: float}}.
        Duplicate labels keep the higher-confidence detection.
        """
        out = {}
        for hand in msg.hands:
            if hand.score < self.min_score:
                continue
            label = hand.handedness
            if label in out and hand.score <= out[label]["score"]:
                continue
            img = hand.landmarks_image
            if len(img) != N_LANDMARKS:
                continue
            pts = np.array([[p.x, p.y] for p in img], dtype=float)
            out[label] = {"pts": pts, "score": float(hand.score)}
        return out

    def _triangulate_point(self, p0_raw, p1_raw):
        """Triangulate one landmark correspondence -> (world point, residual px).

        Residual is the mean reprojection error over both views, in pixels,
        measured against the same points fed to the DLT (undistorted/rectified
        in stereo mode, raw in extrinsics mode).
        """
        n0, n1 = self.camera_names
        if self.mode == "stereo":
            if self.enable_rectification:
                # Images were rectified upstream -> points already in the
                # rectified (P) frame; no per-point undistort/rectify needed.
                p0, p1 = p0_raw, p1_raw
            else:
                p0 = self._undistort_point(n0, p0_raw)
                p1 = self._undistort_point(n1, p1_raw)
            P0, P1 = self.calib[n0]["p"], self.calib[n1]["p"]
        else:
            p0, p1 = p0_raw, p1_raw
            P0, P1 = self.P_ext[n0], self.P_ext[n1]

        X = dlt(P0, P1, p0, p1)
        if not np.all(np.isfinite(X)):
            return X, float("nan")

        residual = 0.5 * (
            self._reproj_error(P0, X, p0) + self._reproj_error(P1, X, p1)
        )
        return X, residual

    @staticmethod
    def _reproj_error(P, X, pt):
        proj = P @ np.array([X[0], X[1], X[2], 1.0])
        if abs(proj[2]) < 1e-12:
            return float("nan")
        uv = proj[:2] / proj[2]
        return float(np.hypot(uv[0] - pt[0], uv[1] - pt[1]))

    def _undistort_point(self, name, pt):
        c = self.calib[name]
        src = np.ascontiguousarray(pt, dtype=np.float64).reshape(-1, 1, 2)
        if c["model"] == "fisheye":
            out = cv2.fisheye.undistortPoints(
                src, c["k"], c["d"][:4].reshape(1, 4), R=c["r"], P=c["p"]
            )
        else:
            out = cv2.undistortPoints(src, c["k"], c["d"], R=c["r"], P=c["p"])
        return out.reshape(2)

    # ------------------------------------------------------------- publishing
    def _publish(self, label, field, values, stamp):
        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = self.world_frame
        msg.name = JOINT_NAMES
        msg.position = values
        self.pubs[label][field].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointTriangulationDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
