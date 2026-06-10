#!/usr/bin/env python3

"""
Hand pose estimation node.

Subscribes to two calibrated camera streams published by ``vision_interfaces``:

    <name>/image_raw    sensor_msgs/Image      (bgr8)
    <name>/camera_info  sensor_msgs/CameraInfo

Runs MediaPipe's HandLandmarker on each view to get 2D hand keypoints, then
triangulates the 21 landmarks to 3D via the Direct Linear Transform (intrinsics
come from the camera_info topics, stereo extrinsics from the node's config).

The 3D skeleton is published as a ``visualization_msgs/MarkerArray`` in the
world frame for visualization in RViz. The annotated 2D views are optionally
republished as ``<name>/handpose/annotated``.

On top of the raw per-joint triangulation, a model-based wrist-pose pipeline
(docs/estimation_guide_v1.md) emits one temporally smooth 6-DoF pose per hand
as ``geometry_msgs/PoseStamped`` on ``handpose/wrist_left`` and
``handpose/wrist_right``:

    A1 confidence-weighted DLT -> A2 reprojection residuals ->
    B1 reachability gate -> C Procrustes/Kabsch fit (B2 RANSAC-wrapped) ->
    D constant-velocity Kalman + SLERP orientation LPF -> E One-Euro polish

Each stage has an ``*.enabled`` parameter so the pipeline can be brought up
incrementally. The pipeline runs in METRES (triangulated world units are
multiplied by ``effective_scale`` first), so all thresholds are physical in
both triangulation modes.
"""

import os

import cv2
import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from handpose_estimation.triangulation import (
    make_projection_matrix,
    rotation_matrix_to_quaternion,
    triangulate_point,
)
from handpose_estimation.wrist_pose_pipeline import (
    ReachabilityShell,
    WristTracker,
)

# Hand skeleton connections (21 landmarks), formerly mp.solutions.hands.HAND_CONNECTIONS
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]
N_LANDMARKS = 21

# Standard scenario: one person, up to two hands. Hands are matched across the
# two cameras by MediaPipe's handedness label.
HAND_LABELS = ("Left", "Right")
# RViz marker colors (RGBA) per hand.
HAND_COLORS = {
    "Left": ColorRGBA(r=0.2, g=0.6, b=1.0, a=1.0),   # blue
    "Right": ColorRGBA(r=1.0, g=0.5, b=0.2, a=1.0),  # orange
}
# BGR colors for the 2D annotated overlay (OpenCV order).
HAND_BGR = {"Left": (255, 150, 50), "Right": (50, 150, 255)}
# Stable (joints, bones) marker ids per hand so updates replace in place.
HAND_MARKER_IDS = {"Left": (0, 1), "Right": (2, 3)}
# Marker ids for the fitted-template skeleton (R*template+t, Stage C output),
# published alongside the raw joints for A/B comparison in RViz.
FITTED_MARKER_IDS = {"Left": (4, 5), "Right": (6, 7)}
FITTED_COLORS = {
    "Left": ColorRGBA(r=0.5, g=0.9, b=0.5, a=0.8),   # green-ish
    "Right": ColorRGBA(r=0.9, g=0.9, b=0.3, a=0.8),  # yellow-ish
}

# ===== HOTFIX(rotate-90): TEMPORARY =========================================
# The current rosbag publishes a 90deg-rotated image (hardware limitation).
# We rotate it upright before detection so MediaPipe works and the frame lines
# up with the camera_info calibration (landscape 640x480 -> portrait 480x640).
# Set to None to disable, or DELETE this constant + the fenced block in
# _on_images, once the upstream publishes upright images.
#   options: cv2.ROTATE_90_CLOCKWISE / cv2.ROTATE_90_COUNTERCLOCKWISE / cv2.ROTATE_180
_HOTFIX_ROTATE = cv2.ROTATE_90_CLOCKWISE
# ============================================================================


class HandPoseNode(Node):
    def __init__(self):
        super().__init__("handpose_node")

        # --- parameters -----------------------------------------------------
        self.declare_parameter("camera_names", ["camera0", "camera1"])
        self.declare_parameter(
            "model_path",
            "/workspace/ros2_ws/src/handpose_estimation/models/hand_landmarker.task",
        )
        self.declare_parameter(
            "extrinsics_file",
            "/workspace/ros2_ws/src/handpose_estimation/config/extrinsics.yaml",
        )
        # true  -> triangulate from the camera_info stereo calibration
        #          (undistort/rectify with K/D/R/P + cv2.triangulatePoints).
        # false -> raw K + extrinsics.yaml + DLT (no distortion handling).
        self.declare_parameter("use_camera_info_extrinsics", False)
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("num_hands", 2)
        # Flip camera1's Left/Right labels if the two views are mirror-flipped
        # relative to each other (handedness then disagrees across cameras).
        self.declare_parameter("swap_handedness_camera1", False)
        self.declare_parameter("min_hand_detection_confidence", 0.5)
        self.declare_parameter("min_hand_presence_confidence", 0.5)
        self.declare_parameter("min_tracking_confidence", 0.5)
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("sync_queue_size", 10)
        # Calibration world units -> metres for RViz-friendly marker sizes.
        self.declare_parameter("scale", 0.05)
        self.declare_parameter("joint_size", 0.02)
        self.declare_parameter("line_width", 0.01)
        self.declare_parameter("publish_annotated", True)
        # Publish each camera's pose (from extrinsics) as TF + a frustum marker.
        self.declare_parameter("publish_camera_pose", True)
        self.declare_parameter("camera_marker_size", 0.08)

        self.camera_names = list(self.get_parameter("camera_names").value)
        if len(self.camera_names) != 2:
            raise ValueError("handpose_node requires exactly 2 camera_names")
        self.model_path = self.get_parameter("model_path").value
        self.extrinsics_file = self.get_parameter("extrinsics_file").value
        self.use_camera_info_extrinsics = bool(
            self.get_parameter("use_camera_info_extrinsics").value
        )
        self.world_frame = self.get_parameter("world_frame").value
        self.num_hands = int(self.get_parameter("num_hands").value)
        self.swap_handedness_camera1 = bool(
            self.get_parameter("swap_handedness_camera1").value
        )
        self.scale = float(self.get_parameter("scale").value)
        self.joint_size = float(self.get_parameter("joint_size").value)
        self.line_width = float(self.get_parameter("line_width").value)
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self.publish_camera_pose = bool(
            self.get_parameter("publish_camera_pose").value
        )
        self.camera_marker_size = float(
            self.get_parameter("camera_marker_size").value
        )

        # --- wrist-pose pipeline parameters ----------------------------------
        self._declare_pipeline_params()
        self._load_template_and_shell()
        self._build_trackers()

        # --- mediapipe detectors (one per camera so VIDEO timestamps stay
        # independent) ------------------------------------------------------
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Hand landmark model not found at {self.model_path}. "
                "Run scripts/download_model.sh to fetch hand_landmarker.task."
            )
        self.detectors = [self._make_landmarker() for _ in self.camera_names]
        self._frame_idx = 0

        # --- calibration state ---------------------------------------------
        # extrinsics.yaml is only needed for the 'extrinsics' path; skip loading
        # it when triangulating from the camera_info stereo calibration.
        self.extrinsics = (
            None
            if self.use_camera_info_extrinsics
            else self._load_extrinsics(self.extrinsics_file)
        )
        # Full per-camera calibration (k/d/r/p/model) captured from camera_info;
        # the triangulation mode is chosen once both have arrived.
        self.calib = {name: None for name in self.camera_names}
        self.P_ext = {name: None for name in self.camera_names}
        self.mode = None
        self.ready = False
        self.effective_scale = self.scale

        # --- subscriptions & publishers ------------------------------------
        self.info_subs = []
        for name in self.camera_names:
            self.info_subs.append(
                self.create_subscription(
                    CameraInfo,
                    f"{name}/camera_info",
                    lambda msg, n=name: self._on_camera_info(msg, n),
                    qos_profile_sensor_data,
                )
            )

        image_subs = [
            Subscriber(self, Image, f"{name}/image_raw", qos_profile=qos_profile_sensor_data)
            for name in self.camera_names
        ]
        self.sync = ApproximateTimeSynchronizer(
            image_subs,
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop").value),
        )
        self.sync.registerCallback(self._on_images)

        self.marker_pub = self.create_publisher(MarkerArray, "handpose/markers", 10)
        # 6-DoF wrist pose per hand (wrist-pose pipeline output).
        self.wrist_pubs = {
            "Left": self.create_publisher(PoseStamped, "handpose/wrist_left", 10),
            "Right": self.create_publisher(PoseStamped, "handpose/wrist_right", 10),
        }
        self.annotated_pubs = {}
        if self.publish_annotated:
            for name in self.camera_names:
                self.annotated_pubs[name] = self.create_publisher(
                    Image, f"{name}/handpose/annotated", qos_profile_sensor_data
                )

        # --- camera poses ---------------------------------------------------
        # Broadcast each camera's pose as static TF (this also gives RViz the
        # 'world' frame to use as fixed frame) and a latched frustum marker so
        # late-joining RViz still receives it.
        # Poses are published from _on_calibration_ready() — the stereo path
        # derives them from camera_info P, which hasn't arrived yet.
        self.static_tf_broadcaster = None
        self.camera_marker_pub = None
        if self.publish_camera_pose:
            self.static_tf_broadcaster = StaticTransformBroadcaster(self)
            latching_qos = QoSProfile(
                depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
            )
            self.camera_marker_pub = self.create_publisher(
                MarkerArray, "handpose/cameras", latching_qos
            )

        self.get_logger().info(
            f"handpose_node ready: cameras={self.camera_names}, "
            f"world_frame='{self.world_frame}', waiting for camera_info + images..."
        )

    # ------------------------------------------------------------------ setup
    def _make_landmarker(self):
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=self.num_hands,
            min_hand_detection_confidence=float(
                self.get_parameter("min_hand_detection_confidence").value
            ),
            min_hand_presence_confidence=float(
                self.get_parameter("min_hand_presence_confidence").value
            ),
            min_tracking_confidence=float(
                self.get_parameter("min_tracking_confidence").value
            ),
        )
        return mp_vision.HandLandmarker.create_from_options(options)

    def _load_extrinsics(self, path):
        """Load per-camera world->camera rotation/translation from YAML."""
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

    # ------------------------------------------------- wrist-pose pipeline setup
    def _declare_pipeline_params(self):
        """Declare and read the wrist-pose pipeline parameters.

        All lengths are metres (the pipeline runs on metric joints:
        triangulated world units * effective_scale).
        """
        defaults = {
            "nominal_fps": 30.0,
            "template_file": "/workspace/ros2_ws/src/handpose_estimation/"
                             "config/hand_template.yaml",
            "publish_fitted_skeleton": True,
            "weight_source": "product",
            "reproj_resid_scale": 2.0,
            "weighted_dlt.enabled": True,
            "reprojection_residual.enabled": True,
            "reprojection_residual.publish_debug": False,
            "reachability_gate.enabled": True,
            "reachability_gate.d_min": 0.10,
            "reachability_gate.d_max": 0.85,
            "reachability_gate.behind_margin": 0.10,
            "reachability_gate.forward_axis": [0.0, 0.0, 1.0],
            "procrustes.enabled": True,
            "procrustes.min_joints": 6,
            "ransac.enabled": True,
            "ransac.iterations": 50,
            "ransac.sample_size": 4,
            "ransac.inlier_thresh": 0.02,
            "kalman.enabled": True,
            "kalman.process_noise_pos": 10.0,
            "kalman.measurement_noise_pos": 0.0006,
            "kalman.gate_threshold": 11.345,
            "kalman.orientation_lpf": 0.5,
            "kalman.max_coast_frames": 10,
            "one_euro.enabled": True,
            "one_euro.min_cutoff": 1.0,
            "one_euro.beta": 0.007,
            "one_euro.d_cutoff": 1.0,
        }
        for name, val in defaults.items():
            self.declare_parameter(name, val)
        self.pipe_params = {n: self.get_parameter(n).value for n in defaults}
        self.pipe_flags = {
            "weighted_dlt": bool(self.pipe_params["weighted_dlt.enabled"]),
            "reprojection_residual": bool(
                self.pipe_params["reprojection_residual.enabled"]
            ),
            "reachability_gate": bool(
                self.pipe_params["reachability_gate.enabled"]
            ),
            "procrustes": bool(self.pipe_params["procrustes.enabled"]),
            "ransac": bool(self.pipe_params["ransac.enabled"]),
            "kalman": bool(self.pipe_params["kalman.enabled"]),
            "one_euro": bool(self.pipe_params["one_euro.enabled"]),
        }
        self.publish_fitted_skeleton = bool(
            self.pipe_params["publish_fitted_skeleton"]
        )

    def _load_template_and_shell(self):
        """Load the canonical 21-landmark template + reachability shell (m)."""
        path = self.pipe_params["template_file"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Hand template file not found: {path}")
        with open(path, "r") as f:
            doc = yaml.safe_load(f)
        self.hand_template = np.asarray(
            doc["template"]["landmarks"], dtype=float
        )
        if self.hand_template.shape != (N_LANDMARKS, 3):
            raise ValueError(
                f"hand template must be ({N_LANDMARKS}, 3), "
                f"got {self.hand_template.shape}"
            )
        shell = doc.get("shell", {})
        self.reach_shell = ReachabilityShell(
            shoulder_left=shell.get("shoulder_left", [-0.18, 0.25, 0.0]),
            shoulder_right=shell.get("shoulder_right", [0.18, 0.25, 0.0]),
            d_min=self.pipe_params["reachability_gate.d_min"],
            d_max=self.pipe_params["reachability_gate.d_max"],
            forward_axis=self.pipe_params["reachability_gate.forward_axis"],
            behind_margin=self.pipe_params["reachability_gate.behind_margin"],
        )

    def _build_trackers(self):
        """One WristTracker per hand; the left hand gets a y-mirrored template."""
        self.trackers = {}
        for i, hand in enumerate(HAND_LABELS):
            tmpl = self.hand_template.copy()
            if hand == "Left":
                tmpl[:, 1] *= -1.0  # template is right-handed; mirror y
            self.trackers[hand] = WristTracker(
                handedness=hand,
                template=tmpl,
                flags=self.pipe_flags,
                params=self.pipe_params,
                shell=self.reach_shell,
                rng_seed=i,
            )

    # ------------------------------------------------------------ camera poses
    def _broadcast_camera_poses(self):
        """Publish world->camera static transforms (mode-dependent).

        - extrinsics mode: extrinsics are world->camera (X_cam = R X_world + t),
          so the pose in world is the inverse: orientation R^T, centre -R^T t.
        - stereo mode: rectified cameras share orientation (identity); each
          camera centre in the left rectified frame is -K^-1 P[:,3].

        Centres are multiplied by effective_scale to match the hand markers.
        """
        stamp = self.get_clock().now().to_msg()
        transforms = []
        for name in self.camera_names:
            if self.mode == "stereo":
                c = self.calib[name]
                center = -np.linalg.inv(c["k"]) @ c["p"][:, 3]
                R_wc = np.eye(3)
            else:
                R, t = self.extrinsics[name]
                R_wc = R.T
                center = -R_wc @ t
            center = center * self.effective_scale
            q = rotation_matrix_to_quaternion(R_wc)

            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.world_frame
            tf.child_frame_id = name
            tf.transform.translation.x = float(center[0])
            tf.transform.translation.y = float(center[1])
            tf.transform.translation.z = float(center[2])
            tf.transform.rotation.x = float(q[0])
            tf.transform.rotation.y = float(q[1])
            tf.transform.rotation.z = float(q[2])
            tf.transform.rotation.w = float(q[3])
            transforms.append(tf)
        self.static_tf_broadcaster.sendTransform(transforms)
        self.get_logger().info(
            f"Broadcast camera poses to TF: {self.camera_names}"
        )

    def _publish_camera_markers(self):
        """Draw a small frustum per camera in its own (optical) frame."""
        d = self.camera_marker_size
        w, h = d * 0.6, d * 0.45
        # Optical-frame convention: x right, y down, z forward.
        corners = [(-w, -h, d), (w, -h, d), (w, h, d), (-w, h, d)]

        array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, name in enumerate(self.camera_names):
            m = Marker()
            m.header.frame_id = name
            m.header.stamp = stamp
            m.ns = "camera_frustum"
            m.id = i
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = max(d * 0.02, 0.002)
            m.color = ColorRGBA(r=0.2, g=0.8, b=1.0, a=1.0)
            m.pose.orientation.w = 1.0

            apex = Point(x=0.0, y=0.0, z=0.0)
            cpts = [Point(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in corners]
            for cp in cpts:  # apex -> each corner
                m.points.append(apex)
                m.points.append(cp)
            for j in range(4):  # rectangle around the far plane
                m.points.append(cpts[j])
                m.points.append(cpts[(j + 1) % 4])
            array.markers.append(m)
        self.camera_marker_pub.publish(array)

    # --------------------------------------------------------------- callbacks
    def _on_camera_info(self, msg, name):
        if self.calib[name] is not None:
            return  # calibration is static; capture once
        d = np.array(msg.d, dtype=float).ravel()
        if d.size == 0:
            d = np.zeros(5)  # "no distortion advertised" -> zeros
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
        """Both calibrations are in — set the triangulation mode + poses."""
        if self.use_camera_info_extrinsics:
            self.mode = "stereo"
            self.effective_scale = 1.0  # rectified P is already metric
            P1 = self.calib[self.camera_names[1]]["p"]
            baseline = max(abs(P1[0, 3]), abs(P1[1, 3]))
            if baseline <= 1e-9:
                self.get_logger().warn(
                    "use_camera_info_extrinsics=true, but camera_info P has no "
                    "baseline (P[0,3]=P[1,3]=0): the cameras are not jointly "
                    "stereo-calibrated. Triangulation will be degenerate."
                )
            else:
                self.get_logger().info(
                    "Triangulation mode: STEREO from camera_info "
                    f"(baseline={baseline / P1[0, 0]:.4f} m); extrinsics.yaml ignored."
                )
        else:
            self.mode = "extrinsics"
            self.effective_scale = self.scale
            for name in self.camera_names:
                R, t = self.extrinsics[name]
                self.P_ext[name] = make_projection_matrix(
                    self.calib[name]["k"], R, t
                )
            self.get_logger().info(
                "Triangulation mode: EXTRINSICS (extrinsics.yaml + raw K + DLT)."
            )

        self.ready = True
        if self.publish_camera_pose:
            self._broadcast_camera_poses()
            self._publish_camera_markers()

    def _on_images(self, *msgs):
        # Calibration on both cameras must arrive before we can triangulate.
        if not self.ready:
            self.get_logger().warn(
                "Waiting for camera_info on all cameras...",
                throttle_duration_sec=5.0,
            )
            return

        timestamp_ms = self._frame_idx * 33  # monotonically increasing for VIDEO mode
        self._frame_idx += 1

        # Per camera, detect all hands keyed by handedness: {label: (21, 2)}.
        hands_2d = []
        scores_2d = []
        for i, (name, msg) in enumerate(zip(self.camera_names, msgs)):
            frame_bgr = self._decode_to_bgr(msg)

            # ===== HOTFIX(rotate-90): TEMPORARY =============================
            # Un-rotate the rosbag image (see _HOTFIX_ROTATE at top of file).
            # DELETE this block once the upstream publishes upright images.
            if _HOTFIX_ROTATE is not None:
                frame_bgr = cv2.rotate(frame_bgr, _HOTFIX_ROTATE)
            # ===== END HOTFIX ===============================================

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Use the (possibly rotated) frame's own dimensions for scaling.
            h, w = frame_bgr.shape[:2]
            hands, scores = self._detect_hands(
                self.detectors[i], frame_rgb, timestamp_ms, w, h
            )
            if i == 1 and self.swap_handedness_camera1:
                hands = {self._other_label(lbl): kp for lbl, kp in hands.items()}
                scores = {self._other_label(lbl): s for lbl, s in scores.items()}
            hands_2d.append(hands)
            scores_2d.append(scores)

            if self.publish_annotated and name in self.annotated_pubs:
                self._publish_annotated(name, frame_bgr.copy(), hands, msg.header)

        # Match hands across cameras by handedness label, triangulate each
        # (Stage A1 weighted DLT + Stage A2 reprojection residuals).
        stamp = msgs[0].header.stamp
        stamp_sec = stamp.sec + stamp.nanosec * 1e-9
        points_3d_by_hand = {}
        fitted_by_hand = {}
        for label in HAND_LABELS:
            kp0 = hands_2d[0].get(label)
            kp1 = hands_2d[1].get(label)
            if kp0 is None or kp1 is None:
                # Hand missing in a view: let the tracker coast (Stage D).
                res = self.trackers[label].update(
                    stamp_sec,
                    np.full((N_LANDMARKS, 3), np.nan),
                    np.zeros(N_LANDMARKS),
                    0.0,
                )
                self._publish_wrist_pose(label, res, stamp)
                continue

            view_scores = [scores_2d[0][label], scores_2d[1][label]]
            points_3d, residuals = self._triangulate_hand(kp0, kp1, view_scores)
            points_3d_by_hand[label] = points_3d

            # Run the wrist-pose pipeline in metres.
            joints_m = points_3d * self.effective_scale
            weights = self._per_joint_weights(view_scores, residuals)
            agg_conf = float(np.clip(np.mean(view_scores), 0.5, 1.0))
            res = self.trackers[label].update(
                stamp_sec, joints_m, weights, agg_conf
            )
            self._publish_wrist_pose(label, res, stamp)

            if self.pipe_params["reprojection_residual.publish_debug"]:
                self.get_logger().info(
                    f"{label} reproj resid px: mean={np.nanmean(residuals):.2f} "
                    f"max={np.nanmax(residuals):.2f}",
                    throttle_duration_sec=1.0,
                )

            # Fitted-template skeleton (R*template+t, metres) for RViz A/B.
            if (
                self.publish_fitted_skeleton
                and res is not None
                and res["valid"]
                and self.trackers[label].last_fit is not None
            ):
                R, t = self.trackers[label].last_fit
                tmpl = self.trackers[label].template
                fitted_by_hand[label] = (R @ tmpl.T).T + t

        # Periodic visibility into handedness agreement across the two views.
        self.get_logger().info(
            f"cam0={sorted(hands_2d[0])} cam1={sorted(hands_2d[1])} "
            f"-> triangulated {sorted(points_3d_by_hand)}",
            throttle_duration_sec=5.0,
        )

        self._publish_markers(points_3d_by_hand, stamp, fitted_by_hand)

    @staticmethod
    def _other_label(label):
        return "Right" if label == "Left" else "Left"

    def _detect_hands(self, detector, frame_rgb, timestamp_ms, width, height):
        """Run the landmarker; return ({label: (21, 2) pixels}, {label: score}).

        The score is MediaPipe's handedness confidence (>= 0.5) — the only
        per-hand confidence the Tasks API exposes (per-landmark visibility/
        presence are never populated; MediaPipe issue #5212). If the same
        label is reported twice (rare), the higher-confidence hand wins so
        each of Left/Right maps to a single detection.
        """
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb)
        )
        result = detector.detect_for_video(mp_image, timestamp_ms)
        hands, scores = {}, {}
        if result.hand_landmarks:
            for lm_list, handed in zip(result.hand_landmarks, result.handedness):
                label = handed[0].category_name  # "Left" / "Right"
                score = handed[0].score
                if label in hands and score <= scores[label]:
                    continue
                hands[label] = np.array(
                    [[lm.x * width, lm.y * height] for lm in lm_list], dtype=float
                )
                scores[label] = score
        return hands, scores

    def _undistort_points(self, name, pts):
        """Undistort + rectify pixel points into the camera's rectified frame."""
        c = self.calib[name]
        src = np.ascontiguousarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        if c["model"] == "fisheye":
            out = cv2.fisheye.undistortPoints(
                src, c["k"], c["d"][:4].reshape(1, 4), R=c["r"], P=c["p"]
            )
        else:  # plumb_bob / rational_polynomial
            out = cv2.undistortPoints(src, c["k"], c["d"], R=c["r"], P=c["p"])
        return out.reshape(-1, 2)

    def _triangulate_hand(self, kp0, kp1, view_scores=None):
        """Triangulate one matched hand's 21 landmarks via (weighted) DLT.

        Stage A1: per-view weights (the handedness scores) scale each camera's
        DLT rows when `weighted_dlt.enabled` is true.
        Stage A2: each joint's mean reprojection residual (pixels) is returned
        as a per-joint trust signal.

        - stereo mode: points are undistorted/rectified with camera_info
          K/D/R/P first, then triangulated against the two rectified P
          matrices (which carry the baseline).
        - extrinsics mode: raw pixels against P = K [R|t] from extrinsics.yaml.

        Returns:
            points_3d: (21, 3) world-unit points (NaN where missing).
            residuals: (21,) mean reprojection residual in pixels (NaN where
                missing).
        """
        n0, n1 = self.camera_names
        points_3d = np.full((N_LANDMARKS, 3), np.nan)
        residuals = np.full(N_LANDMARKS, np.nan)
        valid = ~(np.isnan(kp0[:, 0]) | np.isnan(kp1[:, 0]))
        if not valid.any():
            return points_3d, residuals
        idx = np.where(valid)[0]

        if self.mode == "stereo":
            pts0 = np.full((N_LANDMARKS, 2), np.nan)
            pts1 = np.full((N_LANDMARKS, 2), np.nan)
            pts0[idx] = self._undistort_points(n0, kp0[idx])
            pts1[idx] = self._undistort_points(n1, kp1[idx])
            P0, P1 = self.calib[n0]["p"], self.calib[n1]["p"]
        else:
            pts0, pts1 = kp0, kp1
            P0, P1 = self.P_ext[n0], self.P_ext[n1]

        weights = None
        if self.pipe_flags["weighted_dlt"] and view_scores is not None:
            weights = np.asarray(view_scores, dtype=float)

        for p in idx:
            X, mean_resid, _ = triangulate_point(
                [P0, P1], [pts0[p], pts1[p]], weights
            )
            points_3d[p] = X
            residuals[p] = mean_resid
        return points_3d, residuals

    def _per_joint_weights(self, view_scores, residuals):
        """Synthesise per-joint weights (Stage A1/A2 outputs -> Stage C input).

        The Tasks-API HandLandmarker provides no per-landmark confidence
        (visibility/presence always 0, MediaPipe issue #5212), so weights are
        formed per the `weight_source` parameter from the per-hand handedness
        score and/or the per-joint reprojection residual.
        """
        src = self.pipe_params["weight_source"]
        hand_w = float(np.clip(np.mean(view_scores), 0.5, 1.0))
        if self.pipe_flags["reprojection_residual"]:
            rs = float(self.pipe_params["reproj_resid_scale"])
            resid = np.nan_to_num(residuals, nan=1e6)
            reproj_w = 1.0 / (1.0 + resid / rs)  # per joint, in (0,1]
        else:
            reproj_w = np.ones(N_LANDMARKS)
        if src == "uniform":
            return np.ones(N_LANDMARKS)
        if src == "handedness":
            return np.full(N_LANDMARKS, hand_w)
        if src == "reprojection":
            return reproj_w
        return hand_w * reproj_w  # "product" (default)

    def _decode_to_bgr(self, msg):
        """Decode a sensor_msgs/Image to a contiguous bgr8 ndarray.

        Honors msg.encoding (rgb8/bgr8/rgba8/bgra8/mono8) and msg.step (row
        stride / padding). The previous code hard-assumed bgr8 with no padding,
        which silently swaps R/B (poor MediaPipe detection) for rgb8 sources or
        shears the image when rows are padded.
        """
        enc = (msg.encoding or "bgr8").lower()
        channels = {
            "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8uc1": 1,
        }.get(enc, 3)

        buf = np.frombuffer(msg.data, dtype=np.uint8)
        step = msg.step if msg.step else msg.width * channels
        # Reshape by stride, then drop any trailing row padding.
        arr = buf[: step * msg.height].reshape(msg.height, step)
        arr = arr[:, : msg.width * channels].reshape(msg.height, msg.width, channels)

        if enc == "rgb8":
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif enc == "rgba8":
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif enc == "bgra8":
            bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif enc in ("mono8", "8uc1"):
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:  # bgr8 or unknown 3-channel
            bgr = arr[:, :, :3]
        return np.ascontiguousarray(bgr)

    # ------------------------------------------------------------- publishing
    def _publish_wrist_pose(self, label, res, stamp):
        """Publish one hand's 6-DoF wrist pose (metres, world frame).

        ``res`` is the WristTracker output; None (track lost/reset) publishes
        nothing, a coasting result (valid=False) still publishes the
        prediction so downstream consumers bridge short dropouts.
        """
        if res is None:
            return
        msg = PoseStamped()
        msg.header.frame_id = self.world_frame
        msg.header.stamp = stamp
        p, q = res["pos"], res["quat"]
        msg.pose.position.x = float(p[0])
        msg.pose.position.y = float(p[1])
        msg.pose.position.z = float(p[2])
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        self.wrist_pubs[label].publish(msg)

    def _publish_markers(self, points_3d_by_hand, stamp, fitted_by_hand=None):
        """Publish a joints + bones marker per hand.

        Both Left and Right are always published (with stable ids); a hand that
        is absent this frame is published with no points, which clears its
        previous skeleton in RViz instead of leaving it stale. When the
        wrist-pose pipeline produced a rigid fit, the fitted template skeleton
        (R*template+t, already in metres) is published alongside the raw
        joints for A/B comparison.
        """
        fitted_by_hand = fitted_by_hand or {}
        marker_array = MarkerArray()
        for label in HAND_LABELS:
            points_3d = points_3d_by_hand.get(label)
            color = HAND_COLORS[label]
            joint_id, bone_id = HAND_MARKER_IDS[label]

            joints = Marker()
            joints.header.frame_id = self.world_frame
            joints.header.stamp = stamp
            joints.ns = f"hand_{label.lower()}_joints"
            joints.id = joint_id
            joints.type = Marker.SPHERE_LIST
            joints.action = Marker.ADD
            joints.scale.x = joints.scale.y = joints.scale.z = self.joint_size
            joints.color = color
            joints.lifetime = Duration(sec=0, nanosec=200_000_000)
            joints.pose.orientation.w = 1.0

            bones = Marker()
            bones.header.frame_id = self.world_frame
            bones.header.stamp = stamp
            bones.ns = f"hand_{label.lower()}_bones"
            bones.id = bone_id
            bones.type = Marker.LINE_LIST
            bones.action = Marker.ADD
            bones.scale.x = self.line_width
            bones.color = color
            bones.lifetime = Duration(sec=0, nanosec=200_000_000)
            bones.pose.orientation.w = 1.0

            if points_3d is not None:
                def to_point(idx, _p3d=points_3d):
                    x, y, z = _p3d[idx] * self.effective_scale
                    return Point(x=float(x), y=float(y), z=float(z))

                valid = ~np.isnan(points_3d[:, 0])
                for p in range(N_LANDMARKS):
                    if valid[p]:
                        joints.points.append(to_point(p))
                for a, b in HAND_CONNECTIONS:
                    if valid[a] and valid[b]:
                        bones.points.append(to_point(a))
                        bones.points.append(to_point(b))

            marker_array.markers.append(joints)
            marker_array.markers.append(bones)

            # Fitted-template skeleton (wrist-pose pipeline Stage C output).
            # Positions are already metric — no effective_scale multiply.
            if self.publish_fitted_skeleton:
                fitted = fitted_by_hand.get(label)
                fcolor = FITTED_COLORS[label]
                fjoint_id, fbone_id = FITTED_MARKER_IDS[label]

                fjoints = Marker()
                fjoints.header.frame_id = self.world_frame
                fjoints.header.stamp = stamp
                fjoints.ns = f"hand_{label.lower()}_fitted_joints"
                fjoints.id = fjoint_id
                fjoints.type = Marker.SPHERE_LIST
                fjoints.action = Marker.ADD
                fjoints.scale.x = fjoints.scale.y = fjoints.scale.z = (
                    self.joint_size * 0.7
                )
                fjoints.color = fcolor
                fjoints.lifetime = Duration(sec=0, nanosec=200_000_000)
                fjoints.pose.orientation.w = 1.0

                fbones = Marker()
                fbones.header.frame_id = self.world_frame
                fbones.header.stamp = stamp
                fbones.ns = f"hand_{label.lower()}_fitted_bones"
                fbones.id = fbone_id
                fbones.type = Marker.LINE_LIST
                fbones.action = Marker.ADD
                fbones.scale.x = self.line_width * 0.7
                fbones.color = fcolor
                fbones.lifetime = Duration(sec=0, nanosec=200_000_000)
                fbones.pose.orientation.w = 1.0

                if fitted is not None:
                    fpts = [
                        Point(x=float(x), y=float(y), z=float(z))
                        for x, y, z in fitted
                    ]
                    fjoints.points.extend(fpts)
                    for a, b in HAND_CONNECTIONS:
                        fbones.points.append(fpts[a])
                        fbones.points.append(fpts[b])

                marker_array.markers.append(fjoints)
                marker_array.markers.append(fbones)
        self.marker_pub.publish(marker_array)

    def _publish_annotated(self, name, frame_bgr, hands, header):
        h, w = frame_bgr.shape[:2]
        # Draw every detected hand, color-coded by handedness.
        for label, kpts in hands.items():
            color = HAND_BGR.get(label, (255, 255, 255))
            pts = {
                p: (int(round(kpts[p, 0])), int(round(kpts[p, 1])))
                for p in range(N_LANDMARKS)
                if not np.isnan(kpts[p, 0])
            }
            for a, b in HAND_CONNECTIONS:
                if a in pts and b in pts:
                    cv2.line(frame_bgr, pts[a], pts[b], color, 2)
            for p in pts.values():
                cv2.circle(frame_bgr, p, 3, color, -1)

        img = Image()
        img.header = header
        img.height = h
        img.width = w
        img.encoding = "bgr8"
        img.is_bigendian = 0
        img.step = w * 3
        img.data = np.ascontiguousarray(frame_bgr).tobytes()
        self.annotated_pubs[name].publish(img)

    def shutdown(self):
        for d in self.detectors:
            d.close()


def main(args=None):
    rclpy.init(args=args)
    node = HandPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
