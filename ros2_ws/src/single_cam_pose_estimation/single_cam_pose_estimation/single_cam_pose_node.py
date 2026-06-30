#!/usr/bin/env python3

"""
Monocular model-based 6-DoF hand-pose estimation from EACH of N cameras.

Idea
----
``mediapie_landmarks_extraction`` runs MediaPipe per camera and publishes, per
hand, the 21 2D image landmarks AND MediaPipe's hand-local metric 3D model
(``hand_world_landmarks``, metres) as ``handpose3d_msgs/HandLandmarks``.
``calibration_multi_cam`` publishes the rig intrinsics (intrinsics-only
``CameraInfo``: K + distortion, NO R/P) and extrinsics (``T_world_cam`` 4x4 per
camera, world frame = first camera).

Unlike ``handpose_depth_estimation`` (which needs TWO cameras and triangulates
the 21 joints), this node estimates the hand pose from a SINGLE camera by
PnP: it fits the rigid 6-DoF transform ``T_world_hand`` that places the
hand-local model so it reprojects onto the detected pixels (with a cheirality
penalty to break the front/back mirror). See ``pose_estimation.py``.

Many cameras, independent estimates
-----------------------------------
``camera_names`` is a LIST (any length). Each camera is processed on its own —
its own landmarks, its own ``camera_info``, its own pose estimate — all expressed
in the shared world frame. Running several cameras at once lets you eyeball how
well the independent monocular estimates agree, which is a calibration sanity
check. There is NO synchronization between cameras (a single view is all each
estimate needs).

Pipeline, per camera, per detected hand (with both image + world landmarks):

1. ``estimate_hand_pose(K, T_world_cam, landmarks_image, landmarks_world)`` ->
   ``T_world_hand`` (hand-local -> world) by minimising reprojection error.
2. Place the hand-local model into the world: ``X_world = X_hand @ R^T + t``
   -> 21 world-frame joints.
3. Publish:
     * ``visualization_msgs/MarkerArray`` skeleton (per camera+hand namespace,
       one colour per camera so the views are distinguishable in RViz).
     * ``geometry_msgs/PoseArray`` (21 joints, world frame) per camera+hand.
     * ``geometry_msgs/PoseStamped`` 6-DoF hand pose per camera+hand.
     * TF ``world -> <cam>_hand_<label>`` so RViz draws the hand-frame axes.
4. (QA) Reproject the placed joints back onto the camera's ``image_raw`` and
   publish an annotated image, drawn over the upstream 2D detection with the
   mean per-joint reprojection error in pixels.

Inputs (per camera in ``camera_names``)
----------------------------------------
    <cam>/image_raw/landmarks/hands   handpose3d_msgs/HandLandmarks
    <cam>/image_raw                   sensor_msgs/Image          (QA overlay)
    <cam>/camera_info                 sensor_msgs/CameraInfo     (latched)
    extrinsics_file (T_world_cam yaml, from calibration_multi_cam)

Outputs
-------
    single_cam_pose/markers                       visualization_msgs/MarkerArray
    single_cam_pose/<cam>/joints_{left,right}     geometry_msgs/PoseArray
    single_cam_pose/<cam>/hand_pose_{left,right}  geometry_msgs/PoseStamped
    <cam>/image_raw/pose/reprojected              sensor_msgs/Image (QA, optional)
    TF: world -> <cam>_hand_{Left,Right}          (optional)
"""

import os
from collections import deque

import cv2
import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from handpose3d_msgs.msg import HandLandmarks

from single_cam_pose_estimation import pose_estimation as pe

# Hand skeleton connections (21 landmarks).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]
N_LANDMARKS = 21
HAND_LABELS = ("Left", "Right")

# A distinct colour per camera so several cameras' independent estimates are
# distinguishable in one RViz scene. Cycled if there are more cameras than hues.
CAM_RGBA = [
    ColorRGBA(r=0.20, g=0.60, b=1.00, a=1.0),   # blue
    ColorRGBA(r=1.00, g=0.50, b=0.20, a=1.0),   # orange
    ColorRGBA(r=0.20, g=0.80, b=0.30, a=1.0),   # green
    ColorRGBA(r=0.80, g=0.30, b=0.80, a=1.0),   # purple
    ColorRGBA(r=0.95, g=0.80, b=0.15, a=1.0),   # yellow
]
CAM_BGR = [
    (255, 150, 50), (50, 150, 255), (75, 200, 75),
    (200, 75, 200), (40, 200, 240),
]
DETECTED_BGR = (60, 220, 60)   # green hollow dots = upstream 2D detection


class SingleCamPoseNode(Node):
    def __init__(self):
        super().__init__("single_cam_pose_node")

        # --- parameters -----------------------------------------------------
        # ANY number of cameras; each estimated independently.
        self.declare_parameter("camera_names", ["camera0", "camera1", "camera2"])
        # Per-camera topics; left empty -> derived from camera_names as
        # <cam>{suffix} / <cam>/camera_info.
        self.declare_parameter("landmark_topics", [""])
        self.declare_parameter("image_topics", [""])
        self.declare_parameter("camera_info_topics", [""])
        self.declare_parameter("image_suffix", "/image_raw")
        self.declare_parameter("landmarks_suffix", "/image_raw/landmarks/hands")

        self.declare_parameter(
            "extrinsics_file",
            "/workspace/ros2_ws/src/calibration_multi_cam/config/extrinsics.yaml",
        )
        # Output frame. Empty -> use the extrinsics file's world_frame (the first
        # camera), matching the TF tree published by calibration_multi_cam.
        self.declare_parameter("world_frame", "")

        # Upstream landmarks are already in the undistorted pinhole image
        # (mediapie_landmarks_extraction enable_undistortion=true): the PnP cost
        # is pinhole, reprojection uses pinhole P, and image_raw is undistorted
        # before the overlay. If false, reprojection uses the full distortion
        # model onto the raw image (the PnP itself is still pinhole).
        self.declare_parameter("landmarks_undistorted", True)

        # Drop detections below this handedness/detection score.
        self.declare_parameter("min_score", 0.5)

        # --- PnP / cheirality (see pose_estimation.py) ----------------------
        self.declare_parameter("cheirality_margin", pe.CHEIRALITY_MARGIN_M)
        self.declare_parameter("cheirality_weight", pe.CHEIRALITY_WEIGHT_PX_PER_M)

        # --- 6-DoF hand pose / TF -------------------------------------------
        self.declare_parameter("publish_hand_pose", True)
        self.declare_parameter("publish_tf", True)

        # --- reprojection QA overlay ----------------------------------------
        self.declare_parameter("reproject_overlay", True)
        self.declare_parameter("reprojected_suffix", "/pose/reprojected")
        self.declare_parameter("draw_detected", True)
        self.declare_parameter("line_thickness", 2)
        self.declare_parameter("point_radius", 4)
        self.declare_parameter("image_buffer_size", 30)
        self.declare_parameter("image_match_tol", 0.05)

        # --- RViz markers ---------------------------------------------------
        self.declare_parameter("markers_topic", "single_cam_pose/markers")
        self.declare_parameter("joint_size", 0.012)   # m, sphere diameter
        self.declare_parameter("line_width", 0.006)    # m, bone thickness

        # ---- resolve parameters -------------------------------------------
        self.camera_names = list(self.get_parameter("camera_names").value)
        if not self.camera_names:
            raise ValueError("single_cam_pose_node needs >=1 camera_names")
        img_suffix = self.get_parameter("image_suffix").value
        lm_suffix = self.get_parameter("landmarks_suffix").value
        self.landmark_topics = self._resolve_topics("landmark_topics", lm_suffix)
        self.image_topics = self._resolve_topics("image_topics", img_suffix)
        self.camera_info_topics = self._resolve_topics(
            "camera_info_topics", "/camera_info"
        )

        self.extrinsics_file = self.get_parameter("extrinsics_file").value
        self.landmarks_undistorted = bool(
            self.get_parameter("landmarks_undistorted").value
        )
        self.min_score = float(self.get_parameter("min_score").value)
        self.cheirality_margin = float(
            self.get_parameter("cheirality_margin").value
        )
        self.cheirality_weight = float(
            self.get_parameter("cheirality_weight").value
        )
        self.publish_hand_pose = bool(
            self.get_parameter("publish_hand_pose").value
        )
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.reproject_overlay = bool(
            self.get_parameter("reproject_overlay").value
        )
        self.draw_detected = bool(self.get_parameter("draw_detected").value)
        self.line_thickness = int(self.get_parameter("line_thickness").value)
        self.point_radius = int(self.get_parameter("point_radius").value)
        self.image_buffer_size = int(self.get_parameter("image_buffer_size").value)
        self.image_match_tol = float(self.get_parameter("image_match_tol").value)
        self.joint_size = float(self.get_parameter("joint_size").value)
        self.line_width = float(self.get_parameter("line_width").value)

        # ---- extrinsics: T_world_cam (cam->world) per camera + world frame -
        self.T_world_cam, ext_world_frame = self._load_extrinsics(
            self.extrinsics_file
        )
        self.world_frame = (
            self.get_parameter("world_frame").value or ext_world_frame
        )

        # ---- per-camera calibration state (filled from camera_info) --------
        self.calib = {n: None for n in self.camera_names}     # {k,d,model,size}
        self.P = {n: None for n in self.camera_names}          # K[R_cw|t_cw]
        self.undistort_map = {n: None for n in self.camera_names}
        self.ready = {n: False for n in self.camera_names}
        # Stable per-camera colour / marker-id base.
        self.cam_idx = {n: i for i, n in enumerate(self.camera_names)}

        # ---- image ring buffers (stamp_ns -> Image) per camera -------------
        self.image_buffers = {
            n: deque(maxlen=self.image_buffer_size) for n in self.camera_names
        }

        # ---- subscriptions / publishers -----------------------------------
        latching_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.info_subs, self.image_subs, self.lm_subs = [], [], []
        for name in self.camera_names:
            info_topic = self.camera_info_topics[self.cam_idx[name]]
            self.info_subs.append(self.create_subscription(
                CameraInfo, info_topic,
                lambda msg, n=name: self._on_camera_info(msg, n), latching_qos))

            lm_topic = self.landmark_topics[self.cam_idx[name]]
            self.lm_subs.append(self.create_subscription(
                HandLandmarks, lm_topic,
                lambda msg, n=name: self._on_landmarks(msg, n), 10))

            if self.reproject_overlay:
                img_topic = self.image_topics[self.cam_idx[name]]
                self.image_subs.append(self.create_subscription(
                    Image, img_topic,
                    lambda msg, n=name: self._on_image(msg, n),
                    qos_profile_sensor_data))

        self.marker_pub = self.create_publisher(
            MarkerArray, self.get_parameter("markers_topic").value, 10)

        # Per-camera, per-hand joint/pose publishers.
        self.joints_pubs = {}     # (cam, label) -> PoseArray pub
        self.pose_pubs = {}       # (cam, label) -> PoseStamped pub
        for name in self.camera_names:
            for label in HAND_LABELS:
                self.joints_pubs[(name, label)] = self.create_publisher(
                    PoseArray, f"single_cam_pose/{name}/joints_{label.lower()}", 10)
                if self.publish_hand_pose:
                    self.pose_pubs[(name, label)] = self.create_publisher(
                        PoseStamped,
                        f"single_cam_pose/{name}/hand_pose_{label.lower()}", 10)

        self.reproj_pubs = {}
        if self.reproject_overlay:
            suffix = self.get_parameter("reprojected_suffix").value
            for name in self.camera_names:
                img_topic = self.image_topics[self.cam_idx[name]]
                self.reproj_pubs[name] = self.create_publisher(
                    Image, img_topic + suffix, qos_profile_sensor_data)

        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.get_logger().info(
            f"single_cam_pose_node up: cameras={self.camera_names}, "
            f"world_frame='{self.world_frame}', "
            f"landmarks_undistorted={self.landmarks_undistorted}, "
            f"reproject_overlay={self.reproject_overlay}; waiting for "
            "camera_info per camera...")

    # ------------------------------------------------------------------ setup
    def _resolve_topics(self, param, suffix):
        """Resolve a per-camera topic list parameter.

        Explicit non-empty entries must be 1:1 with camera_names; an empty list
        (or [""]) derives ``<camera>{suffix}``.
        """
        vals = [t for t in self.get_parameter(param).value if t]
        if vals:
            if len(vals) != len(self.camera_names):
                raise ValueError(
                    f"{param}, when set, must be 1:1 with camera_names "
                    f"({len(vals)} vs {len(self.camera_names)})")
            return vals
        return [name + suffix for name in self.camera_names]

    def _load_extrinsics(self, path):
        """Load ``T_world_cam`` (4x4, camera->world) per camera + world frame.

        Returns ``({name: T_world_cam (4,4)}, world_frame)``. The PnP estimator
        consumes ``T_world_cam`` directly (it computes world->cam internally).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Extrinsics file not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        world_frame = data.get("world_frame", "camera0")
        cameras = data["cameras"]
        ext = {}
        for name in self.camera_names:
            if name not in cameras:
                raise KeyError(f"No extrinsics for camera '{name}' in {path}")
            ext[name] = np.asarray(
                cameras[name]["T_world_cam"], dtype=float).reshape(4, 4)
        return ext, world_frame

    def _on_camera_info(self, msg, name):
        if self.calib[name] is not None:
            return  # intrinsics are static; capture once
        d = np.array(msg.d, dtype=float).ravel()
        if d.size == 0:
            d = np.zeros(5)
        K = np.array(msg.k, dtype=float).reshape(3, 3)
        self.calib[name] = {
            "k": K,
            "d": d,
            "model": (msg.distortion_model or "plumb_bob").lower(),
            "size": (int(msg.width), int(msg.height)),
        }
        # Reprojection projection matrix P = K[R_cw|t_cw] (world->cam).
        R_cw, t_cw = pe.world_to_cam(self.T_world_cam[name])
        self.P[name] = K @ np.hstack([R_cw, t_cw.reshape(3, 1)])
        if self.landmarks_undistorted and msg.width and msg.height:
            map1, map2 = cv2.initUndistortRectifyMap(
                K, d, None, K, (msg.width, msg.height), cv2.CV_16SC2)
            self.undistort_map[name] = (map1, map2)
        self.ready[name] = True
        self.get_logger().info(f"Captured intrinsics for {name}; estimating.")

    def _on_image(self, msg, name):
        self.image_buffers[name].append((self._stamp_ns(msg.header.stamp), msg))

    # --------------------------------------------------------------- callback
    def _on_landmarks(self, msg, name):
        if not self.ready[name]:
            self.get_logger().warn(
                f"[{name}] waiting for camera_info...",
                throttle_duration_sec=5.0)
            return

        hands = self._index_hands(msg)        # {label: {image,(21,2), world,(21,3)}}
        stamp = msg.header.stamp
        K = self.calib[name]["k"]
        T = self.T_world_cam[name]

        joints_by_hand = {}     # label -> (21,3) world joints
        pose_by_hand = {}       # label -> (4,4) T_world_hand
        for label, h in hands.items():
            if h["world"] is None:
                continue        # PnP needs MediaPipe's hand-local model
            r = pe.estimate_hand_pose(
                K, T, h["image"], h["world"],
                cheirality_margin=self.cheirality_margin,
                cheirality_weight=self.cheirality_weight)
            if not r.success:
                continue
            joints_by_hand[label] = self._place_in_world(r.T_world_hand, h["world"])
            pose_by_hand[label] = r.T_world_hand

        self._publish_markers(name, joints_by_hand, stamp)
        self._publish_joint_poses(name, joints_by_hand, stamp)
        if self.publish_hand_pose:
            self._publish_hand_poses(name, pose_by_hand, stamp)
        if self.tf_broadcaster is not None:
            self._broadcast_tf(name, pose_by_hand, stamp)
        if self.reproject_overlay:
            self._publish_reprojection(name, msg, hands, joints_by_hand)

        summary = {l: int(np.sum(np.all(np.isfinite(J), axis=1)))
                   for l, J in joints_by_hand.items()}
        self.get_logger().info(
            f"[{name}] hands={sorted(hands)} -> posed joints/hand={summary}",
            throttle_duration_sec=5.0)

    # ------------------------------------------------------------- core maths
    def _index_hands(self, msg):
        """Index a HandLandmarks msg by handedness.

        Returns ``{label: {image:(21,2), world:(21,3)|None, score}}``; duplicate
        labels keep the higher-confidence detection. Hands without 21 image
        landmarks are dropped.
        """
        out = {}
        for hand in msg.hands:
            if hand.score < self.min_score:
                continue
            label = hand.handedness
            if label in out and hand.score <= out[label]["score"]:
                continue
            if len(hand.landmarks_image) != N_LANDMARKS:
                continue
            img = np.array(
                [[p.x, p.y] for p in hand.landmarks_image], dtype=float)
            world = None
            if len(hand.landmarks_world) == N_LANDMARKS:
                world = np.array(
                    [[p.x, p.y, p.z] for p in hand.landmarks_world], dtype=float)
            out[label] = {"image": img, "world": world,
                          "score": float(hand.score)}
        return out

    @staticmethod
    def _place_in_world(T_world_hand, world_model):
        """Place the hand-local model into the world via the estimated pose.

        ``X_world = X_hand @ R^T + t`` for the (21,3) hand-local model.
        """
        R = T_world_hand[:3, :3]
        t = T_world_hand[:3, 3]
        return world_model @ R.T + t

    # ------------------------------------------------------------- publishing
    def _publish_joint_poses(self, name, joints_by_hand, stamp):
        """Publish each hand's 21 placed joints as a PoseArray (world frame).

        Both hands always published (empty when absent) so a vanished hand
        clears downstream. Only finite joints are included.
        """
        for label in HAND_LABELS:
            pa = PoseArray()
            pa.header.frame_id = self.world_frame
            pa.header.stamp = stamp
            J = joints_by_hand.get(label)
            if J is not None:
                for i in range(N_LANDMARKS):
                    if not np.all(np.isfinite(J[i])):
                        continue
                    pose = Pose()
                    pose.position.x = float(J[i, 0])
                    pose.position.y = float(J[i, 1])
                    pose.position.z = float(J[i, 2])
                    pose.orientation.w = 1.0
                    pa.poses.append(pose)
            self.joints_pubs[(name, label)].publish(pa)

    def _publish_hand_poses(self, name, pose_by_hand, stamp):
        """Publish the 6-DoF hand frame (T_world_hand) per hand as PoseStamped.

        A present hand carries its estimated translation + orientation quaternion;
        an absent hand publishes an identity pose at the origin so the topic still
        ticks (consumers can treat origin/identity as "no detection").
        """
        for label in HAND_LABELS:
            ps = PoseStamped()
            ps.header.frame_id = self.world_frame
            ps.header.stamp = stamp
            T = pose_by_hand.get(label)
            if T is not None:
                q = Rotation.from_matrix(T[:3, :3]).as_quat()  # (x,y,z,w)
                ps.pose.position.x = float(T[0, 3])
                ps.pose.position.y = float(T[1, 3])
                ps.pose.position.z = float(T[2, 3])
                ps.pose.orientation.x = float(q[0])
                ps.pose.orientation.y = float(q[1])
                ps.pose.orientation.z = float(q[2])
                ps.pose.orientation.w = float(q[3])
            else:
                ps.pose.orientation.w = 1.0
            self.pose_pubs[(name, label)].publish(ps)

    def _broadcast_tf(self, name, pose_by_hand, stamp):
        """Broadcast world -> <cam>_hand_<label> so RViz draws the hand axes."""
        for label, T in pose_by_hand.items():
            q = Rotation.from_matrix(T[:3, :3]).as_quat()  # (x,y,z,w)
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.world_frame
            tf.child_frame_id = f"{name}_hand_{label}"
            tf.transform.translation.x = float(T[0, 3])
            tf.transform.translation.y = float(T[1, 3])
            tf.transform.translation.z = float(T[2, 3])
            tf.transform.rotation.x = float(q[0])
            tf.transform.rotation.y = float(q[1])
            tf.transform.rotation.z = float(q[2])
            tf.transform.rotation.w = float(q[3])
            self.tf_broadcaster.sendTransform(tf)

    def _publish_markers(self, name, joints_by_hand, stamp):
        """Publish the placed 21-joint skeleton per hand for RViz.

        One colour per camera (so several cameras' estimates are distinguishable);
        a per-camera+hand namespace and stable ids so updates replace in place.
        """
        ci = self.cam_idx[name]
        color = CAM_RGBA[ci % len(CAM_RGBA)]
        marker_array = MarkerArray()
        for h, label in enumerate(HAND_LABELS):
            J = joints_by_hand.get(label)
            base = (ci * len(HAND_LABELS) + h) * 2     # unique (joint, bone) ids

            joints = self._new_marker(
                name, label, base, "joints", Marker.SPHERE_LIST, color, stamp)
            joints.scale.x = joints.scale.y = joints.scale.z = self.joint_size
            bones = self._new_marker(
                name, label, base + 1, "bones", Marker.LINE_LIST, color, stamp)
            bones.scale.x = self.line_width

            if J is not None:
                fin = [
                    Point(x=float(J[i, 0]), y=float(J[i, 1]), z=float(J[i, 2]))
                    if np.all(np.isfinite(J[i])) else None
                    for i in range(N_LANDMARKS)
                ]
                joints.points.extend(p for p in fin if p is not None)
                for a, b in HAND_CONNECTIONS:
                    if fin[a] is not None and fin[b] is not None:
                        bones.points.append(fin[a])
                        bones.points.append(fin[b])
            marker_array.markers.extend([joints, bones])
        self.marker_pub.publish(marker_array)

    def _new_marker(self, cam, label, marker_id, suffix, mtype, color, stamp):
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = stamp
        m.ns = f"{cam}_hand_{label.lower()}_{suffix}"
        m.id = marker_id
        m.type = mtype
        m.action = Marker.ADD
        m.color = color
        m.lifetime = Duration(sec=0, nanosec=300_000_000)
        m.pose.orientation.w = 1.0
        return m

    def _publish_reprojection(self, name, lm_msg, hands, joints_by_hand):
        """Reproject the placed 3D joints onto image_raw and publish.

        Finds the buffered image_raw frame matching the landmark stamp,
        (optionally) undistorts it, draws the reprojected skeleton + the upstream
        2D detection, overlays the mean per-joint reprojection error, publishes.
        """
        img_msg = self._match_image(name, lm_msg.header.stamp)
        if img_msg is None:
            self.get_logger().warn(
                f"[{name}] no buffered image_raw within "
                f"{self.image_match_tol:.3f}s of landmarks; skipping overlay",
                throttle_duration_sec=5.0)
            return
        frame = self._decode_to_bgr(img_msg)
        if self.landmarks_undistorted and self.undistort_map[name] is not None:
            m1, m2 = self.undistort_map[name]
            frame = cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)

        color = CAM_BGR[self.cam_idx[name] % len(CAM_BGR)]
        errors = []
        for label in HAND_LABELS:
            J = joints_by_hand.get(label)
            if J is None:
                continue
            proj = self._project(name, J)
            self._draw_skeleton(frame, proj, color)
            det = hands.get(label)
            if det is not None and self.draw_detected:
                self._draw_detected(frame, det["image"])
            if det is not None:
                errors.extend(self._joint_errors(proj, det["image"]))

        if errors:
            cv2.putText(
                frame, f"reproj err: {np.mean(errors):.1f}px (n={len(errors)})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        self._publish_image(name, frame, img_msg.header)

    def _project(self, name, pts3d):
        """Project (21,3) world points to (21,2) pixels for camera ``name``.

        landmarks_undistorted -> pinhole via P (matches the undistorted overlay).
        Else full distortion model (cv2.projectPoints) onto the raw image. NaN
        rows stay NaN.
        """
        out = np.full((N_LANDMARKS, 2), np.nan)
        valid = np.all(np.isfinite(pts3d), axis=1)
        if not valid.any():
            return out
        if self.landmarks_undistorted:
            P = self.P[name]
            for i in np.nonzero(valid)[0]:
                xh = P @ np.array([pts3d[i, 0], pts3d[i, 1], pts3d[i, 2], 1.0])
                if abs(xh[2]) < 1e-9:
                    continue
                out[i] = xh[:2] / xh[2]
        else:
            R_cw, t_cw = pe.world_to_cam(self.T_world_cam[name])
            c = self.calib[name]
            rvec, _ = cv2.Rodrigues(R_cw)
            proj, _ = cv2.projectPoints(
                pts3d[valid].reshape(-1, 1, 3),
                rvec, t_cw.reshape(3, 1), c["k"], c["d"])
            out[valid] = proj.reshape(-1, 2)
        return out

    def _match_image(self, name, stamp):
        """Return the buffered image_raw msg closest to ``stamp`` within tol."""
        target = self._stamp_ns(stamp)
        best, best_dt = None, None
        for s_ns, msg in self.image_buffers[name]:
            dt = abs(s_ns - target)
            if best_dt is None or dt < best_dt:
                best, best_dt = msg, dt
        if best is None or best_dt > self.image_match_tol * 1e9:
            return None
        return best

    def _draw_skeleton(self, frame, pts2d, color):
        pts = {
            i: (int(round(pts2d[i, 0])), int(round(pts2d[i, 1])))
            for i in range(N_LANDMARKS)
            if np.all(np.isfinite(pts2d[i]))
        }
        for a, b in HAND_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], color, self.line_thickness)
        for p in pts.values():
            cv2.circle(frame, p, self.point_radius, color, -1)

    def _draw_detected(self, frame, pts2d):
        for i in range(N_LANDMARKS):
            if not np.all(np.isfinite(pts2d[i])):
                continue
            p = (int(round(pts2d[i, 0])), int(round(pts2d[i, 1])))
            cv2.circle(frame, p, self.point_radius + 2, DETECTED_BGR, 1)

    @staticmethod
    def _joint_errors(proj, det):
        errs = []
        for i in range(N_LANDMARKS):
            if np.all(np.isfinite(proj[i])) and np.all(np.isfinite(det[i])):
                errs.append(float(np.linalg.norm(proj[i] - det[i])))
        return errs

    def _publish_image(self, name, frame, header):
        h, w = frame.shape[:2]
        img = Image()
        img.header = header
        img.height = h
        img.width = w
        img.encoding = "bgr8"
        img.is_bigendian = 0
        img.step = w * 3
        img.data = np.ascontiguousarray(frame).tobytes()
        self.reproj_pubs[name].publish(img)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _stamp_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _decode_to_bgr(msg):
        """Decode a sensor_msgs/Image to a contiguous bgr8 ndarray.

        Honors msg.encoding (rgb8/bgr8/rgba8/bgra8/mono8) and msg.step (row
        stride) so rgb8 isn't R/B-swapped and padded rows aren't sheared.
        """
        enc = (msg.encoding or "bgr8").lower()
        channels = {
            "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8uc1": 1,
        }.get(enc, 3)
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        step = msg.step if msg.step else msg.width * channels
        arr = buf[: step * msg.height].reshape(msg.height, step)
        arr = arr[:, : msg.width * channels].reshape(
            msg.height, msg.width, channels)
        if enc == "rgb8":
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif enc == "rgba8":
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif enc == "bgra8":
            bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif enc in ("mono8", "8uc1"):
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            bgr = arr[:, :, :3]
        return np.ascontiguousarray(bgr)


def main(args=None):
    rclpy.init(args=args)
    node = SingleCamPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
