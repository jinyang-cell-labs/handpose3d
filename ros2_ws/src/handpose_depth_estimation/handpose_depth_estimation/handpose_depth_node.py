#!/usr/bin/env python3

"""
Per-joint 3D hand-pose estimation by triangulation of two selected cameras.

Idea
----
``mediapie_landmarks_extraction`` runs MediaPipe per camera and publishes, per
hand, the 21 2D image landmarks + handedness/score as
``handpose3d_msgs/HandLandmarks``. ``calibration_multi_cam`` publishes the rig
intrinsics (as intrinsics-only ``CameraInfo``: K + distortion, NO R/P) and the
extrinsics (``T_world_cam`` 4x4 per camera, world frame = first camera).

The upstream rig has THREE cameras; this node lets you pick exactly TWO
(``camera_names``, e.g. camera0 + camera2) and, unlike
``stereo_handpose_estimation`` (which triangulates a single robust centroid and
hangs MediaPipe's hand-local shape off it), triangulates **all 21 joints
independently** via DLT to recover the true metric 3D hand.

Pipeline, per synchronized landmark pair (same handedness matched across views):

1. For each of the 21 joints, take the 2D landmark in each view and triangulate
   with DLT using ``P = K @ [R|t]`` (world->camera from the extrinsics, raw K).
   Landmarks are assumed to already be in the undistorted pinhole image
   (``landmarks_undistorted=true``; the upstream node ran with undistortion on),
   so they are fed to DLT directly. If false, each point is undistorted with K/D
   first.
2. Publish the 21 world-frame joints per hand:
     * ``visualization_msgs/MarkerArray`` skeleton for RViz.
     * ``geometry_msgs/PoseArray`` (21 poses) per hand for downstream consumers.
3. Re-project the triangulated 3D joints back onto EACH selected camera's
   ``image_raw`` and publish an annotated image, so you can eyeball how well the
   3D reconstruction lines up with what the camera actually saw. The reprojected
   skeleton (solid) is drawn over the upstream 2D detection (hollow dots); the
   mean per-joint reprojection error (px) is overlaid as text. When
   ``landmarks_undistorted=true`` the image is undistorted first so both the
   detection and the pinhole reprojection live in the same coordinates.

Inputs
------
    <cam>/image_raw/landmarks/hands   handpose3d_msgs/HandLandmarks   (x2)
    <cam>/image_raw                   sensor_msgs/Image               (x2)
    <cam>/camera_info                 sensor_msgs/CameraInfo          (x2, latched)
    extrinsics_file (T_world_cam yaml, from calibration_multi_cam)

Outputs
-------
    handpose_depth/markers            visualization_msgs/MarkerArray  (RViz)
    handpose_depth/joints_left        geometry_msgs/PoseArray  (21 joints)
    handpose_depth/joints_right       geometry_msgs/PoseArray  (21 joints)
    <cam>/image_raw/depth/reprojected sensor_msgs/Image        (QA overlay, x2)
"""

import os
from collections import deque

import cv2
import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, PoseArray
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from handpose3d_msgs.msg import HandLandmarks

from handpose_depth_estimation.triangulation import dlt, make_projection_matrix

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

# RViz marker colors (RGBA) per handedness.
HAND_RGBA = {
    "Left": ColorRGBA(r=0.2, g=0.6, b=1.0, a=1.0),   # blue
    "Right": ColorRGBA(r=1.0, g=0.5, b=0.2, a=1.0),  # orange
}
# BGR overlay colors (OpenCV order): reprojected skeleton per hand, and the
# upstream 2D detection (drawn hollow so the gap = reprojection error).
HAND_BGR = {"Left": (255, 150, 50), "Right": (50, 150, 255)}
DETECTED_BGR = (60, 220, 60)  # green hollow dots = upstream 2D detection
# Stable (joints, bones) marker ids per hand so updates replace in place.
HAND_MARKER_IDS = {"Left": (0, 1), "Right": (2, 3)}


class HandposeDepthNode(Node):
    def __init__(self):
        super().__init__("handpose_depth_node")

        # --- parameters -----------------------------------------------------
        # Exactly TWO of the (three) published cameras to triangulate from.
        self.declare_parameter("camera_names", ["camera0", "camera2"])
        # Per-camera topics; left empty -> derived from camera_names as
        # <cam>/image_raw, <cam>/image_raw/landmarks/hands, <cam>/camera_info.
        self.declare_parameter("image_topics", [""])
        self.declare_parameter("landmark_topics", [""])
        self.declare_parameter("camera_info_topics", [""])
        self.declare_parameter("image_suffix", "/image_raw")
        self.declare_parameter("landmarks_suffix", "/image_raw/landmarks/hands")

        self.declare_parameter(
            "extrinsics_file",
            "/workspace/ros2_ws/src/calibration_multi_cam/config/extrinsics.yaml",
        )
        # Output frame. Empty -> use the extrinsics file's world_frame (the first
        # camera), which matches the TF tree published by calibration_multi_cam.
        self.declare_parameter("world_frame", "")

        # Upstream landmarks are already in the undistorted pinhole image
        # (mediapie_landmarks_extraction enable_undistortion=true): feed them to
        # DLT directly, reproject without distortion, and undistort image_raw
        # before overlaying. If false, undistort each 2D point with K/D before
        # DLT and reproject WITH the distortion model onto the raw image.
        self.declare_parameter("landmarks_undistorted", True)

        # Drop detections below this handedness/detection score.
        self.declare_parameter("min_score", 0.5)

        # --- reprojection QA overlay ---------------------------------------
        self.declare_parameter("reproject_overlay", True)
        self.declare_parameter("reprojected_suffix", "/depth/reprojected")
        # Also draw the upstream 2D detection (hollow) so the reprojection gap is
        # visible; the mean per-joint error (px) is always overlaid as text.
        self.declare_parameter("draw_detected", True)
        self.declare_parameter("line_thickness", 2)
        self.declare_parameter("point_radius", 4)
        # Recent image_raw frames kept per camera to pair with a (delayed)
        # landmark message by timestamp.
        self.declare_parameter("image_buffer_size", 30)
        # Max |stamp| difference (s) for an image to count as the landmark's frame.
        self.declare_parameter("image_match_tol", 0.05)

        # --- RViz markers ---------------------------------------------------
        self.declare_parameter("markers_topic", "handpose_depth/markers")
        self.declare_parameter("joint_size", 0.012)   # m, sphere diameter
        self.declare_parameter("line_width", 0.006)    # m, bone thickness

        # --- sync -----------------------------------------------------------
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("sync_queue_size", 20)

        # ---- resolve parameters -------------------------------------------
        self.camera_names = list(self.get_parameter("camera_names").value)
        if len(self.camera_names) != 2:
            raise ValueError(
                "handpose_depth_node requires exactly 2 camera_names, got "
                f"{self.camera_names}"
            )
        img_suffix = self.get_parameter("image_suffix").value
        lm_suffix = self.get_parameter("landmarks_suffix").value
        self.image_topics = self._resolve_topics("image_topics", img_suffix)
        self.landmark_topics = self._resolve_topics("landmark_topics", lm_suffix)
        self.camera_info_topics = self._resolve_topics(
            "camera_info_topics", "/camera_info"
        )

        self.extrinsics_file = self.get_parameter("extrinsics_file").value
        self.landmarks_undistorted = bool(
            self.get_parameter("landmarks_undistorted").value
        )
        self.min_score = float(self.get_parameter("min_score").value)
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

        # ---- extrinsics: world->camera (R, t) per camera, + world frame ----
        self.extrinsics, ext_world_frame = self._load_extrinsics(
            self.extrinsics_file
        )
        self.world_frame = (
            self.get_parameter("world_frame").value or ext_world_frame
        )

        # ---- per-camera calibration state (filled from camera_info) --------
        self.calib = {name: None for name in self.camera_names}
        self.P = {name: None for name in self.camera_names}
        self.undistort_map = {name: None for name in self.camera_names}
        self.ready = False

        # ---- image ring buffers (stamp_ns -> Image msg) per camera ---------
        self.image_buffers = {
            name: deque(maxlen=self.image_buffer_size)
            for name in self.camera_names
        }

        # ---- subscriptions / publishers -----------------------------------
        latching_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.info_subs = []
        for name, topic in zip(self.camera_names, self.camera_info_topics):
            self.info_subs.append(
                self.create_subscription(
                    CameraInfo,
                    topic,
                    lambda msg, n=name: self._on_camera_info(msg, n),
                    latching_qos,
                )
            )

        # image_raw kept in per-camera ring buffers for QA reprojection.
        self.image_subs = []
        if self.reproject_overlay:
            for name, topic in zip(self.camera_names, self.image_topics):
                self.image_subs.append(
                    self.create_subscription(
                        Image,
                        topic,
                        lambda msg, n=name: self._on_image(msg, n),
                        qos_profile_sensor_data,
                    )
                )

        # Landmark pair synchronized across the two views.
        lm_subs = [
            Subscriber(self, HandLandmarks, t) for t in self.landmark_topics
        ]
        self.sync = ApproximateTimeSynchronizer(
            lm_subs,
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop").value),
        )
        self.sync.registerCallback(self._on_landmarks)

        self.marker_pub = self.create_publisher(
            MarkerArray, self.get_parameter("markers_topic").value, 10
        )
        self.joints_pubs = {
            "Left": self.create_publisher(
                PoseArray, "handpose_depth/joints_left", 10
            ),
            "Right": self.create_publisher(
                PoseArray, "handpose_depth/joints_right", 10
            ),
        }
        self.reproj_pubs = {}
        if self.reproject_overlay:
            suffix = self.get_parameter("reprojected_suffix").value
            for name, img_topic in zip(self.camera_names, self.image_topics):
                self.reproj_pubs[name] = self.create_publisher(
                    Image, img_topic + suffix, qos_profile_sensor_data
                )

        self.get_logger().info(
            f"handpose_depth_node up: cameras={self.camera_names}, "
            f"world_frame='{self.world_frame}', "
            f"landmarks_undistorted={self.landmarks_undistorted}, "
            f"reproject_overlay={self.reproject_overlay}; waiting for "
            "camera_info on both cameras..."
        )

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
                    f"({len(vals)} vs {len(self.camera_names)})"
                )
            return vals
        return [name + suffix for name in self.camera_names]

    def _load_extrinsics(self, path):
        """Load ``T_world_cam`` (4x4, camera->world) and return world->camera.

        calibration_multi_cam writes, per camera, ``T_world_cam`` whose rotation
        ``R_wc`` and translation ``c`` place the camera in the world frame
        (X_world = R_wc X_cam + c). Triangulation/projection need world->camera:
        ``R_cw = R_wc^T``, ``t_cw = -R_cw c``. Returns
        ``({name: (R_cw, t_cw)}, world_frame)``.
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
            T = np.asarray(cameras[name]["T_world_cam"], dtype=float).reshape(4, 4)
            R_wc = T[:3, :3]
            c = T[:3, 3]
            R_cw = R_wc.T
            t_cw = -R_cw @ c
            ext[name] = (R_cw, t_cw)
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
        R_cw, t_cw = self.extrinsics[name]
        self.P[name] = make_projection_matrix(K, R_cw, t_cw)
        if self.landmarks_undistorted and msg.width and msg.height:
            # Undistort image_raw to the SAME pinhole frame the landmarks live
            # in (R=identity, output intrinsics = K) so detection + reprojection
            # overlay in the same coordinates.
            map1, map2 = cv2.initUndistortRectifyMap(
                K, d, None, K, (msg.width, msg.height), cv2.CV_16SC2
            )
            self.undistort_map[name] = (map1, map2)
        self.get_logger().info(f"Captured intrinsics for {name}")
        if all(self.calib[n] is not None for n in self.camera_names):
            self.ready = True
            self.get_logger().info("Both cameras calibrated; triangulating.")

    def _on_image(self, msg, name):
        stamp_ns = self._stamp_ns(msg.header.stamp)
        self.image_buffers[name].append((stamp_ns, msg))

    # --------------------------------------------------------------- callbacks
    def _on_landmarks(self, msg0, msg1):
        if not self.ready:
            self.get_logger().warn(
                "Waiting for camera_info on both cameras...",
                throttle_duration_sec=5.0,
            )
            return

        n0, n1 = self.camera_names
        hands0 = self._index_hands(msg0)
        hands1 = self._index_hands(msg1)
        stamp = msg0.header.stamp

        joints_by_hand = {}      # label -> (21, 3) world points (NaN where bad)
        for label in HAND_LABELS:
            h0 = hands0.get(label)
            h1 = hands1.get(label)
            if h0 is None or h1 is None:
                continue  # need the hand in both views
            joints_by_hand[label] = self._triangulate_joints(
                h0["image"], h1["image"]
            )

        self._publish_markers(joints_by_hand, stamp)
        self._publish_joint_poses(joints_by_hand, stamp)

        if self.reproject_overlay:
            self._publish_reprojections(
                {n0: (msg0, hands0), n1: (msg1, hands1)}, joints_by_hand
            )

        summary = {
            lbl: int(np.sum(np.all(np.isfinite(pts), axis=1)))
            for lbl, pts in joints_by_hand.items()
        }
        self.get_logger().info(
            f"cam0={sorted(hands0)} cam1={sorted(hands1)} -> "
            f"triangulated joints/hand={summary}",
            throttle_duration_sec=5.0,
        )

    # ------------------------------------------------------------- core maths
    def _index_hands(self, msg):
        """Index a HandLandmarks msg by handedness.

        Returns ``{label: {image: (21, 2), score}}``; duplicate labels keep the
        higher-confidence detection. Hands without 21 image landmarks are
        dropped (per-joint triangulation needs the full set).
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
                [[p.x, p.y] for p in hand.landmarks_image], dtype=float
            )
            out[label] = {"image": img, "score": float(hand.score)}
        return out

    def _triangulate_joints(self, pts0, pts1):
        """Triangulate 21 joint correspondences into world points.

        ``pts0``/``pts1`` are (21, 2) 2D landmarks in each view. Returns a
        (21, 3) array of world points; rows that are degenerate (or whose input
        is non-finite) are NaN.
        """
        n0, n1 = self.camera_names
        P0, P1 = self.P[n0], self.P[n1]
        out = np.full((N_LANDMARKS, 3), np.nan)
        for i in range(N_LANDMARKS):
            a, b = pts0[i], pts1[i]
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                continue
            if not self.landmarks_undistorted:
                a = self._undistort_point(n0, a)
                b = self._undistort_point(n1, b)
            X = dlt(P0, P1, a, b)
            if np.all(np.isfinite(X)):
                out[i] = X
        return out

    def _undistort_point(self, name, pt):
        c = self.calib[name]
        src = np.ascontiguousarray(pt, dtype=np.float64).reshape(-1, 1, 2)
        if c["model"] == "fisheye":
            out = cv2.fisheye.undistortPoints(
                src, c["k"], c["d"][:4].reshape(1, 4), P=c["k"]
            )
        else:
            out = cv2.undistortPoints(src, c["k"], c["d"], P=c["k"])
        return out.reshape(2)

    def _project(self, name, pts3d):
        """Project (21, 3) world points to (21, 2) pixels for camera ``name``.

        landmarks_undistorted -> pinhole projection via P (matches the
        undistorted image the overlay is drawn on). Else full distortion model
        (cv2.projectPoints with K/D) onto the raw image. NaN rows stay NaN.
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
            R_cw, t_cw = self.extrinsics[name]
            c = self.calib[name]
            rvec, _ = cv2.Rodrigues(R_cw)
            proj, _ = cv2.projectPoints(
                pts3d[valid].reshape(-1, 1, 3),
                rvec, t_cw.reshape(3, 1), c["k"], c["d"],
            )
            out[valid] = proj.reshape(-1, 2)
        return out

    # ------------------------------------------------------------- publishing
    def _publish_joint_poses(self, joints_by_hand, stamp):
        """Publish each hand's 21 triangulated joints as a PoseArray.

        Both hands always published (empty when absent) so a vanished hand
        clears downstream. Only finite joints are included.
        """
        for label in HAND_LABELS:
            pa = PoseArray()
            pa.header.frame_id = self.world_frame
            pa.header.stamp = stamp
            pts = joints_by_hand.get(label)
            if pts is not None:
                for i in range(N_LANDMARKS):
                    if not np.all(np.isfinite(pts[i])):
                        continue
                    pose = Pose()
                    pose.position.x = float(pts[i, 0])
                    pose.position.y = float(pts[i, 1])
                    pose.position.z = float(pts[i, 2])
                    pose.orientation.w = 1.0
                    pa.poses.append(pose)
            self.joints_pubs[label].publish(pa)

    def _publish_markers(self, joints_by_hand, stamp):
        """Publish the triangulated 21-joint skeleton per hand for RViz."""
        marker_array = MarkerArray()
        for label in HAND_LABELS:
            pts = joints_by_hand.get(label)
            color = HAND_RGBA[label]
            joint_id, bone_id = HAND_MARKER_IDS[label]

            joints = self._new_marker(
                label, joint_id, "joints", Marker.SPHERE_LIST, color, stamp
            )
            joints.scale.x = joints.scale.y = joints.scale.z = self.joint_size
            bones = self._new_marker(
                label, bone_id, "bones", Marker.LINE_LIST, color, stamp
            )
            bones.scale.x = self.line_width

            if pts is not None:
                fin = [
                    Point(x=float(pts[i, 0]), y=float(pts[i, 1]),
                          z=float(pts[i, 2]))
                    if np.all(np.isfinite(pts[i])) else None
                    for i in range(N_LANDMARKS)
                ]
                joints.points.extend(p for p in fin if p is not None)
                for a, b in HAND_CONNECTIONS:
                    if fin[a] is not None and fin[b] is not None:
                        bones.points.append(fin[a])
                        bones.points.append(fin[b])
            marker_array.markers.extend([joints, bones])
        self.marker_pub.publish(marker_array)

    def _new_marker(self, label, marker_id, suffix, mtype, color, stamp):
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = stamp
        m.ns = f"hand_{label.lower()}_{suffix}"
        m.id = marker_id
        m.type = mtype
        m.action = Marker.ADD
        m.color = color
        m.lifetime = Duration(sec=0, nanosec=300_000_000)
        m.pose.orientation.w = 1.0
        return m

    def _publish_reprojections(self, per_cam, joints_by_hand):
        """Reproject the 3D joints onto each camera's image_raw and publish.

        ``per_cam`` maps camera name -> (landmark_msg, indexed_hands). For each
        camera we find the buffered image_raw frame matching the landmark stamp,
        (optionally) undistort it, draw the reprojected skeleton + the upstream
        2D detection, overlay the mean per-joint reprojection error, and publish.
        """
        for name, (lm_msg, hands) in per_cam.items():
            img_msg = self._match_image(name, lm_msg.header.stamp)
            if img_msg is None:
                self.get_logger().warn(
                    f"[{name}] no buffered image_raw within "
                    f"{self.image_match_tol:.3f}s of landmarks; skipping overlay",
                    throttle_duration_sec=5.0,
                )
                continue
            frame = self._decode_to_bgr(img_msg)
            if self.landmarks_undistorted and self.undistort_map[name] is not None:
                m1, m2 = self.undistort_map[name]
                frame = cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)

            errors = []
            for label in HAND_LABELS:
                pts3d = joints_by_hand.get(label)
                if pts3d is None:
                    continue
                proj = self._project(name, pts3d)
                color = HAND_BGR[label]
                self._draw_skeleton(frame, proj, color)
                det = hands.get(label)
                if det is not None and self.draw_detected:
                    self._draw_detected(frame, det["image"])
                if det is not None:
                    errors.extend(self._joint_errors(proj, det["image"]))

            if errors:
                mean_err = float(np.mean(errors))
                cv2.putText(
                    frame, f"reproj err: {mean_err:.1f}px (n={len(errors)})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                )

            self._publish_image(name, frame, img_msg.header)

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
            msg.height, msg.width, channels
        )
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
    node = HandposeDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
