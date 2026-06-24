#!/usr/bin/env python3

"""
Stereo hand-pose estimation from MediaPipe landmark messages.

Idea
----
``mediapie_landmarks_extraction`` already runs MediaPipe per camera and
publishes, per hand, the 2D image landmarks, the metric ``hand_world_landmarks``
(hand-local shape, in metres) and the handedness — as
``handpose3d_msgs/HandLandmarks``. MediaPipe's world landmarks carry the hand's
*shape* but no absolute placement; a single view also can't give true metric
scale or world position.

This node recovers the missing absolute placement with stereo vision, robustly:

1. From each camera's HandLandmarks, per hand, compute the **centroid** of the
   21 2D image landmarks (mean x, y). One well-averaged point per hand is far
   more stable to triangulate than 21 noisy per-joint correspondences, and the
   cross-view match is trivial (same handedness label).
2. Triangulate the two centroids -> the hand's **3D position in the world
   frame** (DLT; stereo-from-camera_info or extrinsics + raw K, like
   handpose_estimation).
3. Place the metric hand shape at that position:

       final_landmark[i] = centroid_world + R_(world<-cam) @ hand_world[i]

   ``hand_world_landmarks`` are offsets from the hand center in the source
   camera's optical frame, so they are rotated into the world frame by that
   camera's camera->world rotation. With identity extrinsics this reduces to a
   plain ``centroid + hand_world`` (toggle: ``apply_camera_rotation``).

Inputs
------
    <cam>/.../landmarks/hands   handpose3d_msgs/HandLandmarks   (x2)
    <cam>/camera_info           sensor_msgs/CameraInfo          (x2)

Outputs
-------
    stereo_handpose/markers     visualization_msgs/MarkerArray  (RViz skeleton)
    stereo_handpose/hand_left   geometry_msgs/PoseStamped       (centroid pose)
    stereo_handpose/hand_right  geometry_msgs/PoseStamped
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
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import ColorRGBA
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from handpose3d_msgs.msg import HandLandmarks

from stereo_handpose_estimation.triangulation import (
    dlt,
    make_projection_matrix,
    rotation_matrix_to_quaternion,
)

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

# MediaPipe 21-landmark hand model, in index order. Used to map the
# centroid_filter_list names onto landmark indices.
JOINT_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

HAND_LABELS = ("Left", "Right")
HAND_COLORS = {
    "Left": ColorRGBA(r=0.2, g=0.6, b=1.0, a=1.0),   # blue
    "Right": ColorRGBA(r=1.0, g=0.5, b=0.2, a=1.0),  # orange
}
CENTROID_COLOR = ColorRGBA(r=1.0, g=1.0, b=0.2, a=1.0)  # yellow
# Stable (joints, bones, centroid) marker ids per hand.
HAND_MARKER_IDS = {"Left": (0, 1, 2), "Right": (3, 4, 5)}


class StereoHandPoseNode(Node):
    def __init__(self):
        super().__init__("stereo_handpose_node")

        # --- parameters -----------------------------------------------------
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
        # true  -> triangulate from the camera_info stereo calibration
        #          (undistort/rectify with K/D/R/P + the rectified P matrices).
        # false -> raw K + extrinsics.yaml + DLT (no distortion handling).
        self.declare_parameter("use_camera_info_extrinsics", False)
        # Set true when the upstream landmarks come from ALREADY-rectified images
        # (mediapie_landmarks_extraction enable_rectification=true). Rectification
        # is then done once on the image, so the 2D centroids are already in the
        # rectified (P) frame and are fed straight to DLT — the per-point
        # undistort/rectify step is skipped. Only affects STEREO mode.
        self.declare_parameter("enable_rectification", False)
        self.declare_parameter("world_frame", "world")
        # Rotate hand_world_landmarks from the source camera's optical frame
        # into the world frame before adding the triangulated centroid. With
        # identity extrinsics this is a no-op (final = centroid + hand_world).
        self.declare_parameter("apply_camera_rotation", True)
        # Optional per-axis sign flip applied to hand_world_landmarks before the
        # rotation, to reconcile MediaPipe's world-axis convention with the
        # camera optical frame if the skeleton renders mirrored/upside-down.
        self.declare_parameter("world_landmark_sign", [1.0, 1.0, 1.0])
        # Ignore detections below this handedness/detection score.
        self.declare_parameter("min_score", 0.5)
        # Landmark names (MediaPipe joint names, see JOINT_NAMES) to exclude from
        # the 2D image-centroid mean that becomes the hand's triangulated 3D
        # position. Empty -> all 21 landmarks contribute.
        self.declare_parameter("centroid_filter_list", [""])
        # Calibration world units -> metres (extrinsics mode; stereo mode = 1).
        self.declare_parameter("scale", 0.05)
        self.declare_parameter("joint_size", 0.012)
        self.declare_parameter("line_width", 0.006)
        self.declare_parameter("centroid_size", 0.03)
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("publish_camera_pose", True)
        self.declare_parameter("camera_marker_size", 0.08)

        self.camera_names = list(self.get_parameter("camera_names").value)
        if len(self.camera_names) != 2:
            raise ValueError("stereo_handpose_node requires exactly 2 cameras")
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
        self.apply_camera_rotation = bool(
            self.get_parameter("apply_camera_rotation").value
        )
        self.world_landmark_sign = np.array(
            self.get_parameter("world_landmark_sign").value, dtype=float
        )
        self.min_score = float(self.get_parameter("min_score").value)
        # Resolve centroid_filter_list -> indices kept for the centroid mean.
        filter_names = [
            n for n in self.get_parameter("centroid_filter_list").value if n
        ]
        unknown = [n for n in filter_names if n not in JOINT_NAMES]
        if unknown:
            self.get_logger().warn(
                f"centroid_filter_list has unknown landmark names {unknown}; "
                f"valid names are {JOINT_NAMES}"
            )
        filter_set = set(filter_names)
        self.centroid_idx = [
            i for i, name in enumerate(JOINT_NAMES) if name not in filter_set
        ]
        if not self.centroid_idx:
            self.get_logger().warn(
                "centroid_filter_list excludes all landmarks; "
                "falling back to all 21 for the centroid"
            )
            self.centroid_idx = list(range(N_LANDMARKS))
        if filter_names:
            self.get_logger().info(
                f"centroid uses {len(self.centroid_idx)}/{N_LANDMARKS} "
                f"landmarks; excluded={sorted(filter_set)}"
            )
        self.scale = float(self.get_parameter("scale").value)
        self.joint_size = float(self.get_parameter("joint_size").value)
        self.line_width = float(self.get_parameter("line_width").value)
        self.centroid_size = float(self.get_parameter("centroid_size").value)
        self.publish_camera_pose = bool(
            self.get_parameter("publish_camera_pose").value
        )
        self.camera_marker_size = float(
            self.get_parameter("camera_marker_size").value
        )

        # --- calibration state ---------------------------------------------
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

        # --- subscriptions & publishers ------------------------------------
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

        self.marker_pub = self.create_publisher(
            MarkerArray, "stereo_handpose/markers", 10
        )
        self.hand_pubs = {
            "Left": self.create_publisher(
                PoseStamped, "stereo_handpose/hand_left", 10
            ),
            "Right": self.create_publisher(
                PoseStamped, "stereo_handpose/hand_right", 10
            ),
        }

        self.static_tf_broadcaster = None
        self.camera_marker_pub = None
        if self.publish_camera_pose:
            self.static_tf_broadcaster = StaticTransformBroadcaster(self)
            latching_qos = QoSProfile(
                depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
            )
            self.camera_marker_pub = self.create_publisher(
                MarkerArray, "stereo_handpose/cameras", latching_qos
            )

        self.get_logger().info(
            f"stereo_handpose_node ready: cameras={self.camera_names}, "
            f"world_frame='{self.world_frame}', waiting for camera_info..."
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

    # --------------------------------------------------------------- callbacks
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
                    "baseline: cameras are not jointly stereo-calibrated. "
                    "Triangulation will be degenerate."
                )
            else:
                pre = (
                    "centroids pre-rectified upstream"
                    if self.enable_rectification
                    else "undistort/rectify per centroid"
                )
                self.get_logger().info(
                    "Triangulation mode: STEREO from camera_info "
                    f"(baseline={baseline / P1[0, 0]:.4f} m, {pre})."
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

        placed_by_hand = {}
        centroid_by_hand = {}
        for label in HAND_LABELS:
            h0 = hands0.get(label)
            h1 = hands1.get(label)
            if h0 is None or h1 is None:
                continue  # need the hand in both views to triangulate

            # Stereo-triangulate the 2D centroid -> world position (metres).
            centroid_w = self._triangulate_centroid(h0["centroid"], h1["centroid"])
            if not np.all(np.isfinite(centroid_w)):
                continue
            centroid_m = centroid_w * self.effective_scale
            centroid_by_hand[label] = centroid_m

            # Place the metric hand shape from the higher-confidence view.
            src, src_idx = (h0, 0) if h0["score"] >= h1["score"] else (h1, 1)
            if src["world"] is not None:
                R_align = self._world_align_rotation(src_idx)
                shape = (R_align @ (src["world"] * self.world_landmark_sign).T).T
                placed_by_hand[label] = centroid_m + shape

            self._publish_hand_pose(label, centroid_m, src_idx, stamp)

        self.get_logger().info(
            f"cam0={sorted(hands0)} cam1={sorted(hands1)} "
            f"-> placed {sorted(placed_by_hand)}",
            throttle_duration_sec=5.0,
        )
        self._publish_markers(placed_by_hand, centroid_by_hand, stamp)

    # ------------------------------------------------------------- core maths
    def _index_hands(self, msg):
        """Index a HandLandmarks msg by handedness.

        Returns {label: {centroid: (2,), score: float, world: (21,3)|None}}.
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
            if not img:
                continue
            # Centroid over only the non-filtered landmarks (centroid_idx). If
            # the message is missing landmarks for some indices, fall back to
            # averaging whatever is present.
            if len(img) == N_LANDMARKS:
                pts = [img[i] for i in self.centroid_idx]
            else:
                pts = list(img)
            cx = float(np.mean([p.x for p in pts]))
            cy = float(np.mean([p.y for p in pts]))
            world = None
            if len(hand.landmarks_world) == N_LANDMARKS:
                world = np.array(
                    [[p.x, p.y, p.z] for p in hand.landmarks_world], dtype=float
                )
            out[label] = {
                "centroid": np.array([cx, cy], dtype=float),
                "score": float(hand.score),
                "world": world,
            }
        return out

    def _triangulate_centroid(self, c0, c1):
        """Triangulate one 2D centroid correspondence into a 3D world point."""
        n0, n1 = self.camera_names
        if self.mode == "stereo":
            if self.enable_rectification:
                # Images were rectified upstream -> centroids already in the
                # rectified (P) frame; no per-point undistort/rectify needed.
                p0, p1 = c0, c1
            else:
                p0 = self._undistort_point(n0, c0)
                p1 = self._undistort_point(n1, c1)
            P0, P1 = self.calib[n0]["p"], self.calib[n1]["p"]
        else:
            p0, p1 = c0, c1
            P0, P1 = self.P_ext[n0], self.P_ext[n1]
        return dlt(P0, P1, p0, p1)

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

    def _world_align_rotation(self, src_idx):
        """Rotation taking source-camera optical-frame vectors into world.

        - apply_camera_rotation off: identity (final = centroid + hand_world).
        - stereo mode: identity (rectified reference frame == world).
        - extrinsics mode: R_(world<-cam) = R^T, where extrinsics give
          world->camera (X_cam = R X_world + t).
        """
        if not self.apply_camera_rotation or self.mode == "stereo":
            return np.eye(3)
        R, _ = self.extrinsics[self.camera_names[src_idx]]
        return R.T

    # ------------------------------------------------------------- publishing
    def _publish_hand_pose(self, label, centroid_m, src_idx, stamp):
        msg = PoseStamped()
        msg.header.frame_id = self.world_frame
        msg.header.stamp = stamp
        msg.pose.position.x = float(centroid_m[0])
        msg.pose.position.y = float(centroid_m[1])
        msg.pose.position.z = float(centroid_m[2])
        q = rotation_matrix_to_quaternion(self._world_align_rotation(src_idx))
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        self.hand_pubs[label].publish(msg)

    def _publish_markers(self, placed_by_hand, centroid_by_hand, stamp):
        """Publish placed hand skeleton + centroid per hand (metres, world).

        Both hands are always published (empty when absent) so a vanished hand
        clears in RViz instead of leaving a stale skeleton.
        """
        marker_array = MarkerArray()
        for label in HAND_LABELS:
            placed = placed_by_hand.get(label)
            centroid = centroid_by_hand.get(label)
            color = HAND_COLORS[label]
            joint_id, bone_id, centroid_id = HAND_MARKER_IDS[label]

            joints = self._new_marker(
                label, joint_id, "joints", Marker.SPHERE_LIST, color, stamp
            )
            joints.scale.x = joints.scale.y = joints.scale.z = self.joint_size

            bones = self._new_marker(
                label, bone_id, "bones", Marker.LINE_LIST, color, stamp
            )
            bones.scale.x = self.line_width

            cmark = self._new_marker(
                label, centroid_id, "centroid", Marker.SPHERE_LIST,
                CENTROID_COLOR, stamp
            )
            cmark.scale.x = cmark.scale.y = cmark.scale.z = self.centroid_size

            if placed is not None:
                pts = [Point(x=float(x), y=float(y), z=float(z))
                       for x, y, z in placed]
                joints.points.extend(pts)
                for a, b in HAND_CONNECTIONS:
                    bones.points.append(pts[a])
                    bones.points.append(pts[b])
            if centroid is not None:
                cmark.points.append(
                    Point(x=float(centroid[0]), y=float(centroid[1]),
                          z=float(centroid[2]))
                )

            marker_array.markers.extend([joints, bones, cmark])
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

    # ------------------------------------------------------------ camera poses
    def _broadcast_camera_poses(self):
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
        self.get_logger().info(f"Broadcast camera poses to TF: {self.camera_names}")

    def _publish_camera_markers(self):
        d = self.camera_marker_size
        w, h = d * 0.6, d * 0.45
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
            cpts = [Point(x=float(c[0]), y=float(c[1]), z=float(c[2]))
                    for c in corners]
            for cp in cpts:
                m.points.append(apex)
                m.points.append(cp)
            for j in range(4):
                m.points.append(cpts[j])
                m.points.append(cpts[(j + 1) % 4])
            array.markers.append(m)
        self.camera_marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = StereoHandPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
