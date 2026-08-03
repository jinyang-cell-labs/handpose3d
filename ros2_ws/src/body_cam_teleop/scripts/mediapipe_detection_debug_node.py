#!/usr/bin/env python3
"""Standalone MediaPipe 2D hand-landmark debug viewer (body_cam_teleop).

A self-contained diagnostic node: it opens a camera, runs MediaPipe
HandLandmarker on the frames, and publishes the raw 2D detections for RViz. It
does NOT publish teleop and shares no state with the teleop pipeline — it is
here to answer "what is MediaPipe actually seeing?".

With ``enable_stereo`` it additionally opens a SECOND camera
(``stereo_camera_device``), runs its own HandLandmarker + label tracker on it,
and triangulates every accepted hand whose resolved Left/Right label matches in
both views: pixels are undistorted through the fisheye intrinsics and
triangulated with the camera extrinsics (``intrinsics_file`` /
``extrinsics_file``, empty = the packaged config yamls; both camera names must
exist in them). The result is published as 3D skeletons on ``markers_3d``
(metres, in the extrinsics world frame = camera0's optical frame; the node adds
a static root_frame -> <world>_3d TF rotated so the RViz view matches the
image). Stats, detection logs and the landmarks message keep following the
primary camera; stereo only adds ``image_annotated_stereo`` and the 3D markers.
Stereo forces synchronous inference and runs a second detector, so it roughly
doubles the per-frame cost.

With ``enable_refine`` (default on) the published skeleton is not the raw
per-joint triangulation (whose depth noise lets bone lengths flex) but a rigid
fit: the MediaPipe world-landmark hand model, scaled by
``hand_size_scaling_factor``, posed by minimizing pixel reprojection error in
every camera that accepted the hand (Levenberg-Marquardt with a Huber loss —
the same problem the C++ pipeline solves with Ceres). Two views pin down both
position and orientation; when only ONE camera sees the hand the same solve
runs on that single view — orientation and bearing stay well constrained — and
the depth along the ray from the RIG CENTRE (midpoint of the two camera
centres) is taken from a per-hand EMA that only two-view frames update. The
raw triangulation is still drawn in grey (``draw_raw_triangulation``) so the
two can be compared in RViz.

With ``enable_pinhole_crops`` (stereo + refine) the fit model comes from a
SECOND, UmeTrack-style inference pass instead of the full-frame result. The
full frame is heavily fisheye-distorted, and MediaPipe's world-landmark
regression — trained on ordinary perspective images — returns a skewed 3D hand
shape for off-centre hands. So: pass 1 (the normal full-frame run) provides
the detection and the 2D pixel measurements; then for each accepted hand a
virtual PERSPECTIVE camera (``crop_size_px`` square, FOV adapted to the hand's
angular extent, optical axis rotated to pass through the hand) is rendered
from the fisheye image, and MediaPipe runs AGAIN on that crop — locally
distortion-free and in-distribution. The refine fit poses the crop's clean
WORLD landmarks against pass 1's 2D pixels. Falls back to the full-frame
world landmarks when the crop re-detect misses; the 3D label shows the model
source ([stereo|crop] vs [stereo|full]) and the annotated crops are published
on ``crops``. Cost: one extra inference per hand per frame (PoC).

Because it opens the camera itself it cannot run at the same time as
hand_landmarks_node on the SAME device (V4L gives exclusive access). Either
stop the pipeline, or point this node at a second camera.

Outputs (all under <ns>/mediapipe_debug/):
  image_annotated  sensor_msgs/Image - the frame with the 21-point skeleton,
                   landmark indices, handedness label + score and a per-frame
                   header drawn on it. View with an RViz Image display; this is
                   the main view.
  markers          visualization_msgs/MarkerArray - the same 2D landmarks laid
                   out on a virtual image plane in 3D (pixels scaled to
                   plane_width_m), so the RViz 3D view shows the skeleton too,
                   with a border rectangle for scale and a text label per hand.
  landmarks        handpose3d_msgs/HandLandmarks - the same message the real
                   pipeline publishes, so existing tooling can consume it.

Hands the filters would reject are still DRAWN (in reject_color, tagged with
the reason) when draw_rejected is true — the point of a debug view is to show
what a stricter setting would have thrown away. They are excluded from the
landmarks message, which mirrors the pipeline's behaviour exactly.

``mirror_input`` flips the frame horizontally BEFORE inference, which flips
MediaPipe's Left/Right classification: its handedness assumes a third-person,
unmirrored view, so an egocentric body camera often labels the operator's hand
as the opposite side. Toggle it and watch the per-window summary to find out
which setting labels your hand consistently.

The per-window summary counts detections per label, the mean handedness score,
and how many times the label FLIPPED - a hand that keeps changing side is what
makes the teleop pipeline's hand_filter_mode gate drop frames intermittently.

With ``use_label_tracker`` (default on) MediaPipe's per-frame handedness label
is replaced by hand_label_tracker.HandLabelTracker, which exploits the
egocentric rig's constraints (one operator, hands never cross): two visible
hands are labelled by wrist x-order, a lone hand keeps the sticky label of its
track, and a lone NEW hand fuses weak cues (image side, MediaPipe's vote,
exclusivity) for a few frames before committing. The overlay marks provisional
labels with "?" and tracker overrides with "(mp:<label>)"; the summary counts
``corrected_labels`` so the override rate is measurable.
"""
import json
import os
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from scipy.optimize import least_squares
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, TransformStamped
from hand_label_tracker import HandLabelTracker, Resolution
from handpose3d_msgs.msg import Hand, HandLandmarks
from rclpy.duration import Duration
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

N_LANDMARKS = 21

# Same skeleton the teleop pipeline draws (hand_pose_node kHandConnections).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                # palm base
)

# Quasi-rigid palm subset (wrist, thumb CMC, finger MCPs): moves with the palm
# regardless of articulation, so it anchors the depth ray / palm centroid.
PALM_IDX = (0, 1, 5, 9, 13, 17)

# Fixed slots of the world-models debug shelf: every raw hand_world_landmarks
# set a tick can produce (2 cameras x 2 hands x 2 inference passes).
WM_GRID = tuple(
    (cam, label, src)
    for cam in (0, 1)
    for label in ("Left", "Right")
    for src in ("full", "crop"))

# A per-hand fit state (warm start + filtered ray depth) older than this is
# forgotten: a hand gone for seconds may reappear anywhere.
FIT_STATE_TIMEOUT_SEC = 3.0

FONT = cv2.FONT_HERSHEY_SIMPLEX


class MediapipeDetectionDebugNode(Node):
    def __init__(self):
        super().__init__("mediapipe_detection_debug_node")

        # --- camera (same semantics as hand_landmarks_node) ------------------
        self.camera_name = self.declare_parameter("camera_name", "camera0").value
        camera_device = str(self.declare_parameter("camera_device", "0").value)
        self.frame_rate = float(self.declare_parameter("frame_rate", 30.0).value)
        self.capture_width = int(self.declare_parameter("capture_width", 1280).value)
        self.capture_height = int(self.declare_parameter("capture_height", 720).value)
        self.fourcc = str(self.declare_parameter("fourcc", "MJPG").value)

        # --- MediaPipe -------------------------------------------------------
        model_path = self.declare_parameter("model_path", "").value
        self.num_hands = int(self.declare_parameter("num_hands", 2).value)
        self.min_hand_detection_confidence = float(
            self.declare_parameter("min_hand_detection_confidence", 0.5).value)
        self.min_hand_presence_confidence = float(
            self.declare_parameter("min_hand_presence_confidence", 0.5).value)
        self.min_tracking_confidence = float(
            self.declare_parameter("min_tracking_confidence", 0.5).value)
        self.delegate = str(self.declare_parameter("delegate", "gpu").value).lower()
        self.async_inference = bool(
            self.declare_parameter("async_inference", False).value)
        # Flip the frame BEFORE inference: this flips MediaPipe's handedness
        # classification (see module docstring).
        self.mirror_input = bool(self.declare_parameter("mirror_input", False).value)

        # --- filters (mirrors the pipeline; rejects are drawn, not hidden) ---
        self.hand_filter_mode = str(
            self.declare_parameter("hand_filter_mode", "left_and_right").value).lower()
        self.min_handedness_confidence = float(
            self.declare_parameter("min_handedness_confidence", 0.0).value)
        allowed_by_mode = {
            "left_only": {"Left"},
            "right_only": {"Right"},
            "left_and_right": {"Left", "Right"},
        }
        if self.hand_filter_mode not in allowed_by_mode:
            raise ValueError(
                f"hand_filter_mode must be one of {sorted(allowed_by_mode)}, "
                f"got '{self.hand_filter_mode}'")
        self.allowed_labels = allowed_by_mode[self.hand_filter_mode]

        # --- left/right label tracker (see hand_label_tracker.py) ------------
        # Replaces MediaPipe's per-frame handedness label with the egocentric
        # position/track/cue logic. false = raw MediaPipe labels (old behaviour).
        self.use_label_tracker = bool(
            self.declare_parameter("use_label_tracker", True).value)
        # True: operator's left hand appears on the image's LEFT (unmirrored
        # egocentric camera). Set false for a mirrored feed.
        self.tracker_left_is_image_left = bool(
            self.declare_parameter("tracker_left_is_image_left", True).value)
        self.tracker_side_dead_zone_frac = float(
            self.declare_parameter("tracker_side_dead_zone_frac", 0.10).value)
        self.tracker_commit_frames = int(
            self.declare_parameter("tracker_commit_frames", 5).value)
        self.tracker_max_gap_sec = float(
            self.declare_parameter("tracker_max_gap_sec", 0.5).value)
        self.tracker_max_jump_frac = float(
            self.declare_parameter("tracker_max_jump_frac", 0.35).value)
        self.tracker_duplicate_sep_frac = float(
            self.declare_parameter("tracker_duplicate_sep_frac", 0.06).value)
        self.tracker = self._make_tracker()

        # --- annotated image -------------------------------------------------
        self.enable_annotated_image = bool(
            self.declare_parameter("enable_annotated_image", True).value)
        self.annotated_image_scale = float(
            self.declare_parameter("annotated_image_scale", 1.0).value)
        self.draw_connections = bool(
            self.declare_parameter("draw_connections", True).value)
        self.draw_landmark_ids = bool(
            self.declare_parameter("draw_landmark_ids", True).value)
        self.draw_handedness = bool(
            self.declare_parameter("draw_handedness", True).value)
        self.draw_bounding_box = bool(
            self.declare_parameter("draw_bounding_box", True).value)
        self.draw_header = bool(self.declare_parameter("draw_header", True).value)
        self.draw_rejected = bool(self.declare_parameter("draw_rejected", True).value)
        self.point_radius = int(self.declare_parameter("point_radius", 4).value)
        self.line_thickness = int(self.declare_parameter("line_thickness", 2).value)
        self.font_scale = float(self.declare_parameter("font_scale", 0.5).value)
        # BGR, matching the RViz marker colors below.
        self.left_color = self._color_param("left_color", [80, 220, 100])
        self.right_color = self._color_param("right_color", [60, 170, 255])
        self.reject_color = self._color_param("reject_color", [70, 70, 235])

        # --- RViz markers (2D landmarks on a virtual image plane) ------------
        self.enable_markers = bool(self.declare_parameter("enable_markers", True).value)
        self.root_frame = self.declare_parameter("root_frame", "mediapipe_debug").value
        # The image plane is plane_width_m wide; pixel coordinates scale onto it
        # so the aspect ratio is preserved and the 3D view matches the image.
        self.plane_width_m = float(self.declare_parameter("plane_width_m", 1.0).value)
        self.plane_distance_m = float(
            self.declare_parameter("plane_distance_m", 0.0).value)
        self.marker_point_size = float(
            self.declare_parameter("marker_point_size", 0.02).value)
        self.marker_line_width = float(
            self.declare_parameter("marker_line_width", 0.008).value)
        self.marker_text_size = float(
            self.declare_parameter("marker_text_size", 0.05).value)
        self.marker_lifetime_sec = float(
            self.declare_parameter("marker_lifetime_sec", 0.3).value)

        # --- landmarks message + logging -------------------------------------
        self.enable_landmarks = bool(
            self.declare_parameter("enable_landmarks", True).value)
        self.log_detections = bool(
            self.declare_parameter("log_detections", False).value)
        self.log_summary = bool(self.declare_parameter("log_summary", True).value)
        self.log_summary_period_sec = float(
            self.declare_parameter("log_summary_period_sec", 5.0).value)
        self.log_throttle_sec = float(
            self.declare_parameter("log_throttle_sec", 1.0).value)

        # --- topics ----------------------------------------------------------
        self.annotated_topic = self.declare_parameter(
            "annotated_topic", "mediapipe_debug/image_annotated").value
        self.markers_topic = self.declare_parameter(
            "markers_topic", "mediapipe_debug/markers").value
        self.landmarks_topic = self.declare_parameter(
            "landmarks_topic", "mediapipe_debug/landmarks").value
        self.stats_topic = self.declare_parameter(
            "stats_topic", "mediapipe_debug/stats").value

        # --- stereo triangulation ---------------------------------------------
        # Second camera + calibration -> triangulated 3D hand skeletons. Both
        # camera names must exist in the intrinsics and extrinsics yamls.
        self.enable_stereo = bool(
            self.declare_parameter("enable_stereo", False).value)
        self.stereo_camera_name = self.declare_parameter(
            "stereo_camera_name", "camera1").value
        stereo_camera_device = str(
            self.declare_parameter("stereo_camera_device", "4").value)
        self.intrinsics_file = self.declare_parameter("intrinsics_file", "").value
        self.extrinsics_file = self.declare_parameter("extrinsics_file", "").value
        self.stereo_annotated_topic = self.declare_parameter(
            "stereo_annotated_topic", "mediapipe_debug/image_annotated_stereo").value
        self.markers3d_topic = self.declare_parameter(
            "markers3d_topic", "mediapipe_debug/markers_3d").value
        # Reprojection refinement (see module docstring / yaml).
        self.enable_refine = bool(
            self.declare_parameter("enable_refine", True).value)
        self.hand_size_scaling_factor = float(
            self.declare_parameter("hand_size_scaling_factor", 1.3).value)
        self.refine_huber_px = float(
            self.declare_parameter("refine_huber_px", 5.0).value)
        self.refine_reject_rms_px = float(
            self.declare_parameter("refine_reject_rms_px", 25.0).value)
        self.ray_filter_alpha = float(
            self.declare_parameter("ray_filter_alpha", 0.25).value)
        self.draw_raw_triangulation = bool(
            self.declare_parameter("draw_raw_triangulation", True).value)
        # Two-pass virtual pinhole crops (see module docstring / yaml).
        self.enable_pinhole_crops = bool(
            self.declare_parameter("enable_pinhole_crops", False).value)
        self.crop_size_px = int(
            self.declare_parameter("crop_size_px", 256).value)
        self.crop_fov_margin = float(
            self.declare_parameter("crop_fov_margin", 1.8).value)
        self.crop_fov_min_deg = float(
            self.declare_parameter("crop_fov_min_deg", 30.0).value)
        self.crop_fov_max_deg = float(
            self.declare_parameter("crop_fov_max_deg", 80.0).value)
        self.crop_match_px = float(
            self.declare_parameter("crop_match_px", 25.0).value)
        self.publish_crops = bool(
            self.declare_parameter("publish_crops", True).value)
        self.crops_topic = self.declare_parameter(
            "crops_topic", "mediapipe_debug/crops").value
        # Debug shelf: every raw world-landmark set of the tick (up to 8) on a
        # fixed grid. Also makes the crop pass run on BOTH cameras.
        self.publish_world_models = bool(
            self.declare_parameter("publish_world_models", False).value)
        self.world_models_topic = self.declare_parameter(
            "world_models_topic", "mediapipe_debug/world_models").value
        if self.enable_stereo and self.async_inference:
            self.get_logger().warn(
                "async_inference is not supported with enable_stereo; "
                "falling back to synchronous VIDEO mode")
            self.async_inference = False

        if not model_path:
            model_path = os.path.join(
                get_package_share_directory("body_cam_teleop"),
                "models", "hand_landmarker.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"hand landmark model not found: {model_path}")

        # Shared between the executor thread and MediaPipe's result thread in
        # async mode: ts_ms -> (ros stamp, bgr frame, t_submit).
        self._pending = {}
        self._lock = threading.Lock()
        # VIDEO/LIVE_STREAM timestamps must be strictly increasing PER detector.
        self._last_ts_ms = {}
        self._diag_last = {}
        # Persists across summary windows: the label of the hand the pipeline
        # would follow, used to count flips.
        self._last_label = None
        self._reset_stats()

        self.detector = self._make_landmarker(model_path)
        self.cap = self._open_capture(camera_device)

        self.detector_stereo = None
        self.cap_stereo = None
        self.tracker_stereo = None
        self.detector_crop = None
        if self.enable_stereo:
            self._load_stereo_calibration()
            # Each camera needs its OWN landmarker (VIDEO mode carries tracking
            # state between frames) and its own label tracker.
            self.detector_stereo = self._make_landmarker(model_path)
            self.cap_stereo = self._open_capture(stereo_camera_device)
            self.tracker_stereo = self._make_tracker()
            # Per-label fit state: last pose (LM warm start) + filtered depth
            # along the rig-centre ray. Entries expire (see _fit_hand_pose).
            self._fit_state = {}
            if self.enable_pinhole_crops:
                # Second-pass landmarker for the virtual pinhole crops: IMAGE
                # mode (every crop is an independent image, no temporal
                # tracking), one hand per crop. Shared across cameras/labels.
                self.detector_crop = self._make_landmarker(
                    model_path, image_mode=True, num_hands=1)

        ns = self.get_namespace()
        prefix = "" if ns == "/" else ns.lstrip("/") + "/"
        self.root_frame_id = prefix + self.root_frame
        self.plane_frame_id = prefix + self.camera_name + "_image_plane"
        self.stereo_plane_frame_id = (
            prefix + self.stereo_camera_name + "_image_plane")

        self.annotated_pub = (
            self.create_publisher(Image, self.annotated_topic, qos_profile_sensor_data)
            if self.enable_annotated_image else None)
        self.markers_pub = (
            self.create_publisher(MarkerArray, self.markers_topic, 5)
            if self.enable_markers else None)
        self.landmarks_pub = (
            self.create_publisher(HandLandmarks, self.landmarks_topic, 5)
            if self.enable_landmarks else None)
        self.stats_pub = self.create_publisher(String, self.stats_topic, 5)
        self.stereo_annotated_pub = None
        self.markers3d_pub = None
        self.crops_pub = None
        self.world_models_pub = None
        if self.enable_stereo:
            if self.enable_annotated_image:
                self.stereo_annotated_pub = self.create_publisher(
                    Image, self.stereo_annotated_topic, qos_profile_sensor_data)
            self.markers3d_pub = self.create_publisher(
                MarkerArray, self.markers3d_topic, 5)
            if self.enable_pinhole_crops:
                self.crops_pub = self.create_publisher(
                    Image, self.crops_topic, qos_profile_sensor_data)
            self.world_models_pub = self.create_publisher(
                MarkerArray, self.world_models_topic, 5)

        # Static TF so RViz has a tree: root_frame -> <camera>_image_plane
        # (plus root_frame -> <world>_3d for stereo). One sendTransform call:
        # the latched static-TF sample must contain every transform.
        self.static_tf = StaticTransformBroadcaster(self)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.root_frame_id
        tf.child_frame_id = self.plane_frame_id
        tf.transform.translation.z = self.plane_distance_m
        tf.transform.rotation.w = 1.0
        tfs = [tf]
        if self.enable_stereo:
            # Triangulated points are in the extrinsics world frame — camera0's
            # OPTICAL frame (x right, y down, z = depth). Rotate 180 deg about x
            # so the RViz default view matches the annotated image: x right,
            # y up, depth increasing away from the viewer (-z).
            self.world3d_frame_id = prefix + self.stereo_world_frame + "_3d"
            tf3d = TransformStamped()
            tf3d.header.stamp = tf.header.stamp
            tf3d.header.frame_id = self.root_frame_id
            tf3d.child_frame_id = self.world3d_frame_id
            tf3d.transform.rotation.x = 1.0
            tf3d.transform.rotation.w = 0.0
            tfs.append(tf3d)
            # The world-models shelf floats beside the image plane, with the
            # same display flip so the raw (y-down) shapes render upright.
            self.world_models_frame_id = prefix + "world_models"
            tfw = TransformStamped()
            tfw.header.stamp = tf.header.stamp
            tfw.header.frame_id = self.root_frame_id
            tfw.child_frame_id = self.world_models_frame_id
            tfw.transform.translation.x = 0.75
            tfw.transform.translation.y = 0.45
            tfw.transform.rotation.x = 1.0
            tfw.transform.rotation.w = 0.0
            tfs.append(tfw)
        self.static_tf.sendTransform(tfs)

        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.summary_timer = self.create_timer(
            max(self.log_summary_period_sec, 0.1), self._publish_summary)
        self.timer = self.create_timer(1.0 / self.frame_rate, self._tick)

        self.get_logger().info(
            f"mediapipe_detection_debug_node up: {self.camera_name} "
            f"(dev {camera_device}) @ {self.frame_rate:.0f} fps, "
            f"delegate={self.delegate}, async={'on' if self.async_inference else 'off'}, "
            f"mirror_input={'on' if self.mirror_input else 'off'}, "
            f"filter={self.hand_filter_mode}, "
            f"label_tracker={'on' if self.use_label_tracker else 'off'}, "
            f"stereo={('on (' + self.stereo_camera_name + ' dev ' + stereo_camera_device + ')') if self.enable_stereo else 'off'}, "
            f"pinhole_crops={'on' if self.detector_crop is not None else 'off'} "
            f"-> {self.annotated_topic}"
            f"{' + ' + self.markers_topic if self.enable_markers else ''}"
            f"{' + ' + self.markers3d_topic if self.enable_stereo else ''}")

    # ------------------------------------------------------------------ setup
    def _make_tracker(self):
        return HandLabelTracker(
            left_is_image_left=self.tracker_left_is_image_left,
            side_dead_zone_frac=self.tracker_side_dead_zone_frac,
            commit_frames=self.tracker_commit_frames,
            max_gap_sec=self.tracker_max_gap_sec,
            max_jump_frac=self.tracker_max_jump_frac,
            duplicate_sep_frac=self.tracker_duplicate_sep_frac)

    def _color_param(self, name, default):
        value = list(self.declare_parameter(name, default).value)
        if len(value) != 3:
            raise ValueError(f"{name} must be a 3-element BGR list, got {value}")
        return tuple(int(v) for v in value)

    def _open_capture(self, source):
        dev = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            raise RuntimeError(
                f"failed to open camera device '{source}' — is the teleop "
                "pipeline already using it? (V4L access is exclusive)")
        # Order matters for V4L2: fourcc, then resolution.
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        # Keep at most one queued frame so a slow loop reads fresh frames.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (w, h) != (self.capture_width, self.capture_height):
            self.get_logger().warn(
                f"requested {self.capture_width}x{self.capture_height}, got {w}x{h} "
                "(V4L2 fell back to a supported mode)")
        return cap

    def _make_landmarker(self, model_path, image_mode=False, num_hands=None):
        """Full-frame landmarker by default; image_mode=True builds the
        stateless IMAGE-mode variant used for the virtual pinhole crops."""
        def build(delegate):
            delegate_enum = (mp_python.BaseOptions.Delegate.GPU
                             if delegate == "gpu"
                             else mp_python.BaseOptions.Delegate.CPU)
            if image_mode:
                running_mode = mp_vision.RunningMode.IMAGE
            elif self.async_inference:
                running_mode = mp_vision.RunningMode.LIVE_STREAM
            else:
                running_mode = mp_vision.RunningMode.VIDEO
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=model_path, delegate=delegate_enum),
                running_mode=running_mode,
                num_hands=num_hands or self.num_hands,
                min_hand_detection_confidence=self.min_hand_detection_confidence,
                min_hand_presence_confidence=self.min_hand_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                result_callback=(self._on_result
                                 if self.async_inference and not image_mode
                                 else None),
            )
            return mp_vision.HandLandmarker.create_from_options(options)

        try:
            return build(self.delegate)
        except Exception as exc:  # noqa: BLE001
            if self.delegate == "gpu":
                self.get_logger().warn(
                    f"GPU delegate unavailable ({exc}); falling back to CPU")
                return build("cpu")
            raise

    # ------------------------------------------------------------------ stereo
    def _load_stereo_calibration(self):
        """K/D per camera + world->cam projections for triangulation."""
        share = get_package_share_directory("body_cam_teleop")
        intr_path = self.intrinsics_file or os.path.join(
            share, "config", "intrinsics.yaml")
        extr_path = self.extrinsics_file or os.path.join(
            share, "config", "extrinsics.yaml")
        for path in (intr_path, extr_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"stereo calibration not found: {path}")
        with open(intr_path) as f:
            intr = yaml.safe_load(f)["cameras"]
        with open(extr_path) as f:
            extr = yaml.safe_load(f)
        self.stereo_world_frame = str(extr.get("world_frame", self.camera_name))
        self.calib = []        # per camera: (K, D, model, calibrated resolution)
        self.proj = []         # per camera: world->cam [R|t] 3x4 (normalized pts)
        self.T_world_cam = []  # per camera: cam->world 4x4
        self.focal = []        # per camera: mean focal (px), scales residuals
        centers = []
        for name in (self.camera_name, self.stereo_camera_name):
            if name not in intr:
                raise KeyError(f"camera '{name}' not in {intr_path}")
            if name not in extr.get("cameras", {}):
                raise KeyError(f"camera '{name}' not in {extr_path}")
            fx, fy, cx, cy = intr[name]["intrinsics"]
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
            D = np.asarray(intr[name]["distortion"], dtype=np.float64)
            model = str(intr[name].get("model", "pinhole"))
            res = [int(v) for v in intr[name].get(
                "resolution", [self.capture_width, self.capture_height])]
            # T_world_cam maps camera coords -> world; projecting a world point
            # into the camera needs the inverse.
            T = np.asarray(extr["cameras"][name]["T_world_cam"], dtype=np.float64)
            self.calib.append((K, D, model, res))
            self.proj.append(np.linalg.inv(T)[:3, :])
            self.T_world_cam.append(T)
            self.focal.append(float((fx + fy) / 2.0))
            centers.append(T[:3, 3])
        # Origin of the depth ray used by the one-view fallback: the rig
        # centre, so the ray is the same whichever camera sees the hand.
        self.ray_origin = np.mean(centers, axis=0)
        baseline = float(np.linalg.norm(centers[1] - centers[0]))
        self.get_logger().info(
            f"stereo calibration: {self.camera_name} + {self.stereo_camera_name}, "
            f"baseline {baseline * 100:.1f} cm, "
            f"world frame '{self.stereo_world_frame}'")

    def _undistort(self, pts_px, cam, size):
        """Pixels -> normalized, undistorted image coordinates (K removed)."""
        K, D, model, calib_res = self.calib[cam]
        w, h = size
        pts = np.asarray(pts_px, dtype=np.float64)
        if self.mirror_input:
            # Inference ran on the flipped frame; un-flip so the pixels match
            # the (unmirrored) calibration.
            pts = pts.copy()
            pts[:, 0] = (w - 1) - pts[:, 0]
        if [w, h] != calib_res:
            K = K.copy()
            K[0] *= w / calib_res[0]
            K[1] *= h / calib_res[1]
        pts = pts.reshape(-1, 1, 2)
        if "equi" in model or "fisheye" in model:
            und = cv2.fisheye.undistortPoints(pts, K, D)
        else:
            und = cv2.undistortPoints(pts, K, D)
        return und.reshape(-1, 2)

    def _triangulate(self, pts0, pts1, size0, size1):
        """Matched pixel landmarks -> Nx3 points in the extrinsics world frame."""
        n0 = self._undistort(pts0, 0, size0)
        n1 = self._undistort(pts1, 1, size1)
        X = cv2.triangulatePoints(self.proj[0], self.proj[1], n0.T, n1.T)
        return (X[:3] / X[3]).T

    def _crop_world_landmarks(self, frame, cam, pts_px, size):
        """UmeTrack-style second pass: render a virtual PERSPECTIVE camera
        aimed at the hand (optical axis through the pass-1 landmark centroid,
        FOV adapted to the hand's angular extent) from the fisheye frame, and
        re-run MediaPipe on it. The crop is locally distortion-free and
        in-distribution, so its world landmarks come back without the fisheye
        skew. `frame` must be the UNMIRRORED capture (the calibration's view).

        Returns (world 21x3 or None, annotated debug tile)."""
        # Hand direction + angular extent from the pass-1 landmarks.
        n = self._undistort(pts_px, cam, size)
        d = np.hstack([n, np.ones((len(n), 1))])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        c = d.mean(axis=0)
        c /= np.linalg.norm(c)
        extent = float(np.max(np.arccos(np.clip(d @ c, -1.0, 1.0))))
        half = float(np.clip(
            self.crop_fov_margin * extent,
            np.radians(self.crop_fov_min_deg) / 2.0,
            np.radians(self.crop_fov_max_deg) / 2.0))

        # Virtual camera axes in the real camera frame (rows of R): z through
        # the hand, x as close to the image x as possible so there is no roll
        # and the hand keeps its full-frame orientation.
        z = c
        x = np.array([1.0, 0.0, 0.0]) - z[0] * z
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        S = int(self.crop_size_px)
        f = (S / 2.0) / np.tan(half)
        K_v = np.array([[f, 0.0, S / 2.0],
                        [0.0, f, S / 2.0],
                        [0.0, 0.0, 1.0]])

        K, D, model, calib_res = self.calib[cam]
        w, h = size
        K = K.copy()
        if [w, h] != calib_res:
            K[0] *= w / calib_res[0]
            K[1] *= h / calib_res[1]
        if "equi" in model or "fisheye" in model:
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, R, K_v, (S, S), cv2.CV_16SC2)
        else:
            m1, m2 = cv2.initUndistortRectifyMap(
                K, D, R, K_v, (S, S), cv2.CV_16SC2)
        crop = cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self.detector_crop.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        world = None
        status = None
        if result.hand_world_landmarks:
            # Consistency gate: the crop detector happily locks onto a
            # DIFFERENT hand (the other hand or the arm entering the crop),
            # and fitting that shape against pass 1's pixels produces the
            # grotesque twisted skeletons. Re-project the crop's own 2D
            # landmarks back into the fisheye frame; the same physical hand
            # lands within a few px of the pass-1 landmarks, another hand
            # lands tens-to-hundreds of px away.
            crop_px = np.array(
                [[lm.x * S, lm.y * S] for lm in result.hand_landmarks[0]])
            dirs = np.column_stack([
                (crop_px - S / 2.0) / f, np.ones(len(crop_px))]) @ R
            if "equi" in model or "fisheye" in model:
                reproj, _ = cv2.fisheye.projectPoints(
                    dirs.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
            else:
                reproj, _ = cv2.projectPoints(
                    dirs.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
            ref = np.asarray(pts_px, dtype=np.float64)
            if self.mirror_input:
                ref = ref.copy()
                ref[:, 0] = (w - 1) - ref[:, 0]
            gap = float(np.mean(np.linalg.norm(
                reproj.reshape(-1, 2) - ref, axis=1)))
            for lm in result.hand_landmarks[0]:
                cv2.circle(crop, (int(lm.x * S), int(lm.y * S)), 2,
                           (80, 220, 100), -1, cv2.LINE_AA)
            if gap <= self.crop_match_px:
                world = np.array(
                    [[lm.x, lm.y, lm.z]
                     for lm in result.hand_world_landmarks[0]])
            else:
                status = f"MISMATCH {gap:.0f}px"
                self._diag(
                    f"crop_mismatch_{cam}",
                    f"crop cam{cam}: re-detected hand is {gap:.0f}px from the "
                    f"pass-1 hand (> crop_match_px "
                    f"{self.crop_match_px:.0f}) — different hand in the "
                    "crop; falling back to the full-frame model", "warn")
        else:
            status = "no hand"
        if status:
            cv2.putText(crop, status, (6, 34), FONT, 0.45,
                        (70, 70, 235), 1, cv2.LINE_AA)
        cv2.putText(crop, f"fov {np.degrees(2 * half):.0f}", (6, S - 8),
                    FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return world, crop

    def _publish_crops(self, tiles, stamp):
        """Side-by-side mosaic of this tick's annotated crops."""
        gap = np.full((tiles[0].shape[0], 4, 3), 40, np.uint8)
        mosaic = tiles[0]
        for tile in tiles[1:]:
            mosaic = np.hstack([mosaic, gap, tile])
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.plane_frame_id
        msg.height, msg.width = mosaic.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = mosaic.shape[1] * 3
        msg.data = np.ascontiguousarray(mosaic).tobytes()
        self.crops_pub.publish(msg)

    @staticmethod
    def _accepted_by_label(hands):
        """First ACCEPTED hand per resolved label -> {label: (pts_px, idx)}
        where idx indexes the MediaPipe result arrays (world landmarks)."""
        out = {}
        for label, _score, pts, accepted, _reason, idx, _res in hands:
            if accepted and label not in out:
                out[label] = (pts, idx)
        return out

    @staticmethod
    def _kabsch(A, B):
        """Rigid R, t (no scale) with B ~= A @ R.T + t, both Nx3."""
        ca, cb = A.mean(axis=0), B.mean(axis=0)
        H = (A - ca).T @ (B - cb)
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        return R, cb - R @ ca

    def _pnp_init(self, model_pts, n_meas, cam):
        """Cold-start pose from one view: SQPnP on the normalized detections,
        mapped from that camera's frame into the world frame."""
        ok, rvec, tvec = cv2.solvePnP(
            model_pts.astype(np.float64),
            n_meas.astype(np.float64).reshape(-1, 1, 2),
            np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        R_cam, _ = cv2.Rodrigues(rvec)
        T = self.T_world_cam[cam]
        R_w = T[:3, :3] @ R_cam
        t_w = T[:3, :3] @ tvec.ravel() + T[:3, 3]
        rvec_w, _ = cv2.Rodrigues(R_w)
        return np.concatenate([rvec_w.ravel(), t_w])

    def _fit_hand_pose(self, label, obs, model_pts, xyz_tri, now):
        """Reprojection refinement: solve the SE(3) pose of the (scaled)
        world-landmark hand model minimizing pixel reprojection error in every
        camera that accepted the hand (LM + Huber; the same problem the C++
        pipeline gives Ceres). obs = [(cam, pts_px, (w, h)), ...].

        Two views constrain depth; a one-view solve keeps orientation and
        bearing from reprojection and takes its depth along the rig-centre ray
        from the EMA that only two-view frames update.

        Returns (Nx3 world points, info dict) or (None, reason string)."""
        state = self._fit_state.get(label)
        if state is not None and now - state["t"] > FIT_STATE_TIMEOUT_SEC:
            del self._fit_state[label]
            state = None

        meas = [(cam, self._undistort(pts, cam, size)) for cam, pts, size in obs]

        def residuals(x):
            R, _ = cv2.Rodrigues(x[:3])
            Xw = model_pts @ R.T + x[3:6]
            res = []
            for cam, n in meas:
                P = self.proj[cam]
                Xc = Xw @ P[:, :3].T + P[:, 3]
                z = np.clip(Xc[:, 2:3], 1e-6, None)
                # Normalized-plane error scaled by f => pixel units, so
                # refine_huber_px / the reject threshold read as pixels.
                res.append(((Xc[:, :2] / z - n) * self.focal[cam]).ravel())
            return np.concatenate(res)

        if state is not None:
            x0 = np.concatenate([state["rvec"], state["tvec"]])
        elif xyz_tri is not None:
            R0, t0 = self._kabsch(model_pts, xyz_tri)
            rvec0, _ = cv2.Rodrigues(R0)
            x0 = np.concatenate([rvec0.ravel(), t0])
        else:
            x0 = self._pnp_init(model_pts, meas[0][1], obs[0][0])
            if x0 is None:
                return None, "PnP initialization failed"

        try:
            sol = least_squares(
                residuals, x0, loss="huber", f_scale=self.refine_huber_px,
                max_nfev=100)
        except Exception as exc:  # noqa: BLE001
            return None, f"solver error: {exc}"
        rms = float(np.sqrt(np.mean(sol.fun ** 2)))
        if rms > self.refine_reject_rms_px:
            return None, f"reproj rms {rms:.1f}px > {self.refine_reject_rms_px:.0f}px"

        R, _ = cv2.Rodrigues(sol.x[:3])
        tvec = sol.x[3:6]
        Xw = model_pts @ R.T + tvec
        palm = Xw[list(PALM_IDX)].mean(axis=0)
        ray = palm - self.ray_origin
        depth = float(np.linalg.norm(ray))
        if depth < 1e-6:
            return None, "degenerate ray"
        u = ray / depth

        views = len(obs)
        if views >= 2:
            # Stereo pins the depth: update the ray filter.
            if state is None:
                filt = depth
            else:
                a = self.ray_filter_alpha
                filt = a * depth + (1.0 - a) * state["depth"]
        elif state is not None:
            # One view: hold the filtered depth (mono depth rides entirely on
            # the model scale, so it never updates the filter).
            filt = state["depth"]
        else:
            # One view, no stereo history: trust the scaled-model depth.
            filt = depth
        if not 0.05 < filt < 5.0:
            return None, f"implausible depth {filt:.2f} m"
        shift = (filt - depth) * u
        tvec = tvec + shift
        Xw = Xw + shift

        rvec_out, _ = cv2.Rodrigues(R)
        self._fit_state[label] = {
            "rvec": rvec_out.ravel(), "tvec": tvec, "depth": filt, "t": now}
        return Xw, {"views": views, "rms": rms, "depth": filt,
                    "cam": obs[0][0]}

    # ----------------------------------------------------------- diagnostics
    def _on_set_parameters(self, params):
        """Drawing/filter/logging parameters are live-tunable."""
        live_bools = (
            "draw_connections", "draw_landmark_ids", "draw_handedness",
            "draw_bounding_box", "draw_header", "draw_rejected",
            "log_detections", "log_summary", "mirror_input",
            "enable_refine", "draw_raw_triangulation",
            "enable_pinhole_crops", "publish_crops", "publish_world_models")
        live_floats = (
            "font_scale", "min_handedness_confidence", "plane_width_m",
            "marker_point_size", "marker_line_width", "marker_text_size",
            "log_throttle_sec", "annotated_image_scale",
            "hand_size_scaling_factor", "refine_huber_px",
            "refine_reject_rms_px", "ray_filter_alpha",
            "crop_fov_margin", "crop_fov_min_deg", "crop_fov_max_deg",
            "crop_match_px")
        live_ints = ("point_radius", "line_thickness")
        tracker_bools = ("use_label_tracker", "tracker_left_is_image_left")
        tracker_floats = (
            "tracker_side_dead_zone_frac", "tracker_max_gap_sec",
            "tracker_max_jump_frac", "tracker_duplicate_sep_frac")
        rebuild_tracker = False
        for p in params:
            if p.name in live_bools:
                setattr(self, p.name, bool(p.value))
            elif p.name in live_floats:
                if p.value < 0.0:
                    return SetParametersResult(
                        successful=False, reason=f"{p.name} must be >= 0")
                setattr(self, p.name, float(p.value))
            elif p.name in live_ints:
                setattr(self, p.name, int(p.value))
            elif p.name in tracker_bools:
                setattr(self, p.name, bool(p.value))
                rebuild_tracker = True
            elif p.name in tracker_floats:
                if p.value < 0.0:
                    return SetParametersResult(
                        successful=False, reason=f"{p.name} must be >= 0")
                setattr(self, p.name, float(p.value))
                rebuild_tracker = True
            elif p.name == "tracker_commit_frames":
                if p.value < 1:
                    return SetParametersResult(
                        successful=False,
                        reason="tracker_commit_frames must be >= 1")
                self.tracker_commit_frames = int(p.value)
                rebuild_tracker = True
            elif p.name == "hand_filter_mode":
                mode = str(p.value).lower()
                allowed = {
                    "left_only": {"Left"},
                    "right_only": {"Right"},
                    "left_and_right": {"Left", "Right"},
                }
                if mode not in allowed:
                    return SetParametersResult(
                        successful=False,
                        reason=f"hand_filter_mode must be one of {sorted(allowed)}")
                self.hand_filter_mode = mode
                self.allowed_labels = allowed[mode]
        if rebuild_tracker:
            self.tracker = self._make_tracker()
            if self.tracker_stereo is not None:
                self.tracker_stereo = self._make_tracker()
        return SetParametersResult(successful=True)

    def _diag(self, key, msg, level="info"):
        now = time.monotonic()
        if now - self._diag_last.get(key, -1e9) < self.log_throttle_sec:
            return
        self._diag_last[key] = now
        # rclpy rejects a severity change at the same call site.
        if level == "warn":
            self.get_logger().warn(msg)
        else:
            self.get_logger().info(msg)

    def _reset_stats(self):
        self._stats = {
            "ticks": 0, "no_frame": 0, "results": 0, "frames_with_hand": 0,
            "raw_hands": 0, "rejected": 0, "label_flips": 0,
            "corrected": 0, "provisional": 0, "duplicates": 0,
            "stereo_pairs": 0,  # two-view 3D hands (stereo mode only)
            "mono_fits": 0,     # one-view refined 3D hands (stereo mode only)
            "crop_hits": 0,     # crop second pass found the hand
            "crop_misses": 0,   # crop second pass missed -> full-frame model
            "per_label": {},  # label -> [count, score_sum]
        }
        self._stats_window_start = time.monotonic()

    def _publish_summary(self):
        with self._lock:
            s = self._stats
            window = max(time.monotonic() - self._stats_window_start, 1e-6)
            self._reset_stats()
            last_label = self._last_label
        per_label = {
            label: {
                "count": n,
                "mean_score": round(total / n, 3) if n else 0.0,
            }
            for label, (n, total) in s["per_label"].items()
        }
        payload = {
            "node": self.get_fully_qualified_name(),
            "window_sec": round(window, 3),
            "fps": round(s["results"] / window, 2),
            "target_fps": self.frame_rate,
            "ticks": s["ticks"],
            "no_frame": s["no_frame"],
            "frames_with_hand": s["frames_with_hand"],
            "raw_hands": s["raw_hands"],
            "rejected_by_filter": s["rejected"],
            "label_flips": s["label_flips"],
            "label_tracker": self.use_label_tracker,
            "corrected_labels": s["corrected"],
            "provisional_labels": s["provisional"],
            "duplicate_hands": s["duplicates"],
            "current_label": last_label,
            "mirror_input": self.mirror_input,
            "hand_filter_mode": self.hand_filter_mode,
            "per_label": per_label,
        }
        if self.enable_stereo:
            payload["stereo_pairs"] = s["stereo_pairs"]
            payload["mono_fits"] = s["mono_fits"]
            if self.enable_pinhole_crops:
                payload["crop_hits"] = s["crop_hits"]
                payload["crop_misses"] = s["crop_misses"]
        msg = String()
        msg.data = json.dumps(payload)
        self.stats_pub.publish(msg)
        if not self.log_summary:
            return
        labels = " ".join(
            f"{label}: {v['count']} (score {v['mean_score']:.2f})"
            for label, v in sorted(per_label.items())) or "none"
        self.get_logger().info(
            f"[mediapipe {window:.1f}s] fps={s['results'] / window:.1f}/"
            f"{self.frame_rate:.0f} frames={s['results']} "
            f"with_hand={s['frames_with_hand']} raw={s['raw_hands']} | {labels} | "
            f"label flips={s['label_flips']} | tracker "
            f"{'corrections=' + str(s['corrected']) if self.use_label_tracker else 'off'}"
            f"{' provisional=' + str(s['provisional']) if s['provisional'] else ''}"
            f"{' duplicates=' + str(s['duplicates']) if s['duplicates'] else ''}"
            f" | rejected by {self.hand_filter_mode} filter={s['rejected']}"
            f"{' | stereo pairs=' + str(s['stereo_pairs']) + ' mono fits=' + str(s['mono_fits']) if self.enable_stereo else ''}"
            f"{' crop hit/miss=' + str(s['crop_hits']) + '/' + str(s['crop_misses']) if self.enable_stereo and self.enable_pinhole_crops else ''}")

    # --------------------------------------------------------------- capture
    def _next_ts_ms(self, stamp, cam):
        """VIDEO/LIVE_STREAM modes need strictly increasing per-detector ts."""
        ts_ms = stamp.sec * 1000 + stamp.nanosec // 1_000_000
        if ts_ms <= self._last_ts_ms.get(cam, -1):
            ts_ms = self._last_ts_ms[cam] + 1
        self._last_ts_ms[cam] = ts_ms
        return ts_ms

    def _tick(self):
        if self.enable_stereo:
            self._tick_stereo()
            return
        ret, frame = self.cap.read()
        stamp = self.get_clock().now().to_msg()
        with self._lock:
            self._stats["ticks"] += 1
            if not ret or frame is None:
                self._stats["no_frame"] += 1
        if not ret or frame is None:
            self.get_logger().warn("no frame from camera", throttle_duration_sec=5.0)
            return

        if self.mirror_input:
            frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts_ms = self._next_ts_ms(stamp, 0)

        if self.async_inference:
            with self._lock:
                self._pending[ts_ms] = (stamp, frame, time.perf_counter())
                while len(self._pending) > 8:
                    self._pending.pop(next(iter(self._pending)))
            self.detector.detect_async(mp_image, ts_ms)
        else:
            result = self.detector.detect_for_video(mp_image, ts_ms)
            self._handle_result(result, stamp, frame)

    def _detect_sync(self, detector, frame, stamp, cam):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return detector.detect_for_video(mp_image, self._next_ts_ms(stamp, cam))

    def _tick_stereo(self):
        """Two cameras, two detectors; triangulate label-matched hands."""
        ret0, frame0 = self.cap.read()
        ret1, frame1 = self.cap_stereo.read()
        stamp = self.get_clock().now().to_msg()
        good0 = ret0 and frame0 is not None
        good1 = ret1 and frame1 is not None
        with self._lock:
            self._stats["ticks"] += 1
            if not (good0 and good1):
                self._stats["no_frame"] += 1
        if not (good0 and good1):
            missing = " + ".join(
                name for name, good in ((self.camera_name, good0),
                                        (self.stereo_camera_name, good1))
                if not good)
            self.get_logger().warn(
                f"no frame from {missing}", throttle_duration_sec=5.0)
            return

        # The crop pass renders through the calibration, so it needs the
        # UNMIRRORED frames (cv2.flip below allocates new arrays).
        raw_frames = (frame0, frame1)
        if self.mirror_input:
            frame0 = cv2.flip(frame0, 1)
            frame1 = cv2.flip(frame1, 1)

        result0 = self._detect_sync(self.detector, frame0, stamp, 0)
        result1 = self._detect_sync(self.detector_stereo, frame1, stamp, 1)

        h0, w0 = frame0.shape[:2]
        h1, w1 = frame1.shape[:2]
        hands0 = self._resolve_hands(result0, w0, h0, self.tracker)
        hands1 = self._resolve_hands(result1, w1, h1, self.tracker_stereo)

        # Stats, detection logs, 2D markers and the landmarks message keep
        # following the primary camera; stereo adds its annotated image and
        # the triangulated 3D skeletons.
        self._accumulate(hands0)
        self._log_detections(hands0)
        if self.annotated_pub is not None:
            self._publish_annotated(frame0, hands0, stamp, w0, h0)
        if self.stereo_annotated_pub is not None:
            self._publish_annotated(
                frame1, hands1, stamp, w1, h1,
                pub=self.stereo_annotated_pub,
                camera_name=self.stereo_camera_name,
                frame_id=self.stereo_plane_frame_id)
        if self.markers_pub is not None:
            self._publish_markers(hands0, stamp, w0, h0)
        if self.landmarks_pub is not None:
            self._publish_landmarks(
                result0, hands0, result0.hand_world_landmarks or [],
                stamp, w0, h0)
        self._publish_markers3d(
            hands0, hands1, result0, result1, raw_frames,
            stamp, (w0, h0), (w1, h1))

    def _on_result(self, result, output_image, ts_ms):
        """LIVE_STREAM callback — runs on MediaPipe's thread."""
        del output_image
        with self._lock:
            entry = self._pending.pop(ts_ms, None)
            for key in [k for k in self._pending if k < ts_ms]:
                del self._pending[key]
        if entry is None:
            return
        stamp, frame, _ = entry
        self._handle_result(result, stamp, frame)

    # ---------------------------------------------------------------- output
    def _resolve_hands(self, result, w, h, tracker):
        """One camera's MediaPipe result -> the filtered/labelled hand list:
        [(label, score, pts_px, accepted, reason, idx, resolution)]."""
        hands = []
        raw = result.handedness if result.hand_landmarks else []
        world = result.hand_world_landmarks or []

        dets, pts_all = [], []
        for i, handed in enumerate(raw):
            pts = [
                (int(round(lm.x * w)), int(round(lm.y * h)))
                for lm in result.hand_landmarks[i]]
            pts_all.append(pts)
            # Wrist (landmark 0) position anchors tracking / side assignment.
            dets.append((float(pts[0][0]), float(pts[0][1]),
                         handed[0].category_name, float(handed[0].score)))

        if self.use_label_tracker:
            resolutions = tracker.update(dets, w, time.monotonic())
        else:
            resolutions = [
                Resolution(label=d[2], source="mediapipe", committed=True,
                           track_id=-1, mp_label=d[2], mp_score=d[3])
                for d in dets]

        for i, (det, res) in enumerate(zip(dets, resolutions)):
            label = res.label or det[2]
            score = det[3]
            reason = None
            if res.source == "duplicate":
                reason = "duplicate detection of one hand"
            elif label not in self.allowed_labels:
                reason = f"hand_filter_mode={self.hand_filter_mode}"
            elif score < self.min_handedness_confidence:
                reason = (f"score {score:.2f} < "
                          f"{self.min_handedness_confidence:.2f}")
            elif i >= len(world):
                reason = "no world landmarks"
            hands.append(
                (label, score, pts_all[i], reason is None, reason, i, res))
        return hands

    def _log_detections(self, hands):
        if not self.log_detections:
            return
        if hands:
            self._diag(
                "detections",
                "[detect] " + ", ".join(
                    f"{label}{'' if res.committed else '?'}"
                    f"[{res.source}]:{score:.2f}"
                    f"{' (mp:' + res.mp_label + ')' if res.corrected else ''}"
                    f"{'' if ok else ' REJECTED(' + reason + ')'}"
                    for label, score, _, ok, reason, _, res in hands))
        else:
            self._diag("detections_none", "[detect] no hands", "warn")

    def _handle_result(self, result, stamp, frame):
        h, w = frame.shape[:2]
        hands = self._resolve_hands(result, w, h, self.tracker)

        self._accumulate(hands)
        self._log_detections(hands)

        if self.annotated_pub is not None:
            self._publish_annotated(frame, hands, stamp, w, h)
        if self.markers_pub is not None:
            self._publish_markers(hands, stamp, w, h)
        if self.landmarks_pub is not None:
            self._publish_landmarks(
                result, hands, result.hand_world_landmarks or [], stamp, w, h)

    def _accumulate(self, hands):
        with self._lock:
            s = self._stats
            s["results"] += 1
            s["raw_hands"] += len(hands)
            if hands:
                s["frames_with_hand"] += 1
            for label, score, _, ok, _, _, res in hands:
                entry = s["per_label"].setdefault(label, [0, 0.0])
                entry[0] += 1
                entry[1] += score
                if not ok:
                    s["rejected"] += 1
                if res.corrected:
                    s["corrected"] += 1
                if not res.committed and res.source == "cues":
                    s["provisional"] += 1
                if res.source == "duplicate":
                    s["duplicates"] += 1
            # Label flips are counted on the highest-scoring hand in the frame:
            # that is the one the teleop pipeline would follow. With the label
            # tracker on this counts flips of the RESOLVED label, so it directly
            # measures how much flicker the tracker leaves.
            if hands:
                best = max(hands, key=lambda hd: hd[1])[0]
                if self._last_label is not None and best != self._last_label:
                    s["label_flips"] += 1
                self._last_label = best

    def _color_for(self, label, accepted):
        if not accepted:
            return self.reject_color
        return self.left_color if label == "Left" else self.right_color

    def _publish_annotated(self, frame, hands, stamp, w, h,
                           pub=None, camera_name=None, frame_id=None):
        pub = pub or self.annotated_pub
        camera_name = camera_name or self.camera_name
        frame_id = frame_id or self.plane_frame_id
        canvas = frame.copy()
        for label, score, pts, accepted, reason, _, res in hands:
            if not accepted and not self.draw_rejected:
                continue
            color = self._color_for(label, accepted)
            if self.draw_connections:
                for a, b in HAND_CONNECTIONS:
                    cv2.line(canvas, pts[a], pts[b], color,
                             self.line_thickness, cv2.LINE_AA)
            for idx, p in enumerate(pts):
                cv2.circle(canvas, p, self.point_radius, color, -1, cv2.LINE_AA)
                if self.draw_landmark_ids:
                    cv2.putText(canvas, str(idx), (p[0] + 5, p[1] - 5), FONT,
                                self.font_scale * 0.8, color, 1, cv2.LINE_AA)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if self.draw_bounding_box:
                cv2.rectangle(canvas, (min(xs) - 8, min(ys) - 8),
                              (max(xs) + 8, max(ys) + 8), color, 1, cv2.LINE_AA)
            if self.draw_handedness:
                # "Left? 0.62 [cues]" = provisional; "(mp:Right)" = the tracker
                # overrode MediaPipe's label.
                text = f"{label}{'' if res.committed else '?'} {score:.2f}"
                if self.use_label_tracker:
                    text += f" [{res.source}]"
                if res.corrected:
                    text += f" (mp:{res.mp_label})"
                if not accepted:
                    text += f"  REJECTED: {reason}"
                cv2.putText(canvas, text, (min(xs) - 8, max(0, min(ys) - 14)),
                            FONT, self.font_scale, color, 2, cv2.LINE_AA)
        if self.draw_header:
            accepted_n = sum(1 for hd in hands if hd[3])
            header = (f"{camera_name} {w}x{h} | hands {accepted_n}/{len(hands)} "
                      f"| filter {self.hand_filter_mode} "
                      f"| tracker {'on' if self.use_label_tracker else 'off'} "
                      f"| mirror_input {'on' if self.mirror_input else 'off'}")
            cv2.putText(canvas, header, (10, 24), FONT, self.font_scale,
                        (240, 240, 240), 2, cv2.LINE_AA)

        if self.annotated_image_scale != 1.0 and self.annotated_image_scale > 0.0:
            canvas = cv2.resize(
                canvas, None, fx=self.annotated_image_scale,
                fy=self.annotated_image_scale, interpolation=cv2.INTER_AREA)
        out_h, out_w = canvas.shape[:2]
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = out_h
        msg.width = out_w
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = out_w * 3
        msg.data = np.ascontiguousarray(canvas).tobytes()
        pub.publish(msg)

    def _to_plane(self, u, v, w, h):
        """Pixel -> image-plane metres: x right, y up, origin at image centre."""
        s = self.plane_width_m / max(w, 1)
        p = Point()
        p.x = (u - w * 0.5) * s
        p.y = -(v - h * 0.5) * s
        p.z = 0.0
        return p

    def _marker_base(self, stamp, ns, marker_id):
        m = Marker()
        m.header.frame_id = self.plane_frame_id
        m.header.stamp = stamp
        m.ns = ns
        m.id = marker_id
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.lifetime = Duration(seconds=self.marker_lifetime_sec).to_msg()
        return m

    def _publish_markers(self, hands, stamp, w, h):
        arr = MarkerArray()

        # Image border, so the 3D view has the same frame of reference as the
        # annotated image.
        border = self._marker_base(stamp, "image_border", 0)
        border.type = Marker.LINE_STRIP
        border.scale.x = self.marker_line_width * 0.5
        border.color.a = 0.7
        border.color.r = border.color.g = border.color.b = 0.6
        border.lifetime = Duration(seconds=0.0).to_msg()
        for u, v in ((0, 0), (w, 0), (w, h), (0, h), (0, 0)):
            border.points.append(self._to_plane(u, v, w, h))
        arr.markers.append(border)

        # Two slots per label so a hand that disappears is explicitly deleted
        # instead of lingering until its lifetime expires.
        drawn = set()
        for slot, (label, score, pts, accepted, reason, _, res) in enumerate(hands[:4]):
            if not accepted and not self.draw_rejected:
                continue
            drawn.add(slot)
            bgr = self._color_for(label, accepted)
            r, g, b = bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0

            joints = self._marker_base(stamp, "joints", slot)
            joints.type = Marker.SPHERE_LIST
            joints.scale.x = joints.scale.y = joints.scale.z = self.marker_point_size
            joints.color.a = 1.0
            joints.color.r, joints.color.g, joints.color.b = r, g, b
            for u, v in pts:
                joints.points.append(self._to_plane(u, v, w, h))
            arr.markers.append(joints)

            bones = self._marker_base(stamp, "bones", slot)
            bones.type = Marker.LINE_LIST
            bones.scale.x = self.marker_line_width
            bones.color.a = 1.0
            bones.color.r, bones.color.g, bones.color.b = r, g, b
            for a, bb in HAND_CONNECTIONS:
                bones.points.append(self._to_plane(*pts[a], w, h))
                bones.points.append(self._to_plane(*pts[bb], w, h))
            arr.markers.append(bones)

            text = self._marker_base(stamp, "label", slot)
            text.type = Marker.TEXT_VIEW_FACING
            text.scale.z = self.marker_text_size
            text.color.a = 1.0
            text.color.r, text.color.g, text.color.b = r, g, b
            text.text = (f"{label}{'' if res.committed else '?'} {score:.2f}"
                         + (f" (mp:{res.mp_label})" if res.corrected else "")
                         + ("" if accepted else f" REJECTED ({reason})"))
            wrist = self._to_plane(pts[0][0], pts[0][1], w, h)
            wrist.z += self.marker_text_size
            text.pose.position = wrist
            arr.markers.append(text)

        for slot in range(4):
            if slot in drawn:
                continue
            for ns in ("joints", "bones", "label"):
                gone = self._marker_base(stamp, ns, slot)
                gone.action = Marker.DELETE
                arr.markers.append(gone)

        self.markers_pub.publish(arr)

    def _skeleton_markers(self, stamp, xyz, suffix, slot, bgr, alpha, text=None,
                          frame_id=None):
        """Joint spheres + bone lines (+ a text label) for one 3D hand, in the
        world3d frame unless frame_id overrides. suffix distinguishes marker
        namespaces (fit vs raw vs world-model shelf)."""
        frame_id = frame_id or self.world3d_frame_id
        r, g, b = bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0
        out = []

        joints = self._marker_base(stamp, "joints3d" + suffix, slot)
        joints.header.frame_id = frame_id
        joints.type = Marker.SPHERE_LIST
        joints.scale.x = joints.scale.y = joints.scale.z = self.marker_point_size
        joints.color.a = alpha
        joints.color.r, joints.color.g, joints.color.b = r, g, b
        joints.points = [
            Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in xyz]
        out.append(joints)

        bones = self._marker_base(stamp, "bones3d" + suffix, slot)
        bones.header.frame_id = frame_id
        bones.type = Marker.LINE_LIST
        bones.scale.x = self.marker_line_width
        bones.color.a = alpha
        bones.color.r, bones.color.g, bones.color.b = r, g, b
        for a, bb in HAND_CONNECTIONS:
            bones.points.append(Point(
                x=float(xyz[a][0]), y=float(xyz[a][1]), z=float(xyz[a][2])))
            bones.points.append(Point(
                x=float(xyz[bb][0]), y=float(xyz[bb][1]), z=float(xyz[bb][2])))
        out.append(bones)

        if text is not None:
            lbl = self._marker_base(stamp, "label3d" + suffix, slot)
            lbl.header.frame_id = frame_id
            lbl.type = Marker.TEXT_VIEW_FACING
            lbl.scale.z = self.marker_text_size
            lbl.color.a = alpha
            lbl.color.r, lbl.color.g, lbl.color.b = r, g, b
            lbl.text = text
            # y is DOWN in the optical frame, so "above the wrist" is -y.
            lbl.pose.position = Point(
                x=float(xyz[0][0]),
                y=float(xyz[0][1]) - self.marker_text_size,
                z=float(xyz[0][2]))
            out.append(lbl)
        return out

    def _publish_markers3d(self, hands0, hands1, result0, result1, frames,
                           stamp, size0, size1):
        """3D skeletons in the extrinsics world frame (metres): the refined
        model fit when enable_refine (two views, or one view + ray-filtered
        depth), plus the raw per-joint triangulation in grey for comparison.
        With enable_refine false: the raw triangulation only, needing both
        views, exactly the old behaviour. `frames` = the unmirrored captures,
        for the virtual pinhole crop pass."""
        arr = MarkerArray()
        accepted = (self._accepted_by_label(hands0),
                    self._accepted_by_label(hands1))
        results = (result0, result1)
        sizes = (size0, size1)
        cam_names = (self.camera_name, self.stereo_camera_name)
        now = time.monotonic()
        drawn_fit, drawn_tri = set(), set()
        tiles = []
        wm = []  # (cam, label, "full"/"crop", world 21x3) for the debug shelf

        for slot, label in enumerate(("Left", "Right")):
            obs = []      # (cam, pts_px, size) per camera that sees this hand
            model = None  # scaled world-landmark model, primary cam preferred
            model_src = "full"
            for cam in (0, 1):
                hit = accepted[cam].get(label)
                if hit is None:
                    continue
                pts, idx = hit
                obs.append((cam, pts, sizes[cam]))
                world = results[cam].hand_world_landmarks
                if idx < len(world):
                    w_full = np.array(
                        [[lm.x, lm.y, lm.z] for lm in world[idx]])
                    if self.publish_world_models:
                        wm.append((cam, label, "full", w_full))
                    if model is None:
                        model = self.hand_size_scaling_factor * w_full
            if not obs:
                continue

            # Second pass: replace the (fisheye-skewed) full-frame world
            # landmarks with the ones from a virtual pinhole crop. The fit
            # uses the first camera's crop; with the world-models shelf on,
            # the crop pass runs on BOTH cameras so every slot can fill.
            if (self.enable_refine and self.enable_pinhole_crops
                    and self.detector_crop is not None):
                crop_obs = obs if self.publish_world_models else obs[:1]
                for k, (cam, pts, sz) in enumerate(crop_obs):
                    crop_world, tile = self._crop_world_landmarks(
                        frames[cam], cam, pts, sz)
                    with self._lock:
                        self._stats["crop_hits" if crop_world is not None
                                    else "crop_misses"] += 1
                    if crop_world is not None:
                        wm.append((cam, label, "crop", crop_world))
                        if k == 0:
                            model = self.hand_size_scaling_factor * crop_world
                            model_src = "crop"
                    cv2.putText(tile, f"{label} {cam_names[cam]}", (6, 16),
                                FONT, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
                    tiles.append(tile)

            xyz_tri = None
            if len(obs) == 2:
                xyz_tri = self._triangulate(
                    obs[0][1], obs[1][1], size0, size1)
                depth_tri = float(np.median(xyz_tri[:, 2]))
                if not 0.05 < depth_tri < 5.0:
                    self._diag(
                        f"stereo_depth_{label}",
                        f"stereo {label}: implausible triangulated depth "
                        f"{depth_tri:.2f} m (bad extrinsics, or the two "
                        "cameras matched different physical hands)", "warn")
                    xyz_tri = None

            xyz_fit = None
            if self.enable_refine and model is not None:
                xyz_fit, info = self._fit_hand_pose(
                    label, obs, model, xyz_tri, now)
                if xyz_fit is None:
                    self._diag(f"fit_fail_{label}",
                               f"refine {label}: {info}", "warn")

            bgr = self.left_color if label == "Left" else self.right_color
            if xyz_fit is not None:
                drawn_fit.add(slot)
                with self._lock:
                    self._stats["stereo_pairs" if info["views"] >= 2
                                else "mono_fits"] += 1
                mode = ("stereo" if info["views"] >= 2
                        else f"mono:{cam_names[info['cam']]}")
                if self.enable_pinhole_crops:
                    mode += f"|{model_src}"
                arr.markers += self._skeleton_markers(
                    stamp, xyz_fit, "", slot, bgr, 1.0,
                    f"{label} {info['depth']:.2f}m [{mode}] "
                    f"rms {info['rms']:.1f}px")
                if xyz_tri is not None and self.draw_raw_triangulation:
                    drawn_tri.add(slot)
                    arr.markers += self._skeleton_markers(
                        stamp, xyz_tri, "_tri", slot, (160, 160, 160), 0.5)
            elif xyz_tri is not None:
                # Refinement off (or this fit rejected): raw triangulation in
                # the main slots, as before.
                drawn_fit.add(slot)
                with self._lock:
                    self._stats["stereo_pairs"] += 1
                depth_tri = float(np.median(xyz_tri[:, 2]))
                arr.markers += self._skeleton_markers(
                    stamp, xyz_tri, "", slot, bgr, 1.0,
                    f"{label} {depth_tri:.2f}m [tri]")

        for slot in range(2):
            if slot not in drawn_fit:
                for ns in ("joints3d", "bones3d", "label3d"):
                    gone = self._marker_base(stamp, ns, slot)
                    gone.header.frame_id = self.world3d_frame_id
                    gone.action = Marker.DELETE
                    arr.markers.append(gone)
            if slot not in drawn_tri:
                for ns in ("joints3d_tri", "bones3d_tri"):
                    gone = self._marker_base(stamp, ns, slot)
                    gone.header.frame_id = self.world3d_frame_id
                    gone.action = Marker.DELETE
                    arr.markers.append(gone)

        self.markers3d_pub.publish(arr)
        if tiles and self.publish_crops and self.crops_pub is not None:
            self._publish_crops(tiles, stamp)
        if self.publish_world_models and self.world_models_pub is not None:
            self._publish_world_models(wm, stamp)

    def _publish_world_models(self, entries, stamp):
        """Debug shelf: every raw hand_world_landmarks set this tick — up to 8
        (2 cameras x 2 hands x 2 passes) — as small skeletons on a fixed grid
        in the world_models frame. Columns: Left full|crop then Right
        full|crop; rows: camera0 (top) / camera1. Handedness is the usual
        left/right colour; the caption names the source (cam0_full, cam1_crop,
        ...). Raw MediaPipe metres, NOT hand_size_scaling_factor-scaled."""
        arr = MarkerArray()
        have = {(cam, label, src): w for cam, label, src, w in entries}
        for slot, key in enumerate(WM_GRID):
            world = have.get(key)
            if world is None:
                for ns in ("joints3d_wm", "bones3d_wm", "label3d_wm"):
                    gone = self._marker_base(stamp, ns, slot)
                    gone.header.frame_id = self.world_models_frame_id
                    gone.action = Marker.DELETE
                    arr.markers.append(gone)
                continue
            cam, label, src = key
            col = (("Left", "Right").index(label) * 2
                   + ("full", "crop").index(src))
            offset = np.array([
                0.30 * col + (0.12 if label == "Right" else 0.0),
                0.34 * cam, 0.0])
            xyz = world - world.mean(axis=0) + offset
            bgr = self.left_color if label == "Left" else self.right_color
            arr.markers += self._skeleton_markers(
                stamp, xyz, "_wm", slot, bgr, 1.0,
                text=f"cam{cam}_{src}",
                frame_id=self.world_models_frame_id)
        self.world_models_pub.publish(arr)

    def _publish_landmarks(self, result, hands, world, stamp, w, h):
        """Same contract as hand_landmarks_node: only accepted hands."""
        msg = HandLandmarks()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_name
        msg.source_topic = self.annotated_topic
        for label, score, pts, accepted, _, idx, _res in hands:
            if not accepted:
                continue
            hand = Hand()
            hand.handedness = label
            hand.score = score
            hand.landmarks_image = [
                Point(x=float(lm.x * w), y=float(lm.y * h), z=float(lm.z))
                for lm in result.hand_landmarks[idx]]
            hand.landmarks_world = [
                Point(x=float(lm.x), y=float(lm.y), z=float(lm.z))
                for lm in world[idx]]
            msg.hands.append(hand)
        self.landmarks_pub.publish(msg)

    def shutdown(self):
        # Close the detectors first: close() joins MediaPipe's worker, so no
        # result callback can fire into a half-torn-down node.
        self.detector.close()
        if self.detector_stereo is not None:
            self.detector_stereo.close()
        if self.detector_crop is not None:
            self.detector_crop.close()
        if self.cap is not None:
            self.cap.release()
        if self.cap_stereo is not None:
            self.cap_stereo.release()


def main(args=None):
    rclpy.init(args=args)
    node = MediapipeDetectionDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
