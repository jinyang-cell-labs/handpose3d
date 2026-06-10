#!/usr/bin/env python3

"""
Stereo depth estimation node.

Subscribes to the same two calibrated camera streams as ``handpose_node``:

    <name>/image_raw    sensor_msgs/Image      (bgr8)
    <name>/camera_info  sensor_msgs/CameraInfo

Rectifies both views with the stereo calibration carried in camera_info
(K/D/R/P), runs semi-global block matching (cv2.StereoSGBM) on the rectified
pair, and converts the disparity to metric depth (Z = f * B / d, with focal
length f and baseline B taken from the rectified projection matrices).

Both horizontal (Tx = P[0,3] != 0) and vertical (Ty = P[1,3] != 0) stereo
pairs are supported: SGBM only matches along image rows, so for a vertical
pair the rectified images are rotated 90deg CCW (epipolar lines become rows,
top camera plays "left"), matched, and the depth map is rotated back.

Published topics (all in the left rectified camera frame):

    stereo/depth        sensor_msgs/Image  32FC1, metres, NaN where invalid
    stereo/depth_color  sensor_msgs/Image  bgr8 colorized depth (near=red)
    stereo/camera_info  sensor_msgs/CameraInfo  rectified left intrinsics
    stereo/image_rect   sensor_msgs/Image  bgr8 rectified left view (optional)

View in RViz with an Image display on ``stereo/depth_color`` (or the raw
``stereo/depth``), or as a point cloud with the DepthCloud display fed by
``stereo/depth`` + ``stereo/camera_info``.
"""

import cv2
import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

# ===== HOTFIX(rotate-90): TEMPORARY =========================================
# The current rosbag publishes a 90deg-rotated image (hardware limitation).
# We rotate it upright before rectification so the frame lines up with the
# camera_info calibration (landscape 640x480 -> portrait 480x640).
# Keep in sync with handpose_node.py; set to None to disable, or DELETE this
# constant + the fenced block in _on_images, once the upstream publishes
# upright images.
#   options: cv2.ROTATE_90_CLOCKWISE / cv2.ROTATE_90_COUNTERCLOCKWISE / cv2.ROTATE_180
_HOTFIX_ROTATE = cv2.ROTATE_90_CLOCKWISE
# ============================================================================


class StereoDepthNode(Node):
    def __init__(self):
        super().__init__("stereo_depth_node")

        # --- parameters -----------------------------------------------------
        self.declare_parameter("camera_names", ["camera0", "camera1"])
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("sync_queue_size", 10)
        # StereoSGBM tuning. num_disparities must be a multiple of 16,
        # block_size odd. Larger num_disparities sees closer objects but
        # costs CPU and crops the left edge of the depth map.
        self.declare_parameter("min_disparity", 0)
        self.declare_parameter("num_disparities", 128)
        self.declare_parameter("block_size", 5)
        self.declare_parameter("uniqueness_ratio", 10)
        self.declare_parameter("speckle_window_size", 100)
        self.declare_parameter("speckle_range", 2)
        self.declare_parameter("disp12_max_diff", 1)
        # Depth validity clamps (metres); outside -> NaN. Also the color
        # mapping range for stereo/depth_color.
        self.declare_parameter("min_depth", 0.1)
        self.declare_parameter("max_depth", 5.0)
        self.declare_parameter("publish_depth_color", True)
        self.declare_parameter("publish_rectified", True)

        self.camera_names = list(self.get_parameter("camera_names").value)
        if len(self.camera_names) != 2:
            raise ValueError("stereo_depth_node requires exactly 2 camera_names")
        self.min_disparity = int(self.get_parameter("min_disparity").value)
        self.num_disparities = int(self.get_parameter("num_disparities").value)
        self.block_size = int(self.get_parameter("block_size").value)
        self.min_depth = float(self.get_parameter("min_depth").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.publish_depth_color = bool(
            self.get_parameter("publish_depth_color").value
        )
        self.publish_rectified = bool(self.get_parameter("publish_rectified").value)

        self.matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8 * self.block_size**2,
            P2=32 * self.block_size**2,
            disp12MaxDiff=int(self.get_parameter("disp12_max_diff").value),
            preFilterCap=31,
            uniquenessRatio=int(self.get_parameter("uniqueness_ratio").value),
            speckleWindowSize=int(self.get_parameter("speckle_window_size").value),
            speckleRange=int(self.get_parameter("speckle_range").value),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        # --- calibration state ---------------------------------------------
        # Per-camera K/D/R/P captured once from camera_info (same pattern as
        # handpose_node). Rectification maps are built lazily from the first
        # image pair so the (possibly hotfix-rotated) frame size is known.
        self.calib = {name: None for name in self.camera_names}
        self.ready = False
        self.maps = None  # {name: (map1, map2)} once built
        self.left = None  # camera playing SGBM "left" (left / top of the pair)
        self.right = None
        self.vertical = False  # baseline along image y (top/bottom pair)
        self.focal = None  # rectified focal length f (px)
        self.baseline = None  # stereo baseline B (m)

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

        self.depth_pub = self.create_publisher(
            Image, "stereo/depth", qos_profile_sensor_data
        )
        self.info_pub = self.create_publisher(
            CameraInfo, "stereo/camera_info", qos_profile_sensor_data
        )
        self.color_pub = None
        if self.publish_depth_color:
            self.color_pub = self.create_publisher(
                Image, "stereo/depth_color", qos_profile_sensor_data
            )
        self.rect_pub = None
        if self.publish_rectified:
            self.rect_pub = self.create_publisher(
                Image, "stereo/image_rect", qos_profile_sensor_data
            )

        self.get_logger().info(
            f"stereo_depth_node ready: cameras={self.camera_names}, "
            "waiting for camera_info + images..."
        )

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
            "width": int(msg.width),
            "height": int(msg.height),
        }
        self.get_logger().info(f"Captured calibration for {name}")
        if all(self.calib[n] is not None for n in self.camera_names):
            self._on_calibration_ready()

    def _on_calibration_ready(self):
        """Both calibrations are in — derive pair geometry, f and B.

        Rectified projection matrices: the reference camera has Tx = Ty = 0;
        the second carries the baseline as Tx = P[0,3] = -fx * B (horizontal
        pair, second camera on the right) or Ty = P[1,3] = -fy * B (vertical
        pair, second camera at the bottom). A positive offset means the
        cameras are physically swapped relative to the calibration, so the
        roles flip. SGBM's "left" role is the left (horizontal) / top
        (vertical) camera — the one whose features sit at the larger
        coordinate along the baseline axis.
        """
        n0, n1 = self.camera_names
        P0 = self.calib[n0]["p"]
        P1 = self.calib[n1]["p"]
        tx = (P0[0, 3] / P0[0, 0], P1[0, 3] / P1[0, 0])
        ty = (P0[1, 3] / P0[1, 1], P1[1, 3] / P1[1, 1])

        if max(abs(t) for t in tx) > 1e-9:
            self.vertical, t = False, tx
        elif max(abs(t) for t in ty) > 1e-9:
            self.vertical, t = True, ty
        else:
            self.get_logger().error(
                "camera_info P carries no baseline (P[0,3]=P[1,3]=0 on both "
                "cameras): the streams are not jointly stereo-calibrated. "
                "Depth will not be published."
            )
            return

        i_other = 0 if abs(t[0]) >= abs(t[1]) else 1
        ref, other = self.camera_names[1 - i_other], self.camera_names[i_other]
        if t[i_other] < 0:  # second camera on the positive side (right/bottom)
            self.left, self.right = ref, other
        else:
            self.left, self.right = other, ref
        self.baseline = abs(t[i_other])
        P_left = self.calib[self.left]["p"]
        self.focal = P_left[1, 1] if self.vertical else P_left[0, 0]

        self.ready = True
        roles = ("top", "bottom") if self.vertical else ("left", "right")
        self.get_logger().info(
            f"Stereo geometry: {'VERTICAL' if self.vertical else 'HORIZONTAL'} "
            f"pair, {roles[0]}={self.left}, {roles[1]}={self.right}, "
            f"f={self.focal:.1f} px, baseline={self.baseline:.4f} m"
        )

    def _ensure_rectify_maps(self, shape):
        """Build the undistort+rectify maps once, from the first frame size."""
        if self.maps is not None:
            return
        h, w = shape[:2]
        self.maps = {}
        for name in self.camera_names:
            c = self.calib[name]
            if (c["width"], c["height"]) not in ((w, h), (0, 0)):
                self.get_logger().warn(
                    f"{name}: image size {w}x{h} does not match camera_info "
                    f"{c['width']}x{c['height']} — rectification may be wrong."
                )
            if c["model"] == "fisheye":
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    c["k"], c["d"][:4].reshape(1, 4), c["r"], c["p"][:, :3],
                    (w, h), cv2.CV_16SC2,
                )
            else:  # plumb_bob / rational_polynomial
                map1, map2 = cv2.initUndistortRectifyMap(
                    c["k"], c["d"], c["r"], c["p"][:, :3], (w, h), cv2.CV_16SC2
                )
            self.maps[name] = (map1, map2)
        self.get_logger().info(f"Built rectification maps for {w}x{h} frames")

    def _on_images(self, *msgs):
        if not self.ready:
            self.get_logger().warn(
                "Waiting for stereo calibration on all cameras...",
                throttle_duration_sec=5.0,
            )
            return

        frames = {}
        for name, msg in zip(self.camera_names, msgs):
            frame_bgr = self._decode_to_bgr(msg)

            # ===== HOTFIX(rotate-90): TEMPORARY =============================
            # Un-rotate the rosbag image (see _HOTFIX_ROTATE at top of file).
            # DELETE this block once the upstream publishes upright images.
            if _HOTFIX_ROTATE is not None:
                frame_bgr = cv2.rotate(frame_bgr, _HOTFIX_ROTATE)
            # ===== END HOTFIX ===============================================

            frames[name] = frame_bgr

        self._ensure_rectify_maps(frames[self.left].shape)

        rect = {}
        for name in (self.left, self.right):
            map1, map2 = self.maps[name]
            rect[name] = cv2.remap(frames[name], map1, map2, cv2.INTER_LINEAR)

        gray_l = cv2.cvtColor(rect[self.left], cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect[self.right], cv2.COLOR_BGR2GRAY)

        # Vertical pair: rotate 90deg CCW so the epipolar (v) axis becomes
        # the row axis with the top camera's features at larger x — exactly
        # SGBM's "left" convention. The depth map is rotated back after.
        if self.vertical:
            gray_l = cv2.rotate(gray_l, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gray_r = cv2.rotate(gray_r, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # SGBM returns fixed-point disparity (x16); invalid pixels get
        # minDisparity - 1.
        disp = self.matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
        valid = disp > max(float(self.min_disparity), 0.0) + 1e-3

        depth = np.full(disp.shape, np.nan, dtype=np.float32)
        depth[valid] = self.focal * self.baseline / disp[valid]
        depth[(depth < self.min_depth) | (depth > self.max_depth)] = np.nan

        if self.vertical:  # back to the rectified (portrait) camera frame
            depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)

        stamp = msgs[self.camera_names.index(self.left)].header.stamp
        self._publish_depth(depth, stamp)
        if self.color_pub is not None:
            self._publish_depth_color(depth, stamp)
        if self.rect_pub is not None:
            self._publish_image(self.rect_pub, rect[self.left], "bgr8", stamp)

        n_valid = int(np.isfinite(depth).sum())
        self.get_logger().info(
            f"depth: {n_valid}/{depth.size} valid px "
            f"({100.0 * n_valid / depth.size:.0f}%)",
            throttle_duration_sec=5.0,
        )

    # ------------------------------------------------------------- publishing
    def _publish_depth(self, depth, stamp):
        h, w = depth.shape
        img = Image()
        img.header.stamp = stamp
        img.header.frame_id = self.left
        img.height = h
        img.width = w
        img.encoding = "32FC1"
        img.is_bigendian = 0
        img.step = w * 4
        img.data = np.ascontiguousarray(depth).tobytes()
        self.depth_pub.publish(img)

        # Rectified-left intrinsics so DepthCloud / depth_image_proc can
        # back-project the depth image.
        P = self.calib[self.left]["p"]
        info = CameraInfo()
        info.header = img.header
        info.height = h
        info.width = w
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = P[:, :3].ravel().tolist()
        info.r = np.eye(3).ravel().tolist()
        info.p = P.ravel().tolist()
        self.info_pub.publish(info)

    def _publish_depth_color(self, depth, stamp):
        """Colorize depth for easy viewing in an RViz Image display.

        Near = red, far = blue (inverted JET over [min_depth, max_depth]);
        invalid pixels are black.
        """
        finite = np.isfinite(depth)
        norm = np.zeros(depth.shape, dtype=np.uint8)
        span = max(self.max_depth - self.min_depth, 1e-6)
        scaled = (np.clip(depth, self.min_depth, self.max_depth) - self.min_depth) / span
        norm[finite] = (255 * (1.0 - scaled[finite])).astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        color[~finite] = 0
        self._publish_image(self.color_pub, color, "bgr8", stamp)

    def _publish_image(self, pub, frame, encoding, stamp):
        h, w = frame.shape[:2]
        img = Image()
        img.header.stamp = stamp
        img.header.frame_id = self.left
        img.height = h
        img.width = w
        img.encoding = encoding
        img.is_bigendian = 0
        img.step = w * (frame.shape[2] if frame.ndim == 3 else 1)
        img.data = np.ascontiguousarray(frame).tobytes()
        pub.publish(img)

    def _decode_to_bgr(self, msg):
        """Decode a sensor_msgs/Image to a contiguous bgr8 ndarray.

        Honors msg.encoding (rgb8/bgr8/rgba8/bgra8/mono8) and msg.step (row
        stride / padding). Mirrors handpose_node._decode_to_bgr.
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


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
