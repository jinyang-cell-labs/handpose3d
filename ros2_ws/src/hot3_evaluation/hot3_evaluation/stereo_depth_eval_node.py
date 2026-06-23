#!/usr/bin/env python3

"""
HOT3D fisheye stereo depth evaluation node.

Takes the two HOT3D SLAM fisheye streams (``1201-1`` left, ``1201-2`` right) as
published by ``hot3_dataset_interface`` and produces a depth map, so you can
*see how good* stereo depth from this camera pair actually is:

    <left>/image_raw    sensor_msgs/Image       (mono8)
    <left>/camera_info  sensor_msgs/CameraInfo  (FISHEYE624)
    <right>/image_raw   sensor_msgs/Image       (mono8)
    <right>/camera_info sensor_msgs/CameraInfo  (FISHEYE624)
    /tf                 world -> <camera frame>  (per-frame extrinsics)

Pipeline:

1. The two cameras are NOT a rectified rig -- they are Aria SLAM tracking
   cameras ~75 deg apart, so camera_info carries no stereo baseline. The
   relative pose is recovered from /tf (the transform between the two camera
   frames, which is static) and fed to ``cv2.fisheye.stereoRectify``.
2. FISHEYE624 (Meta's 6-radial + 2-tangential + 4-thin-prism model) shares the
   same *equidistant* radial basis as OpenCV's fisheye model, so the first four
   FISHEYE624 radial coefficients are used directly as cv2.fisheye k1..k4. The
   small higher-order / tangential / thin-prism terms are dropped.
3. Both views are undistorted + rectified to a common virtual pinhole pair,
   matched with StereoSGBM, and disparity is converted to metric depth
   (Z = f_rect * B / disp).

Published (all in the rectified-left camera frame):

    stereo/depth        sensor_msgs/Image       32FC1 metres, NaN where invalid
    stereo/depth_color  sensor_msgs/Image       bgr8 colorized (near=red)
    stereo/camera_info  sensor_msgs/CameraInfo  rectified-left intrinsics
    stereo/image_rect   sensor_msgs/Image       mono8 rectified-left view

NOTE: the ~75 deg divergence means the overlapping FOV after rectification is
small and the rectified focal is low, so expect sparse, noisy depth -- that is
the property being evaluated, not a bug.
"""

import cv2
import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class StereoDepthEvalNode(Node):
    def __init__(self):
        super().__init__("stereo_depth_eval_node")

        # --- parameters -----------------------------------------------------
        # left first, right second (the FISHEYE624 1201-1 / 1201-2 pair).
        self.declare_parameter("camera_names", ["camera0", "camera1"])
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("sync_queue_size", 10)
        # Fisheye rectification controls. balance in [0,1] trades off how much
        # of the source FOV is kept (0 = crop to valid, 1 = keep all); fov_scale
        # > 1 zooms out. With this divergent pair, balance=0 gives the densest
        # overlap.
        self.declare_parameter("balance", 0.0)
        self.declare_parameter("fov_scale", 1.0)
        # StereoSGBM tuning (num_disparities multiple of 16, block_size odd).
        self.declare_parameter("min_disparity", 0)
        self.declare_parameter("num_disparities", 128)
        self.declare_parameter("block_size", 5)
        self.declare_parameter("uniqueness_ratio", 10)
        self.declare_parameter("speckle_window_size", 100)
        self.declare_parameter("speckle_range", 2)
        self.declare_parameter("disp12_max_diff", 1)
        # Depth validity clamps (metres) + color mapping range.
        self.declare_parameter("min_depth", 0.1)
        self.declare_parameter("max_depth", 5.0)
        self.declare_parameter("publish_depth_color", True)
        self.declare_parameter("publish_rectified", True)

        self.camera_names = list(self.get_parameter("camera_names").value)
        if len(self.camera_names) != 2:
            raise ValueError("stereo_depth_eval_node requires exactly 2 camera_names")
        self.balance = float(self.get_parameter("balance").value)
        self.fov_scale = float(self.get_parameter("fov_scale").value)
        self.min_disparity = int(self.get_parameter("min_disparity").value)
        self.min_depth = float(self.get_parameter("min_depth").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.publish_depth_color = bool(self.get_parameter("publish_depth_color").value)
        self.publish_rectified = bool(self.get_parameter("publish_rectified").value)

        self.matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=int(self.get_parameter("num_disparities").value),
            blockSize=int(self.get_parameter("block_size").value),
            P1=8 * int(self.get_parameter("block_size").value) ** 2,
            P2=32 * int(self.get_parameter("block_size").value) ** 2,
            disp12MaxDiff=int(self.get_parameter("disp12_max_diff").value),
            preFilterCap=31,
            uniquenessRatio=int(self.get_parameter("uniqueness_ratio").value),
            speckleWindowSize=int(self.get_parameter("speckle_window_size").value),
            speckleRange=int(self.get_parameter("speckle_range").value),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        # --- state ----------------------------------------------------------
        # Per-camera intrinsics captured once from camera_info; rectification is
        # built lazily once both calibrations AND the inter-camera tf are known.
        self.calib = {name: None for name in self.camera_names}
        self.ready = False
        self.maps = None  # {name: (map1, map2)}
        self.focal = None  # rectified focal f (px)
        self.baseline = None  # stereo baseline B (m)
        self.P_left = None  # rectified-left projection matrix (for camera_info)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- subscriptions & publishers ------------------------------------
        self.info_subs = [
            self.create_subscription(
                CameraInfo,
                f"{name}/camera_info",
                lambda msg, n=name: self._on_camera_info(msg, n),
                qos_profile_sensor_data,
            )
            for name in self.camera_names
        ]
        image_subs = [
            Subscriber(
                self, Image, f"{name}/image_raw", qos_profile=qos_profile_sensor_data
            )
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
        self.color_pub = (
            self.create_publisher(Image, "stereo/depth_color", qos_profile_sensor_data)
            if self.publish_depth_color
            else None
        )
        self.rect_pub = (
            self.create_publisher(Image, "stereo/image_rect", qos_profile_sensor_data)
            if self.publish_rectified
            else None
        )

        self.get_logger().info(
            f"stereo_depth_eval_node ready: left={self.camera_names[0]}, "
            f"right={self.camera_names[1]}; waiting for camera_info + tf..."
        )

    # --------------------------------------------------------------- callbacks
    def _on_camera_info(self, msg, name):
        if self.calib[name] is not None:
            return  # static; capture once
        params = np.array(msg.d, dtype=np.float64).ravel()
        f = msg.k[0]
        cx, cy = msg.k[2], msg.k[5]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        # FISHEYE624 D = [6 radial, 2 tangential, 4 thin-prism]; cv2.fisheye uses
        # the first four radial coeffs (same equidistant basis).
        d4 = np.zeros((4, 1), dtype=np.float64)
        d4[: min(4, params.size), 0] = params[:4]
        self.calib[name] = {
            "k": K,
            "d": d4,
            "size": (int(msg.width), int(msg.height)),
            "frame": msg.header.frame_id or name,
        }
        self.get_logger().info(f"Captured calibration for {name}")

    def _try_build_rectification(self):
        """Once both calibrations and the inter-camera tf are available, build
        the fisheye stereo rectification maps."""
        if self.ready:
            return True
        if any(self.calib[n] is None for n in self.camera_names):
            return False

        left, right = self.camera_names
        cl, cr = self.calib[left], self.calib[right]
        # Relative pose: p_right = R @ p_left + t  ==  tf right<-left.
        try:
            tf = self.tf_buffer.lookup_transform(
                cr["frame"], cl["frame"], rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            self.get_logger().warn(
                f"Waiting for tf {cr['frame']} <- {cl['frame']} ...",
                throttle_duration_sec=5.0,
            )
            return False

        R = self._quat_to_R(tf.transform.rotation)
        t = np.array(
            [
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ],
            dtype=np.float64,
        ).reshape(3, 1)

        size = cl["size"]
        R1, R2, P1, P2, _ = cv2.fisheye.stereoRectify(
            cl["k"], cl["d"], cr["k"], cr["d"], size, R, t,
            cv2.CALIB_ZERO_DISPARITY,
            newImageSize=size, balance=self.balance, fov_scale=self.fov_scale,
        )
        m1 = cv2.fisheye.initUndistortRectifyMap(
            cl["k"], cl["d"], R1, P1[:3, :3], size, cv2.CV_16SC2
        )
        m2 = cv2.fisheye.initUndistortRectifyMap(
            cr["k"], cr["d"], R2, P2[:3, :3], size, cv2.CV_16SC2
        )
        self.maps = {left: m1, right: m2}
        self.focal = float(P1[0, 0])
        self.baseline = float(-P2[0, 3] / P2[0, 0])
        self.P_left = P1
        self.ready = True
        self.get_logger().info(
            f"Built fisheye rectification: baseline={self.baseline:.4f} m, "
            f"f_rect={self.focal:.1f} px (overlap FOV is small for this "
            "divergent pair -- expect sparse depth)."
        )
        return True

    def _on_images(self, *msgs):
        if not self._try_build_rectification():
            return

        left, right = self.camera_names
        frames = {n: self._decode_to_gray(m) for n, m in zip(self.camera_names, msgs)}
        m1x, m1y = self.maps[left]
        m2x, m2y = self.maps[right]
        rect_l = cv2.remap(frames[left], m1x, m1y, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frames[right], m2x, m2y, cv2.INTER_LINEAR)

        disp = self.matcher.compute(rect_l, rect_r).astype(np.float32) / 16.0
        valid = disp > max(float(self.min_disparity), 0.0) + 1e-3
        depth = np.full(disp.shape, np.nan, dtype=np.float32)
        depth[valid] = self.focal * self.baseline / disp[valid]
        depth[(depth < self.min_depth) | (depth > self.max_depth)] = np.nan

        stamp = msgs[0].header.stamp
        self._publish_depth(depth, stamp)
        if self.color_pub is not None:
            self._publish_depth_color(depth, stamp)
        if self.rect_pub is not None:
            self._publish_image(self.rect_pub, rect_l, "mono8", stamp)

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
        img.header.frame_id = self.camera_names[0]
        img.height, img.width = h, w
        img.encoding = "32FC1"
        img.is_bigendian = 0
        img.step = w * 4
        img.data = np.ascontiguousarray(depth).tobytes()
        self.depth_pub.publish(img)

        info = CameraInfo()
        info.header = img.header
        info.height, info.width = h, w
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = self.P_left[:, :3].ravel().tolist()
        info.r = np.eye(3).ravel().tolist()
        info.p = self.P_left.ravel().tolist()
        self.info_pub.publish(info)

    def _publish_depth_color(self, depth, stamp):
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
        img.header.frame_id = self.camera_names[0]
        img.height, img.width = h, w
        img.encoding = encoding
        img.is_bigendian = 0
        img.step = w * (frame.shape[2] if frame.ndim == 3 else 1)
        img.data = np.ascontiguousarray(frame).tobytes()
        pub.publish(img)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _quat_to_R(q):
        w, x, y, z = q.w, q.x, q.y, q.z
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _decode_to_gray(msg):
        """Decode a sensor_msgs/Image to a contiguous mono8 ndarray."""
        enc = (msg.encoding or "mono8").lower()
        channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8uc1": 1}.get(enc, 1)
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        step = msg.step if msg.step else msg.width * channels
        arr = buf[: step * msg.height].reshape(msg.height, step)
        arr = arr[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
        if channels == 1:
            return np.ascontiguousarray(arr[:, :, 0])
        code = {"rgb8": cv2.COLOR_RGB2GRAY, "rgba8": cv2.COLOR_RGBA2GRAY,
                "bgra8": cv2.COLOR_BGRA2GRAY}.get(enc, cv2.COLOR_BGR2GRAY)
        return np.ascontiguousarray(cv2.cvtColor(arr, code))


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthEvalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
