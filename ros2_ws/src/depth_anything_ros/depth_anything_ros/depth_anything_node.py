#!/usr/bin/env python3

"""
Depth-Anything-V2 depth estimation node.

Subscribes to a color ``sensor_msgs/Image`` stream (e.g. a RealSense
``color/image_raw``), runs Depth-Anything-V2 on the GPU and republishes:

  * ``<depth_topic>``       — ``32FC1`` depth image. With a *relative* model the
    values are unitless inverse-depth-like relative depth; with a *metric*
    model they are true metres.
  * ``<depth_color_topic>`` — ``bgr8`` colorized preview for RViz (optional).

The header (stamp + frame_id) of every input frame is copied onto the outputs
so depth aligns with the source image in the TF tree.

Everything is config-driven (see config/depth_anything_ros.yaml). The model is
loaded once at startup and kept GPU-resident. Defaults target the relative
``vits`` checkpoint that already ships under src/third_party; switch to a
metric checkpoint by setting ``metric: true`` + the matching ``checkpoint``.

The Depth-Anything-V2 source is vendored (not pip-installed), so its package
directory is added to ``sys.path`` at startup. The relative and metric variants
ship two *different* packages both named ``depth_anything_v2``; only one is ever
imported per process, selected by the ``metric`` parameter.
"""

import os
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

# Encoder -> architecture config, copied verbatim from Depth-Anything-V2/run.py.
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

# Named OpenCV colormaps usable for the preview image.
COLORMAPS = {
    "INFERNO": cv2.COLORMAP_INFERNO,
    "MAGMA": cv2.COLORMAP_MAGMA,
    "JET": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
}


class DepthAnythingNode(Node):
    def __init__(self):
        super().__init__("depth_anything_node")

        # --- parameters -----------------------------------------------------
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "depth_anything/depth")
        self.declare_parameter("depth_color_topic", "depth_anything/depth_color")
        self.declare_parameter("publish_color", True)

        # Point cloud. Reprojects every (sub-sampled) pixel to 3D using the
        # camera intrinsics and the estimated depth, colored from image_raw.
        self.declare_parameter("publish_cloud", True)
        self.declare_parameter("cloud_topic", "depth_anything/points")
        # CameraInfo source for intrinsics. Empty -> derived from image_topic
        # (".../image_raw" -> ".../camera_info"), matching the vision_interfaces
        # frontend convention.
        self.declare_parameter("camera_info_topic", "")
        # Manual intrinsics override (used when all four are > 0); otherwise the
        # node waits for CameraInfo.
        self.declare_parameter("fx", 0.0)
        self.declare_parameter("fy", 0.0)
        self.declare_parameter("cx", 0.0)
        self.declare_parameter("cy", 0.0)
        # Subsample stride for the cloud (1 = every pixel, 2 = every other...).
        self.declare_parameter("cloud_stride", 2)
        # Keep only points within [min, max] metres (max<=0 -> no upper bound,
        # or max_depth when metric).
        self.declare_parameter("cloud_min_range", 0.1)
        self.declare_parameter("cloud_max_range", 0.0)
        # Stamp all outputs with this frame_id instead of the source image's.
        # Empty -> preserve the incoming header frame_id.
        self.declare_parameter("output_frame", "")

        # Vendored Depth-Anything-V2 checkout. For metric models the source
        # lives in the metric_depth/ subfolder of the same checkout.
        self.declare_parameter(
            "dav2_root", "/workspace/ros2_ws/src/third_party/Depth-Anything-V2"
        )
        self.declare_parameter("metric", False)
        self.declare_parameter("encoder", "vits")
        # Empty -> derive the default checkpoint path from dav2_root + encoder.
        self.declare_parameter("checkpoint", "")
        # Only used when metric=true: upper bound (metres) the model is trained
        # to. Hypersim (indoor)=20.0, VKITTI (outdoor)=80.0.
        self.declare_parameter("max_depth", 20.0)
        self.declare_parameter("input_size", 518)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("colormap", "INFERNO")

        self.image_topic = self.get_parameter("image_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.depth_color_topic = self.get_parameter("depth_color_topic").value
        self.publish_color = bool(self.get_parameter("publish_color").value)
        self.publish_cloud = bool(self.get_parameter("publish_cloud").value)
        self.cloud_topic = self.get_parameter("cloud_topic").value
        self.cloud_stride = max(1, int(self.get_parameter("cloud_stride").value))
        self.cloud_min_range = float(self.get_parameter("cloud_min_range").value)
        self.cloud_max_range = float(self.get_parameter("cloud_max_range").value)
        self.output_frame = self.get_parameter("output_frame").value
        self.dav2_root = self.get_parameter("dav2_root").value
        self.metric = bool(self.get_parameter("metric").value)
        self.encoder = self.get_parameter("encoder").value
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.input_size = int(self.get_parameter("input_size").value)
        self.device = self.get_parameter("device").value
        cmap_name = str(self.get_parameter("colormap").value).upper()
        self.colormap = COLORMAPS.get(cmap_name, cv2.COLORMAP_INFERNO)

        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown encoder '{self.encoder}'. Choose one of {list(MODEL_CONFIGS)}."
            )

        checkpoint = self.get_parameter("checkpoint").value
        if not checkpoint:
            sub = "metric_depth/checkpoints" if self.metric else "checkpoints"
            checkpoint = os.path.join(
                self.dav2_root, sub, f"depth_anything_v2_{self.encoder}.pth"
            )
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}. Download the "
                f"{'metric' if self.metric else 'relative'} {self.encoder} "
                "checkpoint from the Depth-Anything-V2 model zoo."
            )

        # --- intrinsics -----------------------------------------------------
        # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]. Either supplied manually or
        # captured (once) from CameraInfo. None until known; the cloud is not
        # published before then.
        self.K = None
        self._uv_cache = {}  # (h, w) -> (u_grid, v_grid) for the strided pixels
        fx = float(self.get_parameter("fx").value)
        fy = float(self.get_parameter("fy").value)
        cx = float(self.get_parameter("cx").value)
        cy = float(self.get_parameter("cy").value)
        if min(fx, fy, cx, cy) > 0.0:
            self.K = (fx, fy, cx, cy)
            self.get_logger().info(f"Using manual intrinsics fx={fx} fy={fy} cx={cx} cy={cy}")

        if self.publish_cloud and not self.metric:
            self.get_logger().warn(
                "publish_cloud=true with the RELATIVE model: its output is "
                "unitless inverse-depth, so the cloud is NOT metric and is "
                "geometrically distorted. Use a metric checkpoint (metric=true) "
                "for a correct point cloud."
            )

        # --- model ----------------------------------------------------------
        self.model = self._load_model(checkpoint)

        # --- pub / sub ------------------------------------------------------
        self.depth_pub = self.create_publisher(
            Image, self.depth_topic, qos_profile_sensor_data
        )
        self.color_pub = (
            self.create_publisher(Image, self.depth_color_topic, qos_profile_sensor_data)
            if self.publish_color
            else None
        )
        self.cloud_pub = (
            self.create_publisher(PointCloud2, self.cloud_topic, qos_profile_sensor_data)
            if self.publish_cloud
            else None
        )

        if self.publish_cloud and self.K is None:
            info_topic = self.get_parameter("camera_info_topic").value
            if not info_topic:
                base = self.image_topic.rsplit("/", 1)[0]
                info_topic = f"{base}/camera_info"
            self.info_sub = self.create_subscription(
                CameraInfo, info_topic, self._on_camera_info, qos_profile_sensor_data
            )
            self.get_logger().info(f"Waiting for intrinsics on '{info_topic}'...")

        self.sub = self.create_subscription(
            Image, self.image_topic, self._on_image, qos_profile_sensor_data
        )

        self.get_logger().info(
            f"depth_anything_node ready: {'metric' if self.metric else 'relative'} "
            f"{self.encoder} on {self.device}. '{self.image_topic}' -> "
            f"'{self.depth_topic}'"
            + (f" (+ '{self.depth_color_topic}')" if self.publish_color else "")
            + (f" (+ cloud '{self.cloud_topic}')" if self.publish_cloud else "")
        )

    def _on_camera_info(self, msg):
        if self.K is not None:
            return  # intrinsics are static; capture once
        k = np.array(msg.k, dtype=float).reshape(3, 3)
        self.K = (k[0, 0], k[1, 1], k[0, 2], k[1, 2])
        self.get_logger().info(
            f"Captured intrinsics fx={self.K[0]:.1f} fy={self.K[1]:.1f} "
            f"cx={self.K[2]:.1f} cy={self.K[3]:.1f}"
        )

    def _load_model(self, checkpoint):
        # Put the right vendored package on sys.path. metric_depth/ contains its
        # own depth_anything_v2 package whose model multiplies the head output
        # by max_depth to produce metres.
        root = os.path.join(self.dav2_root, "metric_depth") if self.metric else self.dav2_root
        if root not in sys.path:
            sys.path.insert(0, root)

        import torch  # imported here so the import error is logged by the node
        from depth_anything_v2.dpt import DepthAnythingV2

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.get_logger().warn("CUDA not available; falling back to CPU (slow).")
            self.device = "cpu"
        self._torch = torch

        cfg = dict(MODEL_CONFIGS[self.encoder])
        if self.metric:
            cfg["max_depth"] = self.max_depth
        model = DepthAnythingV2(**cfg)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        return model.to(self.device).eval()

    # --------------------------------------------------------------- callback
    def _on_image(self, msg):
        frame_bgr = self._to_bgr(msg)
        if frame_bgr is None:
            return

        # Returns an HxW float32 array on CPU. The metric variant decorates
        # infer_image with no_grad but the relative one does not, so guard it
        # here for both.
        with self._torch.inference_mode():
            depth = self.model.infer_image(frame_bgr, self.input_size).astype(np.float32)

        header = self._out_header(msg.header)
        self._publish_depth(header, depth)
        if self.color_pub is not None:
            self._publish_color(header, depth)
        if self.cloud_pub is not None and self.K is not None:
            self._publish_cloud(header, depth, frame_bgr)

    def _out_header(self, src_header):
        if self.output_frame:
            src_header.frame_id = self.output_frame
        return src_header

    def _to_bgr(self, msg):
        """Decode an incoming color Image into a contiguous HxWx3 BGR array."""
        h, w = msg.height, msg.width
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        enc = msg.encoding.lower()

        if enc in ("rgb8", "bgr8"):
            channels = 3
        elif enc in ("rgba8", "bgra8"):
            channels = 4
        else:
            self.get_logger().warn(
                f"Unsupported image encoding '{msg.encoding}'; expected "
                "rgb8/bgr8/rgba8/bgra8.",
                throttle_duration_sec=5.0,
            )
            return None

        # step is the row stride in bytes and may include padding.
        img = buf.reshape(h, msg.step)[:, : w * channels].reshape(h, w, channels)
        if enc == "rgb8":
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if enc == "rgba8":
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        if enc == "bgra8":
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return np.ascontiguousarray(img)  # already bgr8

    def _publish_depth(self, header, depth):
        h, w = depth.shape[:2]
        out = Image()
        out.header = header  # preserve source stamp + frame_id
        out.height = h
        out.width = w
        out.encoding = "32FC1"
        out.is_bigendian = 0
        out.step = w * 4
        out.data = np.ascontiguousarray(depth, dtype=np.float32).tobytes()
        self.depth_pub.publish(out)

    def _publish_color(self, header, depth):
        if self.metric:
            # Clip to the model's metric range, near=bright.
            norm = np.clip(depth, 0.0, self.max_depth) / self.max_depth
        else:
            dmin, dmax = float(depth.min()), float(depth.max())
            norm = (depth - dmin) / (dmax - dmin) if dmax > dmin else np.zeros_like(depth)
        u8 = (norm * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(u8, self.colormap)  # -> bgr8

        h, w = color.shape[:2]
        out = Image()
        out.header = header
        out.height = h
        out.width = w
        out.encoding = "bgr8"
        out.is_bigendian = 0
        out.step = w * 3
        out.data = np.ascontiguousarray(color).tobytes()
        self.color_pub.publish(out)

    def _publish_cloud(self, header, depth, frame_bgr):
        """Back-project pixels to a colored XYZRGB cloud in the optical frame.

        Optical-frame convention (REP 103): x right, y down, z forward.
            x = (u - cx) * z / fx,  y = (v - cy) * z / fy,  z = depth(u, v)
        """
        fx, fy, cx, cy = self.K
        s = self.cloud_stride
        h, w = depth.shape[:2]

        # Cache the (sub-sampled) pixel coordinate grids per resolution.
        cache = self._uv_cache.get((h, w))
        if cache is None:
            vs = np.arange(0, h, s, dtype=np.float32)
            us = np.arange(0, w, s, dtype=np.float32)
            u_grid, v_grid = np.meshgrid(us, vs)
            cache = (u_grid.ravel(), v_grid.ravel())
            self._uv_cache[(h, w)] = cache
        u_flat, v_flat = cache

        z = depth[::s, ::s].ravel()

        # Validity mask: finite, positive, within range.
        valid = np.isfinite(z) & (z > self.cloud_min_range)
        upper = self.cloud_max_range
        if upper <= 0.0 and self.metric:
            upper = self.max_depth
        if upper > 0.0:
            valid &= z <= upper
        if not np.any(valid):
            return

        z = z[valid]
        u = u_flat[valid]
        v = v_flat[valid]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Color from the source frame (BGR) packed into a single float32 field.
        bgr = frame_bgr[::s, ::s].reshape(-1, 3)[valid]
        rgb_u32 = (
            (bgr[:, 2].astype(np.uint32) << 16)
            | (bgr[:, 1].astype(np.uint32) << 8)
            | bgr[:, 0].astype(np.uint32)
        )

        cloud = np.empty(
            z.shape[0],
            dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32)],
        )
        cloud["x"] = x
        cloud["y"] = y
        cloud["z"] = z
        cloud["rgb"] = rgb_u32.view(np.float32)

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = cloud.shape[0]
        msg.is_bigendian = False
        msg.is_dense = True  # invalid points already filtered out
        msg.point_step = 16
        msg.row_step = msg.point_step * cloud.shape[0]
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = cloud.tobytes()
        self.cloud_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthAnythingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
