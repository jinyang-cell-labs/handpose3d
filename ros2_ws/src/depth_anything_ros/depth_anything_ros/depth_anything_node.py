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
from sensor_msgs.msg import Image

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
        self.sub = self.create_subscription(
            Image, self.image_topic, self._on_image, qos_profile_sensor_data
        )

        self.get_logger().info(
            f"depth_anything_node ready: {'metric' if self.metric else 'relative'} "
            f"{self.encoder} on {self.device}. '{self.image_topic}' -> "
            f"'{self.depth_topic}'"
            + (f" (+ '{self.depth_color_topic}')" if self.publish_color else "")
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

        self._publish_depth(msg.header, depth)
        if self.color_pub is not None:
            self._publish_color(msg.header, depth)

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
