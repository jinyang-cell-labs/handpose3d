"""Launch the standalone MediaPipe 2D hand-landmark debug viewer.

    ros2 launch body_cam_teleop mediapipe_detection_debug.launch.py
    ros2 launch body_cam_teleop mediapipe_detection_debug.launch.py rviz:=false
    ros2 launch body_cam_teleop mediapipe_detection_debug.launch.py camera_device:=6
    ros2 launch body_cam_teleop mediapipe_detection_debug.launch.py mirror_input:=true
    ros2 launch body_cam_teleop mediapipe_detection_debug.launch.py \
        enable_stereo:=true stereo_camera_device:=4

This is deliberately independent of body_cam_teleop.launch.py: its own node, its
own yaml (config/mediapipe_detection_debug.yaml) and its own RViz config, so it
can be run and tuned without touching the teleop pipeline.

V4L access is exclusive, so the debug node cannot share camera_device with a
running hand_landmarks_node — stop the pipeline first, or pass a different
device. The node fails fast with a clear message if the device is busy.

Launch arguments (each overrides the yaml only when given):
  rviz           "true" (default) opens RViz2 with the packaged debug view:
                 the annotated image plus the 2D skeleton on the image plane.
  namespace      Node namespace (default empty, i.e. the global namespace, which
                 is what the packaged RViz config's topics/frames expect). A
                 non-empty value prefixes both the topics and the TF frames, so
                 the RViz topics need the same prefix.
  camera_device  V4L device index or path.
  camera_name    Label used in the overlay and the image-plane frame name.
  mirror_input   Flip the frame before inference; this flips MediaPipe's
                 Left/Right classification (see the node docstring).
  enable_stereo  true/false: open a second camera and triangulate the hands
                 whose Left/Right label matches in both views into 3D
                 skeletons on mediapipe_debug/markers_3d (see the yaml's
                 stereo section for the calibration requirements).
  stereo_camera_device
                 V4L device index or path of the second camera.
  config         Path to an alternative parameter yaml.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    share = get_package_share_directory("body_cam_teleop")
    config = LaunchConfiguration("config").perform(context).strip()
    if not config:
        config = os.path.join(share, "config", "mediapipe_detection_debug.yaml")
    if not os.path.exists(config):
        raise RuntimeError(f"config:={config} does not exist")
    rviz_config = os.path.join(share, "config", "mediapipe_detection_debug.rviz")

    namespace = LaunchConfiguration("namespace").perform(context).strip()
    rviz_on = LaunchConfiguration("rviz").perform(context).lower() in ("true", "1")

    # Only pass through the arguments that were actually given, so the yaml
    # stays the single source of truth for everything else.
    overrides = {}
    for name, cast in (("camera_device", str), ("camera_name", str),
                       ("stereo_camera_device", str)):
        value = LaunchConfiguration(name).perform(context).strip()
        if value:
            overrides[name] = cast(value)
    for name in ("mirror_input", "enable_stereo"):
        value = LaunchConfiguration(name).perform(context).strip().lower()
        if value in ("true", "1"):
            overrides[name] = True
        elif value in ("false", "0"):
            overrides[name] = False

    nodes = [
        Node(
            package="body_cam_teleop",
            executable="mediapipe_detection_debug_node.py",
            name="mediapipe_detection_debug_node",
            namespace=namespace or None,
            output="screen",
            emulate_tty=True,
            parameters=[config, overrides] if overrides else [config],
        )
    ]
    if rviz_on:
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Open RViz2 with the packaged debug view "
                "(annotated image + 2D skeleton on the image plane).",
            ),
            DeclareLaunchArgument(
                "namespace",
                default_value="",
                description="Node namespace; prefixes the topics and the "
                "marker/TF frames. Empty (default) = global namespace, which is "
                "what the packaged RViz config expects.",
            ),
            DeclareLaunchArgument(
                "camera_device",
                default_value="",
                description="V4L device index or path; empty = keep the yaml value. "
                "Cannot be a device a running hand_landmarks_node holds.",
            ),
            DeclareLaunchArgument(
                "camera_name",
                default_value="",
                description="Label for the overlay and the image-plane frame; "
                "empty = keep the yaml value.",
            ),
            DeclareLaunchArgument(
                "mirror_input",
                default_value="",
                description="true/false to flip the frame before inference "
                "(flips MediaPipe's Left/Right label); empty = keep the yaml value.",
            ),
            DeclareLaunchArgument(
                "enable_stereo",
                default_value="",
                description="true/false to toggle the two-camera triangulation "
                "path (3D skeletons on mediapipe_debug/markers_3d); "
                "empty = keep the yaml value.",
            ),
            DeclareLaunchArgument(
                "stereo_camera_device",
                default_value="",
                description="V4L device index or path of the second camera; "
                "empty = keep the yaml value.",
            ),
            DeclareLaunchArgument(
                "config",
                default_value="",
                description="Alternative parameter yaml; empty = the packaged "
                "config/mediapipe_detection_debug.yaml.",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
