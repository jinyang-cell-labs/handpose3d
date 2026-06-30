"""Single-camera AprilGrid board pose tracker + visualization.

    ros2 launch calibration_multi_cam board_pose.launch.py
    ros2 launch calibration_multi_cam board_pose.launch.py camera:=camera1

Detects the board in one camera stream, derives its pose from the camera's
already-calibrated intrinsics, broadcasts <camera> -> board over TF (and the
static world -> camera rig from extrinsics, so the board shows up in the world
frame for any camera), and republishes the image with the board axes drawn on
it (for eye-alignment).
"""
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "calibration_multi_cam"


def _launch_rviz(context, *args, **kwargs):
    """Render the RViz template for the selected camera, then launch RViz.

    The bundled config defaults to camera0; we rewrite the annotated-image topic
    to the camera being tracked so the Image panel isn't blank for camera1/2.
    The fixed frame stays the world (camera0) — the node publishes the rig TF, so
    the board is reachable from there regardless of which camera is tracked.
    """
    if context.launch_configurations.get("rviz", "true").lower() != "true":
        return []

    camera = LaunchConfiguration("camera").perform(context) or "camera0"
    rviz_in = LaunchConfiguration("rviz_config").perform(context)
    with open(rviz_in, "r") as fh:
        cfg = fh.read()
    cfg = cfg.replace("/camera0/board_pose/image_axes",
                      f"/{camera}/board_pose/image_axes")
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"board_pose_{camera}_", suffix=".rviz", delete=False, mode="w")
    tmp.write(cfg)
    tmp.close()

    return [Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", tmp.name],
    )]


def generate_launch_description():
    share = get_package_share_directory(PKG)
    default_config = os.path.join(share, "config", "calibration.yaml")
    default_rviz = os.path.join(share, "config", "board_pose.rviz")

    config_arg = DeclareLaunchArgument(
        "config", default_value=default_config,
        description="Path to the central calibration YAML.",
    )
    camera_arg = DeclareLaunchArgument(
        "camera", default_value="",
        description="Camera to track (empty = first in camera_names).",
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz", default_value="true",
        description="Launch RViz2 to view the board TF + annotated image.",
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config", default_value=default_rviz,
        description="Path to the RViz2 config template.",
    )
    board_pose = Node(
        package=PKG,
        executable="board_pose_node",
        name="calibration_board_pose",
        output="screen",
        parameters=[
            LaunchConfiguration("config"),
            {"board_pose.camera": LaunchConfiguration("camera")},
        ],
    )
    return LaunchDescription([
        config_arg, camera_arg, rviz_arg, rviz_config_arg,
        board_pose, OpaqueFunction(function=_launch_rviz),
    ])
