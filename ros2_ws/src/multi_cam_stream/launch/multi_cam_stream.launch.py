"""Launch the multi-USB-camera publisher.

    ros2 launch multi_cam_stream multi_cam_stream.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "multi_cam_stream"


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory(PKG), "config", "multi_cam_stream.yaml"
    )
    config_arg = DeclareLaunchArgument(
        "config", default_value=default_config,
        description="Path to the camera stream YAML.",
    )
    node = Node(
        package=PKG,
        executable="camera_stream_node",
        name="camera_stream_node",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )
    return LaunchDescription([config_arg, node])
