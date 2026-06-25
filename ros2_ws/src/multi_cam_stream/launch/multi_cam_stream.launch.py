"""Launch the multi-USB-camera publisher.

    ros2 launch multi_cam_stream multi_cam_stream.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "multi_cam_stream"


def generate_launch_description():
    share = get_package_share_directory(PKG)
    default_config = os.path.join(share, "config", "multi_cam_stream.yaml")
    default_rviz = os.path.join(share, "config", "multi_cam_stream.rviz")

    config_arg = DeclareLaunchArgument(
        "config", default_value=default_config,
        description="Path to the camera stream YAML.",
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz", default_value="true",
        description="Launch RViz2 to view the camera streams.",
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config", default_value=default_rviz,
        description="Path to the RViz2 config.",
    )
    node = Node(
        package=PKG,
        executable="camera_stream_node",
        name="camera_stream_node",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )
    return LaunchDescription([config_arg, rviz_arg, rviz_config_arg, node, rviz])
