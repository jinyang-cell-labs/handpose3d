"""Publish a previously computed calibration.

    ros2 launch calibration_multi_cam publish.launch.py

Loads `result_file` and continuously publishes intrinsics-only CameraInfo
(latched) plus the rig extrinsics over static TF / PoseArray (world == cam0).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "calibration_multi_cam"


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory(PKG), "config", "calibration.yaml"
    )
    config_arg = DeclareLaunchArgument(
        "config", default_value=default_config,
        description="Path to the central calibration YAML.",
    )
    publisher = Node(
        package=PKG,
        executable="publisher_node",
        name="calibration_publisher",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )
    return LaunchDescription([config_arg, publisher])
