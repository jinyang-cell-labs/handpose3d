"""Publish a previously computed calibration.

    ros2 launch calibration_multi_cam publish.launch.py

Loads `result_file` and continuously publishes intrinsics-only CameraInfo
(latched) plus the rig extrinsics over static TF / PoseArray (world == camera0).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "calibration_multi_cam"


def generate_launch_description():
    share = get_package_share_directory(PKG)
    default_config = os.path.join(share, "config", "calibration.yaml")
    default_rviz = os.path.join(share, "config", "extrinsics.rviz")

    config_arg = DeclareLaunchArgument(
        "config", default_value=default_config,
        description="Path to the central calibration YAML.",
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz", default_value="true",
        description="Launch RViz2 to view the camera TF/poses.",
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config", default_value=default_rviz,
        description="Path to the RViz2 config.",
    )
    publisher = Node(
        package=PKG,
        executable="publisher_node",
        name="calibration_publisher",
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
    return LaunchDescription([config_arg, rviz_arg, rviz_config_arg, publisher, rviz])
