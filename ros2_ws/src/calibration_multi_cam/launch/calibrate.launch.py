"""Launch the collector node for a multi-camera calibration session.

    ros2 launch calibration_multi_cam calibrate.launch.py

Move the AprilGrid through the cameras' shared field of view, watch the
collector's status log, then trigger the solve:

    ros2 service call /calibration_collector/calibrate std_srvs/srv/Trigger {}
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
    collector = Node(
        package=PKG,
        executable="collector_node",
        name="calibration_collector",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )
    return LaunchDescription([config_arg, collector])
