"""Stage 1 - intrinsic calibration.

    ros2 launch calibration_multi_cam intrinsic.launch.py

Fill each camera's frame with the AprilGrid at several distances/angles, then:
    ros2 service call /calibration_intrinsic/calibrate std_srvs/srv/Trigger {}
which writes `intrinsics_file`.
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
        get_package_share_directory(PKG), "config", "calibration.yaml")
    config_arg = DeclareLaunchArgument(
        "config", default_value=default_config,
        description="Path to the central calibration YAML.")
    node = Node(
        package=PKG,
        executable="intrinsic_calibrator_node",
        name="calibration_intrinsic",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )
    return LaunchDescription([config_arg, node])
