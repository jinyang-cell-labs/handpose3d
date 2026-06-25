"""Stage 2 - extrinsic calibration.

    ros2 launch calibration_multi_cam extrinsic.launch.py

Loads `intrinsics_file`. Move the AprilGrid across the cameras' overlapping
fields of view until the status log shows rig_connected=True, then:
    ros2 service call /calibration_extrinsic/calibrate std_srvs/srv/Trigger {}
which writes `extrinsics_file`.
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
        executable="extrinsic_calibrator_node",
        name="calibration_extrinsic",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )
    return LaunchDescription([config_arg, node])
