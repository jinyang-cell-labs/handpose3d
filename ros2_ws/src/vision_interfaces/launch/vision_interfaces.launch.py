import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory("vision_interfaces"),
        "config",
        "vision_interfaces.yaml",
    )

    loop = LaunchConfiguration("loop")
    replay_speed = LaunchConfiguration("replay_speed")

    return launch.LaunchDescription(
        [
            DeclareLaunchArgument(
                "loop",
                default_value="true",
                description="Loop video files when they reach EOF (video mode).",
            ),
            DeclareLaunchArgument(
                "replay_speed",
                default_value="1.0",
                description="Video playback speed multiplier (>1 faster, <1 slower).",
            ),
            launch_ros.actions.Node(
                package="vision_interfaces",
                executable="camera_publisher_node",
                name="camera_publisher_node",
                output="screen",
                parameters=[
                    config_file,
                    # Launch-arg overrides on top of the config file.
                    {"loop": ParameterValue(loop, value_type=bool)},
                    {"replay_speed": ParameterValue(replay_speed, value_type=float)},
                ],
            ),
        ]
    )
