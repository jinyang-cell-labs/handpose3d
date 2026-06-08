import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory("vision_interfaces"),
        "config",
        "vision_interfaces.yaml",
    )

    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="vision_interfaces",
                executable="camera_publisher_node",
                name="camera_publisher_node",
                output="screen",
                parameters=[config_file],
            ),
        ]
    )
