import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory("gui_service_call")
    default_config = os.path.join(pkg_share, "config", "services.yaml")

    config_file = LaunchConfiguration("config_file")

    return launch.LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="YAML file defining the service buttons",
            ),
            launch_ros.actions.Node(
                package="gui_service_call",
                executable="service_caller_node",
                name="gui_service_call",
                output="screen",
                parameters=[{"config_file": config_file}],
            ),
        ]
    )
