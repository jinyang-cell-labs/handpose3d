import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("stereo_handpose_estimation")
    config_file = os.path.join(
        pkg_share, "config", "joint_triangulation_debug.yaml"
    )

    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="stereo_handpose_estimation",
                executable="joint_triangulation_debug_node",
                name="joint_triangulation_debug_node",
                output="screen",
                parameters=[config_file],
            ),
        ]
    )
