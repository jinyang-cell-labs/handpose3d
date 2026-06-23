import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory("depth_anything_ros")
    config_file = os.path.join(pkg_share, "config", "depth_anything_ros.yaml")
    rviz_config = os.path.join(pkg_share, "config", "depth_anything_ros.rviz")

    use_rviz = LaunchConfiguration("rviz")

    return launch.LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Launch RViz to view image_raw, the depth image and the cloud",
            ),
            launch_ros.actions.Node(
                package="depth_anything_ros",
                executable="depth_anything_node",
                name="depth_anything_node",
                output="screen",
                parameters=[config_file],
            ),
            launch_ros.actions.Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
                condition=launch.conditions.IfCondition(use_rviz),
            ),
        ]
    )
