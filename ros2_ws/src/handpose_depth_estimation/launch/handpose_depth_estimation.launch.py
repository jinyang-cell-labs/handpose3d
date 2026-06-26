import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory("handpose_depth_estimation")
    config_file = os.path.join(
        pkg_share, "config", "handpose_depth_estimation.yaml"
    )
    rviz_config = os.path.join(pkg_share, "config", "handpose_depth.rviz")

    use_rviz = LaunchConfiguration("rviz")

    return launch.LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Launch RViz for the 3D skeleton + reprojection QA",
            ),
            launch_ros.actions.Node(
                package="handpose_depth_estimation",
                executable="handpose_depth_node",
                name="handpose_depth_node",
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
