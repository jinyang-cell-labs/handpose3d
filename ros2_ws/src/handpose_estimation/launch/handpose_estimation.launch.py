import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory("handpose_estimation")
    config_file = os.path.join(pkg_share, "config", "handpose_estimation.yaml")
    rviz_config = os.path.join(pkg_share, "config", "handpose3d.rviz")

    use_rviz = LaunchConfiguration("rviz")

    return launch.LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz", default_value="true", description="Launch RViz for 3D visualization"
            ),
            launch_ros.actions.Node(
                package="handpose_estimation",
                executable="handpose_node",
                name="handpose_node",
                output="screen",
                parameters=[config_file],
            ),
            # Make the 'world' frame exist in the TF tree so RViz's fixed frame
            # resolves (identity transform world -> camera0).
            launch_ros.actions.Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="world_to_camera0",
                output="screen",
                arguments=["0", "0", "0", "0", "0", "0", "world", "camera0"],
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
