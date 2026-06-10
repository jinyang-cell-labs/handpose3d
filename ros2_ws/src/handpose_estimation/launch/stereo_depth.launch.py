import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory("handpose_estimation")
    config_file = os.path.join(pkg_share, "config", "stereo_depth.yaml")
    rviz_config = os.path.join(pkg_share, "config", "stereo_depth.rviz")

    use_rviz = LaunchConfiguration("rviz")

    return launch.LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz", default_value="true", description="Launch RViz to view the depth image"
            ),
            launch_ros.actions.Node(
                package="handpose_estimation",
                executable="stereo_depth_node",
                name="stereo_depth_node",
                output="screen",
                parameters=[config_file],
                # The node subscribes to <name>/image_raw and <name>/camera_info.
                # The camera_info topics already match; remap the image topics to
                # the rotated streams (same as handpose_estimation.launch.py).
                remappings=[
                    ("camera0/image_raw", "/camera0_rot/image_rotated"),
                    ("camera1/image_raw", "/camera1_rot/image_rotated"),
                ],
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
