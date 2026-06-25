import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory("stereo_handpose_estimation")
    config_file = os.path.join(
        pkg_share, "config", "stereo_handpose_estimation.yaml"
    )
    ekf_config_file = os.path.join(pkg_share, "config", "hand_ekf.yaml")
    rviz_config = os.path.join(pkg_share, "config", "stereo_handpose.rviz")

    use_rviz = LaunchConfiguration("rviz")
    use_ekf = LaunchConfiguration("ekf")

    return launch.LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Launch RViz for 3D visualization",
            ),
            DeclareLaunchArgument(
                "ekf",
                default_value="true",
                description="Launch the hand_ekf_node to smooth hand centroids",
            ),
            launch_ros.actions.Node(
                package="stereo_handpose_estimation",
                executable="stereo_handpose_node",
                name="stereo_handpose_node",
                output="screen",
                parameters=[config_file],
            ),
            launch_ros.actions.Node(
                package="stereo_handpose_estimation",
                executable="hand_ekf_node",
                name="hand_ekf_node",
                output="screen",
                parameters=[ekf_config_file],
                condition=launch.conditions.IfCondition(use_ekf),
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
