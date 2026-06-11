import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory("handpose_estimation_v2")
    config_file = os.path.join(pkg_share, "config", "handpose_estimation_v2.yaml")
    # RViz layout is shared with v1 (same marker/pose topics).
    v1_share = get_package_share_directory("handpose_estimation")
    rviz_config = os.path.join(v1_share, "config", "handpose3d.rviz")

    use_rviz = LaunchConfiguration("rviz")

    return launch.LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz", default_value="true", description="Launch RViz for 3D visualization"
            ),
            launch_ros.actions.Node(
                package="handpose_estimation_v2",
                executable="handpose_node_v2",
                name="handpose_node_v2",
                output="screen",
                parameters=[config_file],
                # The node subscribes to <name>/image_raw and <name>/camera_info.
                # The camera_info topics already match; remap the image topics to
                # the rotated streams. (frame_id stays camera0/camera1 for TF.)
                remappings=[
                    ("camera0/image_raw", "/camera0_rot/image_rotated"),
                    ("camera1/image_raw", "/camera1_rot/image_rotated"),
                ],
            ),
            # Note: handpose_node_v2 broadcasts the world->camera TF for each
            # camera from the calibration, which also provides RViz's 'world'
            # fixed frame — no separate static_transform_publisher needed.
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
