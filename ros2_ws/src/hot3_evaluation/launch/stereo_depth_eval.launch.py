import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory("hot3_evaluation")
    config_file = os.path.join(pkg, "config", "stereo_depth_eval.yaml")
    rviz_config = os.path.join(pkg, "config", "stereo_depth_eval.rviz")

    nodes = [
        launch_ros.actions.Node(
            package="hot3_evaluation",
            executable="stereo_depth_eval_node",
            name="stereo_depth_eval_node",
            output="screen",
            parameters=[config_file],
        )
    ]
    if LaunchConfiguration("rviz").perform(context).lower() == "true":
        nodes.append(
            launch_ros.actions.Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config],
                output="screen",
            )
        )
    return nodes


def generate_launch_description():
    return launch.LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Also launch RViz with the depth-eval layout.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
