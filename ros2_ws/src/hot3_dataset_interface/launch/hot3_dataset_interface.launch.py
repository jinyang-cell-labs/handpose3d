import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _launch_setup(context, *args, **kwargs):
    config_file = os.path.join(
        get_package_share_directory("hot3_dataset_interface"),
        "config",
        "hot3_dataset_interface.yaml",
    )

    # The chosen clip lives in hot3_dataset_interface.yaml. Pass clip:=... to
    # override it without editing the config; an empty value (the default)
    # falls through to the config-file value.
    clip = LaunchConfiguration("clip").perform(context)
    loop = LaunchConfiguration("loop").perform(context).lower() == "true"

    overrides = {"loop": loop}
    if clip:
        overrides["clip"] = clip

    return [
        launch_ros.actions.Node(
            package="hot3_dataset_interface",
            executable="hot3_data_publisher_node",
            name="hot3_data_publisher_node",
            output="screen",
            parameters=[config_file, overrides],
        )
    ]


def generate_launch_description():
    return launch.LaunchDescription(
        [
            DeclareLaunchArgument(
                "clip",
                default_value="",
                description="Override the dataset clip folder "
                "(e.g. clip-001849). Empty keeps the config-file value.",
            ),
            DeclareLaunchArgument(
                "loop",
                default_value="true",
                description="Loop back to the first frame at end of clip.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
