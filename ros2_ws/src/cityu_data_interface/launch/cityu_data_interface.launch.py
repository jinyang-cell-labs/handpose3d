import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _launch_setup(context, *args, **kwargs):
    config_file = os.path.join(
        get_package_share_directory("cityu_data_interface"),
        "config",
        "cityu_data_interface.yaml",
    )

    # The chosen sequence lives in cityu_data_interface.yaml. Pass sequence:=...
    # to override it without editing the config; an empty value (the default)
    # falls through to the config-file value.
    sequence = LaunchConfiguration("sequence").perform(context)
    loop = LaunchConfiguration("loop").perform(context).lower() == "true"

    overrides = {"loop": loop}
    if sequence:
        overrides["sequence"] = sequence

    return [
        launch_ros.actions.Node(
            package="cityu_data_interface",
            executable="cityu_data_publisher_node",
            name="cityu_data_publisher_node",
            output="screen",
            parameters=[config_file, overrides],
        )
    ]


def generate_launch_description():
    return launch.LaunchDescription(
        [
            DeclareLaunchArgument(
                "sequence",
                default_value="",
                description="Override the dataset sequence folder "
                "(e.g. B1Counting). Empty keeps the config-file value.",
            ),
            DeclareLaunchArgument(
                "loop",
                default_value="true",
                description="Loop back to the first frame at end of sequence.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
