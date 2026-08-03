"""Launch the multi-camera hand teleop pipeline.

    ros2 launch body_cam_teleop body_cam_teleop.launch.py                # yaml as-is
    ros2 launch body_cam_teleop body_cam_teleop.launch.py cameras:=cam0  # subset
    ros2 launch body_cam_teleop body_cam_teleop.launch.py rviz:=true     # + RViz2

The camera set is defined ONCE in config/body_cam_teleop.yaml: every top-level
"/<name>:" block is a camera namespace and gets an identical pipeline pair
(hand_landmarks_node + hand_pose_node) launched inside it. A single
teleop_mux_node merges the per-camera <ns>/teleop streams into
/teleop_converted (its input_topics parameter is injected here from the same
blocks). The camera sits at the operator body center, so operator_body is
just a fixed offset from the camera frame (identity by default); the
per-camera debug TF trees are disjoint, so a static identity transform
bridges each extra <ns>/operator_body to the first camera's (the frames
coincide by construction), letting RViz show every camera in one scene.

Launch arguments:
  cameras              "" (default) launches every camera block in the yaml;
                       otherwise a comma-separated subset of the block names
                       (e.g. cameras:=cam0 or cameras:=cam0,cam1).
  enable_reprojection  "auto" (default) keeps the yaml values; "true"/"false"
                       overrides BOTH nodes of every camera at once, so they
                       can never fall out of sync. Use false for deployment
                       (no overlay/markers — minimum CPU).
  rviz                 "true" opens RViz2 with the packaged view and forces
                       enable_reprojection on everywhere.
  calibrate_hand_scale "log" adds hand_scale_calib_node, which estimates the
                       optimal hand_size_scaling_factor from cross-camera
                       agreement and logs it; "apply" also live-updates the
                       hand_pose_nodes until the estimate converges (persist
                       the final value into body_cam_teleop.yaml manually).
                       Forces enable_reprojection on (needs the markers).
  perf                 "true" adds perf_monitor_node: it records the per-stage
                       timings every pipeline node publishes on its
                       body_cam_teleop/perf topic (enable_perf, on by default)
                       plus per-process CPU/RSS and topic rates into CSVs
                       under perf_log_dir. Analyze a run afterwards with
                       scripts/perf_report.py (see the package README).
  perf_log_dir         Directory for the perf CSVs.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _camera_namespaces(cfg):
    """Camera namespace blocks ("/cam0", ...) declared in the config yaml."""
    return [
        key
        for key, value in cfg.items()
        if key.startswith("/") and "*" not in key and isinstance(value, dict) and "hand_landmarks_node" in value
    ]


def _setup(context):
    share = get_package_share_directory("body_cam_teleop")
    config = os.path.join(share, "config", "body_cam_teleop.yaml")
    calibration_dir = os.path.join(share, "config")
    rviz_config = os.path.join(share, "config", "body_cam_teleop.rviz")

    rviz_on = LaunchConfiguration("rviz").perform(context).lower() in ("true", "1")
    reproj_arg = LaunchConfiguration("enable_reprojection").perform(context).lower()
    cameras_arg = LaunchConfiguration("cameras").perform(context).strip()
    calib_arg = LaunchConfiguration("calibrate_hand_scale").perform(context).lower()
    calib_mode = {
        "false": None,
        "0": None,
        "": None,
        "log": "log",
        "true": "log",
        "1": "log",
        "apply": "apply",
    }.get(calib_arg, "invalid")
    if calib_mode == "invalid":
        raise RuntimeError(f"calibrate_hand_scale:={calib_arg}: use false, log or apply")

    with open(config) as f:
        cfg = yaml.safe_load(f)
    namespaces = _camera_namespaces(cfg)
    if not namespaces:
        raise RuntimeError(f"no camera namespace blocks found in {config}")
    if cameras_arg:
        wanted = ["/" + c.strip().lstrip("/") for c in cameras_arg.split(",") if c.strip()]
        unknown = sorted(set(wanted) - set(namespaces))
        if unknown:
            raise RuntimeError(f"cameras:={cameras_arg}: {unknown} not defined in {config} (available: {namespaces})")
        namespaces = wanted

    # rviz and the hand-scale calibration need the reprojection data (markers);
    # an explicit true/false wins over the yaml; "auto" leaves whatever the
    # yaml says (both nodes read the same key).
    overrides = {}
    if rviz_on or calib_mode or reproj_arg in ("true", "1"):
        overrides["enable_reprojection"] = True
    elif reproj_arg in ("false", "0"):
        overrides["enable_reprojection"] = False

    nodes = []
    for ns in namespaces:
        nodes += [
            Node(
                package="body_cam_teleop",
                executable="hand_landmarks_node.py",
                name="hand_landmarks_node",
                namespace=ns,
                output="screen",
                parameters=[config, overrides] if overrides else [config],
            ),
            Node(
                package="body_cam_teleop",
                executable="hand_pose_node",
                name="hand_pose_node",
                namespace=ns,
                output="screen",
                parameters=[config, {"calibration_dir": calibration_dir, **overrides}],
            ),
        ]
    nodes.append(
        Node(
            package="body_cam_teleop",
            executable="teleop_mux_node.py",
            name="teleop_mux_node",
            output="screen",
            parameters=[config, {"input_topics": [f"{ns}/teleop" for ns in namespaces]}],
        )
    )
    # Finger-curl gesture extraction; which camera it listens to is set by
    # hand_gesture_mapping_node.camera_namespace in the yaml. Only launched
    # when that camera is part of this run (cameras:= may exclude it).
    gesture_ns = "/" + cfg.get("hand_gesture_mapping_node", {}).get("ros__parameters", {}).get(
        "camera_namespace", "cam0"
    )
    if gesture_ns in namespaces:
        nodes.append(
            Node(
                package="body_cam_teleop",
                executable="hand_gesture_mapping_node",
                name="hand_gesture_mapping_node",
                output="screen",
                parameters=[config],
            )
        )
    else:
        print(
            f"[body_cam_teleop.launch] hand_gesture_mapping_node skipped: its camera "
            f"{gesture_ns} is not in the launched set {namespaces}"
        )
    # Each hand_pose_node broadcasts a debug TF tree rooted at its own
    # <ns>/operator_body, so the per-camera trees are disjoint. The frames
    # coincide by construction (same fixed camera->body offset), so bridge
    # every extra tree to the first with a static identity transform — this is
    # what lets RViz render all cameras' markers/TF in one scene.
    body_frame = cfg.get("/**/hand_pose_node", {}).get("ros__parameters", {}).get("body_frame", "operator_body")
    root_body = f"{namespaces[0].lstrip('/')}/{body_frame}"
    for ns in namespaces[1:]:
        ns_body = f"{ns.lstrip('/')}/{body_frame}"
        nodes.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=f"operator_body_bridge_{ns.lstrip('/')}",
                output="screen",
                arguments=[
                    "--frame-id",
                    root_body,
                    "--child-frame-id",
                    ns_body,
                ],
            )
        )
    if calib_mode:
        if len(namespaces) < 2:
            raise RuntimeError(f"calibrate_hand_scale needs at least two cameras (launching {namespaces})")
        shared_pose = cfg.get("/**/hand_pose_node", {}).get("ros__parameters", {})
        nodes.append(
            Node(
                package="body_cam_teleop",
                executable="hand_scale_calib_node.py",
                name="hand_scale_calib_node",
                output="screen",
                parameters=[
                    {
                        "camera_namespaces": [ns.lstrip("/") for ns in namespaces],
                        "camera_frames": [
                            cfg[ns]["hand_pose_node"]["ros__parameters"]["camera_name"] for ns in namespaces
                        ],
                        "body_frame": body_frame,
                        "apply": calib_mode == "apply",
                        "initial_factor": float(shared_pose.get("hand_size_scaling_factor", 1.3)),
                        "log_file": LaunchConfiguration("calib_log_file").perform(context),
                    }
                ],
            )
        )
    if LaunchConfiguration("perf").perform(context).lower() in ("true", "1"):
        nodes.append(
            Node(
                package="body_cam_teleop",
                executable="perf_monitor_node.py",
                name="perf_monitor_node",
                output="screen",
                parameters=[
                    {
                        "log_dir": LaunchConfiguration("perf_log_dir").perform(context),
                        "run_tag": "_".join(ns.lstrip("/") for ns in namespaces),
                    }
                ],
            )
        )
    if rviz_on:
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "cameras",
                default_value="",
                description="Comma-separated subset of the camera blocks in "
                "body_cam_teleop.yaml (e.g. cam0,cam1); empty = all.",
            ),
            DeclareLaunchArgument(
                "enable_reprojection",
                default_value="auto",
                description="Override the yaml on all nodes: true/false; auto = keep the yaml values.",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Open RViz2 with the body_cam_teleop view (implies enable_reprojection:=true).",
            ),
            DeclareLaunchArgument(
                "calib_log_file",
                default_value="/workspace/robot/ros2_ws/logs/calib.csv",
                description="CSV path for hand_scale_calib_node's per-window "
                "estimates (empty = no CSV; console log only).",
            ),
            DeclareLaunchArgument(
                "calibrate_hand_scale",
                default_value="false",
                description="Run hand_scale_calib_node to estimate the optimal "
                "hand_size_scaling_factor from cross-camera agreement "
                "(implies enable_reprojection:=true). 'log' only "
                "reports the recommendation; 'apply' also pushes it "
                "to the hand_pose_nodes each window until converged. "
                "Persist the result into body_cam_teleop.yaml manually.",
            ),
            DeclareLaunchArgument(
                "perf",
                default_value="false",
                description="Record pipeline performance (per-stage timings, "
                "process CPU/RSS, topic rates) to CSVs in perf_log_dir; "
                "analyze with scripts/perf_report.py.",
            ),
            DeclareLaunchArgument(
                "perf_log_dir",
                default_value="/workspace/robot/ros2_ws/logs/perf",
                description="Directory for perf_monitor_node's CSV output.",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
