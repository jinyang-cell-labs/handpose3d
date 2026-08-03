# ArmManipulationAnalytics.msg — per-tick analytics for the synthetic
# teleop driver (arm_manipulation/test_trajectory_node). Mirrors the
# pattern of ControllerAnalytics so downstream tooling stays consistent.

std_msgs/Header header

# Per-stage timing for one publish tick (milliseconds).
float32 compute_traj_ms        # Parametric trajectory eval (theta, sin/cos -> waist targets)
float32 convert_frame_ms       # Position basis change REP-103 -> OpenXR for both hands
float32 build_message_ms       # TeleopMessage allocation + field population
float32 publish_ms             # rclpy publisher.publish() call
float32 total_cycle_ms         # End-to-end tick latency

# Rolling-window mean cycle rate (Hz).
float32 publish_fps

# Current trajectory state.
float32 theta_rad                              # Phase angle on the parametric loop (rad)
geometry_msgs/Point left_target_waist          # Left target in REP-103 waist frame (m)
geometry_msgs/Point right_target_waist         # Right target in REP-103 waist frame (m)
