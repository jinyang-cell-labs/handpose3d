std_msgs/Header header

# Send timing
float64 udp_send_interval_us            # Time between consecutive UDP send cycles [us]
float64 actuator_command_interval_us     # Time between consecutive actuator_command messages [us]
float64 command_to_send_us              # Time from command received to UDP sent [us] (NaN if no cmd pending)

# Per-motor feedback intervals (parallel arrays, ordered by actuator name)
string[] motor_names                     # Actuator names
float64[] motor_feedback_interval_us     # Time since previous feedback for this motor [us] (NaN if first)

# IMU feedback interval
float64 imu_feedback_interval_us         # Time since previous IMU feedback [us] (NaN if disabled/first)
