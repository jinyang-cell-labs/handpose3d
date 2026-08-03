# JointCommand.msg - command for any robot in joint space
# All arrays must be equal length, identified by name

std_msgs/Header header

string[] name           # Which joints
float64[] position      # Target position [rad]
float64[] velocity      # Target velocity [rad/s]
float64[] effort        # Feedforward torque [Nm]
float64[] stiffness     # Position gain (kp) [Nm/rad]
float64[] damping       # Velocity gain (kd) [Nm·s/rad]
