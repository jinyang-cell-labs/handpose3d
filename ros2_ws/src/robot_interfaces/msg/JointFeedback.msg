# JointFeedback.msg - feedback for any robot in joint space
# All arrays must be equal length, identified by names

std_msgs/Header header

string[] name           # Which joints
float64[] position      # Measured position [rad]
float64[] velocity      # Measured velocity [rad/s]
float64[] effort        # Measured torque [Nm]
