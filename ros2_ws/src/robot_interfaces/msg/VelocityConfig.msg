# Velocity pipeline config (teleop::velocity_node).
# Mirrors robot.teleop.VelocityConfig in lib/proto-messages.

float64[] max_velocity        # [Vx, Vy, Vw]
float64[] min_velocity        # [Vx, Vy, Vw]
float64[] max_accel           # [Ax, Ay, Aw]
float64[] max_decel           # [Ax, Ay, Aw] (negative)
float64 axis_deadband
float64[] deadband_velocity
