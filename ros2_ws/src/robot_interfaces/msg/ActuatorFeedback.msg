# ActuatorFeedback.msg - Hardware feedback from actuators
# All arrays must be equal length, identified by name

# MotorState enum constants
uint8 MOTOR_STATE_UNKNOWN=0
uint8 MOTOR_STATE_INITIAL=1
uint8 MOTOR_STATE_DISABLED=2
uint8 MOTOR_STATE_DISABLING=3
uint8 MOTOR_STATE_ENABLED=4
uint8 MOTOR_STATE_ENABLING=5
uint8 MOTOR_STATE_HOLD_POSITION=6
uint8 MOTOR_STATE_ZERO_POINT_ALIGNMENT=7
uint8 MOTOR_STATE_RUNNING=8
uint8 MOTOR_STATE_ERROR=9
uint8 MOTOR_STATE_OVERHEATING=10
uint8 MOTOR_STATE_INVALID_ELECTRIC_ALIGNMENT=11

std_msgs/Header header

string[] name           # Which actuators
float64[] position      # Motor position [rad]
float64[] velocity      # Motor velocity [rad/s]
float64[] effort        # Q-current [A] or torque [Nm] depending on calibration
float64[] voltage       # Bus voltage [V]
float64[] temp_motor    # Motor temperature [°C]
float64[] temp_pcb      # PCB temperature [°C]
uint8[] motor_state     # Motor state per actuator (see MOTOR_STATE_* constants)
