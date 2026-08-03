# ActuatorConfig.msg - Configuration parameters for a single actuator

string actuator_name                        # Actuator identifier
uint8 can_id                                # CAN-ID of the actuator
uint8 device_type                           # Device type (application-specific enum)
bool inverse_direction                      # Inverse the direction of the actuator
bool endstop_calibration_inverse            # Inverse the direction of the endstop calibration
bool endstop_calibration_skip               # Skip the requirement for endstops (e.g. for test purposes)
float32 endstop_calibration_damping         # Endstop calibration damping factor
float32 endstop_calibration_stall_threshold # Endstop calibration stall position threshold [rad]
float32 endstop_calibration_stall_duration  # Endstop calibration stall position duration [ms]
float32 endstop_zero_offset                 # Endstop to zero-point offset [rad]
float32 endstop_position_min                # Endstop position min [rad], relative to zero point
float32 endstop_position_max                # Endstop position max [rad], relative to zero point
float32 endstop_approach_threshold          # Endstop approach threshold [rad]
float32 endstop_restore_gain                # Endstop restore gain
float32 endstop_restore_damping             # Endstop restore damping
float32 endstop_restore_min_torque          # Endstop restore minimum torque
uint16 feedback_filter_cutoff               # Angle, velocity, torque IIR filter cutoff frequency [Hz]
uint16 control_filter_cutoff                # Angle, velocity control-loop filter cutoff frequency [Hz]
uint16 pid_bandwidth                        # PID bandwidth
uint8 control_freq_div                      # Control loop torque calculation frequency divider (1 = 20kHz max)
float32 kt_scale                            # Scaling coefficient of current/torque
