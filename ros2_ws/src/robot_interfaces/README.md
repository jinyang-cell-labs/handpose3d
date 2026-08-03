# Robot Interfaces

Shared ROS2 interfaces for communcation.

## Architecture

This package defines a robot-agnostic message structure split into two distinct spaces:

1.  **Joint Space** (`JointCommand`, `JointFeedback`)
    *   Represents the kinematic joints of the robot (Generalized Coordinates).
    *   Used by high-level controllers and kinematics.

2.  **Actuator Space** (`ActuatorCommand`, `ActuatorFeedback`)
    *   Represents physical motors and drives.
    *   Used by hardware drivers.
    *   May differ from joint space (e.g., differential ankles, linkages).

## Usage

### Identifiers
Identifiers are specified as string names to ensure flexibility and readability:
*   **Joint Names**: String identifiers for joints (e.g., `knee_pitch_left`, `hip_roll_right`). See `joint_names.md` for a list of standard joint names.
*   **Actuator Names**: String identifiers for actuators (e.g., `left_knee_motor`, `right_hip_motor`). Actuator names may differ from joint names due to differential mechanisms or linkages.

### Services
Services are provided for hardware management (Start/Stop, Initialization, Configuration). These support:
*   **Batch Operations**: Controlling a specific list of actuators (e.g., `ActuatorStartStop`).
*   **Global Operations**: Controlling the entire robot at once (e.g., `StartStopRobotActuators`).