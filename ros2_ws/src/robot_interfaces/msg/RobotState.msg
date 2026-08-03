# Message definition for the states of the Gen2 robot state machine.

# Robot state constants
uint8 STATE_DEFAULT = 0
uint8 STATE_STANDBY = 1
uint8 STATE_IDLE = 2
uint8 STATE_OPERATING = 3
uint8 STATE_SHUTDOWN = 4
uint8 STATE_OBJECT_GRASPING = 5

# Robot state (values defined above)
uint8 state

# Flag indicating if the current state is active (all required topics are available)
bool active
