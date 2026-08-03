# PolicyAnalytics.msg - Per-cycle policy executor timing breakdown.
# Published every inference cycle (typically 50 Hz).

std_msgs/Header header

float32 read_state_ms           # Sensor read latency this step (ms)
float32 preprocessing_ms        # Observation building latency this step (ms)
float32 inference_ms            # ONNX inference latency this step (ms)
float32 postprocessing_ms       # Command preparation latency this step (ms)
float32 total_cycle_ms          # Total control cycle latency this step (ms)
float32 observation_age_us      # Time since feedback was stamped when inference reads it (us)
float32 imu_age_us              # Time since IMU was stamped when inference reads it (us)
