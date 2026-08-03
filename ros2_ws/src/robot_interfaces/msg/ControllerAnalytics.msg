# ControllerAnalytics.msg - Per-tick controller (arbiter) telemetry.

std_msgs/Header header

# Current mode: forwarding | damping | default_pose | holding
string mode

# Emergency latched (overrides mode → damping output)
bool emergency_latched

# Which subsystem sources are currently in the active set
string[] active_subsystems

# Manual lock holder, or empty if no manual source is locked in
string manual_lock_holder

# Loop timing (last sample, not running stats)
float32 tick_period_ms          # observed time between consecutive ticks (ms)
float32 tick_latency_ms         # cost of the tick callback (ms)

# Per-source freshness. Parallel arrays — index i in source_names
# corresponds to index i in source_age_ms. age_ms < 0 means the source
# has never published.
string[] source_names
float32[] source_age_ms
