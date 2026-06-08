#!/bin/bash
# Sourced from .bashrc for interactive shells in the handpose3d container.

WS=/workspace/ros2_ws

# Build aliases
alias cb="colcon build"
alias cbs="colcon build --packages-select"
alias cbc="cb --cmake-clean-cache"
alias ccw="colcon clean workspace"

# Convenience aliases
alias sw="source $WS/install/setup.bash"
alias sv="source $WS/.venv/bin/activate"

# uv on PATH
export PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

# Source ROS 2
source /opt/ros/jazzy/setup.bash

# colcon argcomplete
if [ -f /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash ]; then
    source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
fi

# Make the uv venv packages (mediapipe, opencv, ...) importable from the
# system interpreter that runs the ROS 2 entry-point scripts.
if [ -d "$WS/.venv" ]; then
    export PYTHONPATH="$WS/.venv/lib/python3.12/site-packages:${PYTHONPATH}"
fi

# Source the workspace overlay if it has been built.
if [ -f "$WS/install/setup.bash" ]; then
    source "$WS/install/setup.bash"
fi

# Start interactive shells in the workspace.
cd "$WS" 2>/dev/null || true
