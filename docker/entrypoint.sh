#!/bin/bash
set -e

cd /workspace/ros2_ws

# Keep the uv-managed virtualenv in sync with pyproject.toml / uv.lock.
export PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"
echo "Syncing virtual environment..."
uv sync --no-dev

# Source ROS 2.
. /opt/ros/jazzy/setup.bash

# Source the workspace overlay if it has been built.
if [ -f "/workspace/ros2_ws/install/setup.bash" ]; then
    . /workspace/ros2_ws/install/setup.bash
fi

# Make the uv venv packages (mediapipe, opencv, ...) importable from the
# system interpreter that runs the ROS2 entry-point scripts.
if [ -d "/workspace/ros2_ws/.venv" ]; then
    export PYTHONPATH="/workspace/ros2_ws/.venv/lib/python3.12/site-packages:${PYTHONPATH}"
fi

exec "$@"
