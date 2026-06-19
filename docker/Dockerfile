FROM osrf/ros:jazzy-desktop-full
SHELL ["/bin/bash", "-c"]

# Base tools + ROS deps. desktop-full already provides rviz2 and the common
# message packages; we add camera_info_manager_py, ffmpeg and the GL/X11 libs
# MediaPipe + OpenCV need at runtime. realsense2-camera pulls in the Intel
# RealSense ROS 2 wrapper + the librealsense2 SDK from the OSRF apt repo.
# `apt upgrade` first: the base image is pinned to an old digest, so its ROS
# libs (e.g. diagnostic_updater) lag the Jazzy sync that the latest
# realsense2-camera binary was built against. Without this, the wrapper fails
# at load with an undefined-symbol error from the ABI skew.
RUN apt update && apt upgrade -y && apt install -y \
    python3-pip \
    locales \
    curl \
    wget \
    python3-colcon-clean \
    ros-jazzy-camera-info-manager-py \
    ros-jazzy-realsense2-camera \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv for fast Python dependency management. Pin the install dir to a location
# already on PATH so we don't depend on where the installer's default lands.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
ENV UV_LINK_MODE=copy

WORKDIR /workspace/ros2_ws

# Pre-resolve the Python environment so `docker compose up` is fast. The repo
# is bind-mounted over /workspace at runtime, so the actual build/launch
# happens against the live source tree (see docker-compose.yaml command).
COPY ros2_ws/pyproject.toml ros2_ws/uv.lock* ./
RUN uv sync --no-dev || uv sync --no-dev

ENV LD_LIBRARY_PATH="/usr/local/lib"
ENV RCUTILS_COLORIZED_OUTPUT=1

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Build aliases (cb / cbs / cbc) + ROS/workspace sourcing for interactive shells.
COPY docker/bashrc_setup.sh /root/.bashrc_setup.sh
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc && \
    echo "source /root/.bashrc_setup.sh" >> /root/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
