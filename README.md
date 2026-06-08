# handpose3d (ROS 2 Jazzy)

Real-time **3D hand pose estimation** from two calibrated cameras, refactored
into a containerized ROS 2 Jazzy stack. MediaPipe detects 2D hand landmarks in
each view; the 21 keypoints are triangulated to 3D and visualized in RViz.

This is a ROS 2 port of [TemugeB/handpose3d](https://github.com/TemugeB/handpose3d).
The original standalone scripts are preserved under [legacy/](legacy/).

## Architecture

The functionality is decoupled into two ROS 2 packages:

```
 camera / video files                      RViz (MarkerArray + annotated views)
        │                                          ▲
        ▼                                          │
┌──────────────────────┐   image_raw   ┌────────────────────────┐
│  vision_interfaces   │ ────────────► │   handpose_estimation  │
│  camera_publisher    │  camera_info  │  MediaPipe + DLT → 3D  │
└──────────────────────┘ ────────────► └────────────────────────┘
```

- **[vision_interfaces](ros2_ws/src/vision_interfaces/)** — opens a live camera
  *or* replays video files and publishes `<name>/image_raw` +
  `<name>/camera_info` per camera (same topic contract as the `camera_s3`
  driver).
- **[handpose_estimation](ros2_ws/src/handpose_estimation/)** — subscribes to
  two camera streams, runs MediaPipe HandLandmarker, triangulates to 3D (camera
  intrinsics from `camera_info`, stereo extrinsics from config) and publishes a
  skeleton `MarkerArray` for RViz.

## Quick start (Docker)

```bash
# 1. Fetch the MediaPipe model bundle (skip if the .task file already exists)
./scripts/download_model.sh

# 2. Allow the container to reach your X server (for RViz)
xhost +local:docker

# 3. Start the container (it just stays up — no auto build/launch)
cd docker
docker compose up -d --build handpose3d

# 4. Exec in and drive it by hand
docker compose exec handpose3d bash
```

Inside the container the repo is mounted at `/workspace` and these build
aliases are available:

| alias | expands to |
|-------|------------|
| `cb`  | `colcon build` |
| `cbs` | `colcon build --packages-select` |
| `cbc` | `colcon build --cmake-clean-cache` |

```bash
# inside the container (starts in /workspace/ros2_ws)
cb                                   # or: cbs vision_interfaces handpose_estimation
source install/setup.bash            # alias: sw
ros2 launch vision_interfaces vision_interfaces.launch.py &
ros2 launch handpose_estimation handpose_estimation.launch.py
```

By default this replays the bundled sample clips in [media/](media/). To use
live cameras, set `source_type: "camera"` (and `camera_devices`) in
[vision_interfaces.yaml](ros2_ws/src/vision_interfaces/config/vision_interfaces.yaml).

## Calibration

- **Intrinsics** live in
  [vision_interfaces/config/camera_info/](ros2_ws/src/vision_interfaces/config/camera_info/)
  (standard `camera_info_manager` YAML) and are published on the `camera_info`
  topics.
- **Stereo extrinsics** (per-camera world→camera `R, t`) live in
  [handpose_estimation/config/extrinsics.yaml](ros2_ws/src/handpose_estimation/config/extrinsics.yaml).

Both were ported from the original `camera_parameters/` calibration (now under
[legacy/camera_parameters/](legacy/camera_parameters/)). Replace them with your
own calibration to use different cameras.

## Repository layout

```
docker/                 Dockerfile, docker-compose.yaml, entrypoint.sh
ros2_ws/
  pyproject.toml        uv-managed Python deps (mediapipe, opencv, ...)
  src/
    vision_interfaces/      camera / video → image_raw + camera_info
    handpose_estimation/    MediaPipe + DLT → 3D MarkerArray (+ RViz)
scripts/download_model.sh   fetch hand_landmarker.task
media/                  sample stereo video clips
legacy/                 original standalone handpose3d scripts
```
