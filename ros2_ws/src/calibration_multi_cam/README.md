# calibration_multi_cam

Multi-camera intrinsic + extrinsic calibration for ROS 2 (Jazzy). Subscribes to
several camera image topics defined in a central YAML, detects an AprilGrid
target, and solves per-camera intrinsics plus the rig extrinsics. **The world
frame is aligned with the first camera** (`camera_names[0]`).

Algorithm follows ethz-asl/kalibr's `kalibr_calibrate_cameras` pipeline,
reimplemented on a modern, pure-Python stack (OpenCV + scipy) — no Boost.Python /
SuiteSparse / catkin. See `../third_party/kalibr` for the reference.

## Output contract (as specified)

- **Intrinsics** → `sensor_msgs/CameraInfo` on `<camera>/camera_info`, carrying
  **K + distortion only**. `R` and `P` are intentionally left empty.
- **Extrinsics** → static **TF** (`world → camera`) and a `geometry_msgs/PoseArray`
  on `~/extrinsics`. The world camera's pose is identity (no self-TF emitted).

## Usage

```bash
# 1. Edit config/calibration.yaml: camera_names, per-camera topics, target dims.
# 2. Collect (move the board through the cameras' shared field of view):
ros2 launch calibration_multi_cam calibrate.launch.py
#    watch the status log for per-camera counts + rig_connected=True
# 3. Solve + persist:
ros2 service call /calibration_collector/calibrate std_srvs/srv/Trigger {}
# 4. Publish the result continuously:
ros2 launch calibration_multi_cam publish.launch.py
```

The collector is source-agnostic — `ros2 bag play` of recorded image topics
works exactly like live cameras (repeatable, offline calibration).

## `result_file` schema (solver → publisher)

```yaml
world_frame: cam0
cameras:
  cam0:
    model: pinhole-radtan
    resolution: [width, height]
    intrinsics: [fx, fy, cx, cy]
    distortion: [k1, k2, p1, p2]      # radtan -> CameraInfo plumb_bob
    T_world_cam:                       # 4x4 pose of camera in world (cam0=identity)
      - [1, 0, 0, 0]
      - [0, 1, 0, 0]
      - [0, 0, 1, 0]
      - [0, 0, 0, 1]
  cam1: { ... }
```

## Status

| Component | State |
|---|---|
| `target.py` — AprilGrid geometry + detection (kalibr layout) | ✅ done, unit-tested |
| `observations.py` — synchronized-view DB | ✅ done, unit-tested |
| `collector_node.py` — subscribe + sync + detect + accumulate | ✅ done |
| `publisher_node.py` — intrinsics-only CameraInfo + TF/Pose | ✅ done |
| `intrinsics.py` — per-camera `cv2.calibrateCamera` / from `camera_info` | ⏳ next |
| `extrinsics.py` — pairwise PnP + covisibility-graph chaining to cam0 | ⏳ next |
| `bundle_adjust.py` — global reprojection BA (scipy `least_squares`, Huber) | ⏳ next |
| `calibrator.py` — orchestrate solve, write `result_file` | ⏳ next |

Design decisions: cameras overlap in **adjacent pairs** (graph chaining, not a
star); intrinsics are **per-camera configurable** (calibrate vs. reuse
`camera_info`); the bundle-adjust backend is **scipy**.
