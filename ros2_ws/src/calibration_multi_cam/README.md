# calibration_multi_cam

Multi-camera intrinsic + extrinsic calibration for ROS 2 (Jazzy). Subscribes to
several camera image topics defined in a central YAML, detects an AprilGrid
target, and solves per-camera intrinsics plus the rig extrinsics. **The world
frame is aligned with the first camera** (`camera_names[0]`).

Algorithm follows ethz-asl/kalibr's `kalibr_calibrate_cameras` pipeline,
reimplemented on a modern, pure-Python stack (OpenCV + scipy) — no Boost.Python /
SuiteSparse / catkin. See `../third_party/kalibr` for the reference.

## Two-stage workflow

Intrinsics are a fixed lens property (calibrate once, reuse); extrinsics change
whenever a camera moves. They also need different board motions. So they are
split into two stages writing two files, which the publisher then serves.

```bash
# Stage 1 - intrinsics: fill each camera's frame with the board (per camera)
ros2 launch calibration_multi_cam intrinsic.launch.py
ros2 service call /calibration_intrinsic/calibrate std_srvs/srv/Trigger {}     # -> intrinsics.yaml

# Stage 2 - extrinsics: move the board across overlapping views (rig_connected=True)
ros2 launch calibration_multi_cam extrinsic.launch.py
ros2 service call /calibration_extrinsic/calibrate std_srvs/srv/Trigger {}     # loads intrinsics.yaml -> extrinsics.yaml

# Publish: intrinsics-only CameraInfo + extrinsics as TF/Pose
ros2 launch calibration_multi_cam publish.launch.py
```

Collection is source-agnostic — `ros2 bag play` of recorded image topics works
exactly like live cameras. Pairs directly with the `multi_cam_stream` package.

### Bounded, diverse collection (keep-most-informative)

Only the extracted corner data is kept (corner ids + subpixel pixel coords),
never raw images — a view is ~2 KB. Collection is bounded: `max_views_per_camera`
(intrinsic) and `max_views` (extrinsic) cap the retained set (`0` = unlimited).
When a buffer is full, an incoming view evicts the **most redundant** stored
view — the one in the closest pair in an appearance-feature space (board
position / scale / tilt) — so the kept set stays maximally varied rather than
filling up with near-duplicates. This is a lightweight stand-in for kalibr's
information-gain view selection, and it also bounds the bundle-adjustment cost.

## Output contract (as specified)

- **Intrinsics** → `sensor_msgs/CameraInfo` on `<camera>/camera_info`, carrying
  **K + distortion only**. `R` and `P` are intentionally left empty.
- **Extrinsics** → static **TF** (`world → camera`) and a `geometry_msgs/PoseArray`
  on `~/extrinsics`. The world camera's pose is identity (no self-TF emitted).

## File schemas

```yaml
# intrinsics.yaml  (stage 1 -> stage 2 + publisher)
cameras:
  camera0: {model: pinhole-radtan, resolution: [w,h],
         intrinsics: [fx,fy,cx,cy], distortion: [k1,k2,p1,p2], reproj_rms: 0.21, num_views: 28}
```
```yaml
# extrinsics.yaml  (stage 2 -> publisher)
world_frame: camera0
cameras:
  camera0: {T_world_cam: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}   # identity
  camera1: {T_world_cam: [[...]]}                                      # pose of camera1 in world
```

## Modules

| File | Role |
|---|---|
| `target.py` | AprilGrid geometry + detection (kalibr layout) |
| `view_buffer.py` | keep-most-informative retention (maximin diversity thinning) |
| `observations.py` | synchronized-view database (bounded, diverse) |
| `se3.py` | SE(3) helpers (Rodrigues, compose, robust average) |
| `intrinsics.py` | per-camera `cv2.calibrateCamera` (4-param radtan) |
| `extrinsics.py` | per-view PnP + pairwise relative pose + spanning-tree chaining to camera0 |
| `bundle_adjust.py` | global reprojection BA (scipy `least_squares`, Huber, intrinsics fixed) |
| `intrinsic_calibrator_node.py` | stage 1 node |
| `extrinsic_calibrator_node.py` | stage 2 node |
| `publisher_node.py` | loads both files; CameraInfo + TF/Pose |

The solver is validated end-to-end on synthetic data (known rig → project →
recover): intrinsics to <0.1%, extrinsics to <0.02 mm / 0.002° after BA.

## Validate when first running on real cameras

- `cv2.aruco` must be available (Ubuntu 24.04 `python3-opencv` includes it).
- The AprilTag corner permutation `_ARUCO_TO_KALIBR` in `target.py` is the first
  thing to re-check if reprojection RMS is large/structured.
