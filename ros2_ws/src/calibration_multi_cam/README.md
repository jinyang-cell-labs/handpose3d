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

## Board pose tracker (single camera)

A standalone visualization/debugging aid: pick **one** camera, detect the board
live, and recover its pose from that camera's already-calibrated intrinsics. It
broadcasts the board over TF and draws the axes back onto the image so the
detection can be eye-aligned in RViz.

```bash
# needs intrinsics.yaml (stage 1) for the selected camera
ros2 launch calibration_multi_cam board_pose.launch.py                 # camera_names[0]
ros2 launch calibration_multi_cam board_pose.launch.py camera:=camera1 # any camera
```

- **TF** — dynamic `<camera> → board_pose.board_frame` (`calib_board`),
  `T_cam_target`, stamped with the image time. The node also latches the static
  `world → camera_i` rig from `extrinsics_file`, so the board is reachable from
  the world frame **whichever camera you track** (without it, RViz pinned to
  `camera0` only shows the board when tracking `camera0` — every other camera's
  frame is disconnected from the fixed frame). Falls back to fixed-frame =
  tracked camera if extrinsics aren't calibrated yet.
- **Annotated image** — `/<camera>/board_pose/image_axes`: the input frame with
  detected corners (green dots) and the board axes (`drawFrameAxes`) drawn on.
- **Save service** — `/calibration_board_pose/save_board_pose`
  (`std_srvs/srv/Trigger`): latches the most recent valid `T_cam_board` to
  `board_pose_file` (default `config/board_pose.yaml`). Point the board at the
  tracked camera until status shows `pose=OK`, then call the service (a button
  exists in `gui_service_call`).

Reuses the same `calibration.yaml`; the `board_pose.*` keys select the camera,
TF child frame, axis length, and output image topic. The launch templates the
bundled `board_pose.rviz` so the Image panel follows the `camera:=` argument.

### operator_body & the full TF chain

The board is mounted at a known, measured offset from the operator's body, so
`calibration.yaml` defines `operator_body.{position,rotation,frame}` as the
rigid TF `<board_frame> → operator_body` (rotation is intrinsic-XYZ euler in
degrees). Once `board_pose.yaml` is saved, `publish.launch.py` emits both the
saved board TF (`<camera> → <board_frame>`) and the operator_body TF over
`/tf_static` alongside the camera rig, completing the chain

```
world (camera0) → <camera> → <board_frame> → operator_body
```

so RViz can visualize the whole tree and hand poses can be expressed in the
operator_body frame. The board TF is skipped (with a warning) until
`board_pose.yaml` exists.

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
| `board_pose_node.py` | single-camera board pose: PnP → TF + axes drawn on image |

The solver is validated end-to-end on synthetic data (known rig → project →
recover): intrinsics to <0.1%, extrinsics to <0.02 mm / 0.002° after BA.

## Validate when first running on real cameras

- `cv2.aruco` must be available (Ubuntu 24.04 `python3-opencv` includes it).
- The AprilTag corner permutation `_ARUCO_TO_KALIBR` in `target.py` is the first
  thing to re-check if reprojection RMS is large/structured.
