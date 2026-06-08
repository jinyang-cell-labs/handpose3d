# handpose_estimation

3D hand pose estimation from two calibrated cameras.

## Pipeline

1. Subscribe to `<name>/image_raw` and `<name>/camera_info` for two cameras
   (published by `vision_interfaces`), paired by an
   `ApproximateTimeSynchronizer`.
2. Run MediaPipe **HandLandmarker** (Tasks API, VIDEO mode) on each view to get
   21 2D hand landmarks.
3. Triangulate each landmark to 3D with the **Direct Linear Transform**.
   Intrinsics `K` come from the `camera_info` topics; stereo extrinsics
   `R, t` come from `config/extrinsics.yaml`. Projection matrix `P = K [R|t]`.
4. Publish the 3D skeleton as a `visualization_msgs/MarkerArray` on
   `handpose/markers` (joints as spheres, bones as lines) in the `world` frame,
   and optionally the annotated 2D views on `<name>/handpose/annotated`.

## Run

```bash
# requires vision_interfaces (or any source) publishing the camera topics
ros2 launch handpose_estimation handpose_estimation.launch.py          # with RViz
ros2 launch handpose_estimation handpose_estimation.launch.py rviz:=false
```

## Model

MediaPipe's `hand_landmarker.task` bundle is expected at the `model_path`
parameter (default `models/hand_landmarker.task` in this package). Fetch it
with `scripts/download_model.sh` from the repo root if missing.

## Topics

| Direction | Topic | Type |
|-----------|-------|------|
| sub | `<name>/image_raw` | `sensor_msgs/Image` |
| sub | `<name>/camera_info` | `sensor_msgs/CameraInfo` |
| pub | `handpose/markers` | `visualization_msgs/MarkerArray` |
| pub | `<name>/handpose/annotated` | `sensor_msgs/Image` |

## Key parameters (`config/handpose_estimation.yaml`)

- `camera_names` — exactly two, must match the published topic namespaces.
- `extrinsics_file` — YAML with per-camera `rotation`/`translation`.
- `scale` — multiplies triangulated world units into metres for RViz sizing.
- `num_hands`, `min_*_confidence`, `sync_slop`, `publish_annotated`.
