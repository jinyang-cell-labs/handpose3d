# handpose_depth_estimation

Per-joint 3D hand-pose estimation by triangulation of **two selected cameras**.

The rig publishes three cameras; this node lets you pick exactly two
(`camera_names`, default `camera0` + `camera2`) and triangulates **all 21
MediaPipe hand landmarks independently** to recover the true metric 3D hand —
unlike `stereo_handpose_estimation`, which triangulates one robust hand centroid
and hangs MediaPipe's hand-local shape off it.

## Data flow

```
multi_cam_stream            ──>  <cam>/image_raw
calibration_multi_cam       ──>  <cam>/camera_info        (K + distortion, NO R/P)
                            ──>  TF world->cam + extrinsics.yaml (T_world_cam)
mediapie_landmarks_extraction ─> <cam>/image_raw/landmarks/hands  (HandLandmarks)

handpose_depth_estimation:
  for each hand seen in BOTH chosen views (matched by handedness):
    triangulate 21 joints with DLT (P = K[R|t], world->cam from extrinsics)
    -> handpose_depth/markers          (RViz skeleton)
    -> handpose_depth/joints_{left,right}   (PoseArray, 21 joints, world frame)
    -> <cam>/image_raw/depth/reprojected    (3D reprojected back onto the image)
```

Because `camera_info` here is **intrinsics-only** (R and P are zero), the
stereo-rectified `camera_info` path is not available — triangulation uses the
extrinsics + raw K. The output/world frame is the extrinsics file's
`world_frame` (the first camera, `camera0`), matching the TF tree published by
`calibration_multi_cam`.

## Distortion / undistortion

`landmarks_undistorted` (default **true**) assumes the upstream
`mediapie_landmarks_extraction` node ran with `enable_undistortion=true`, so the
2D landmarks are already in the undistorted pinhole image:

- triangulation feeds the landmarks straight to DLT;
- reprojection uses the pinhole `P` (no distortion);
- `image_raw` is **undistorted here** before the overlay, so the detection and
  the reprojected skeleton share the same coordinates.

Set it `false` if the landmarks come from raw/distorted frames: each 2D point is
then undistorted with K/D before DLT, and the 3D joints are reprojected with the
full distortion model (`cv2.projectPoints`) onto the raw image.

## Reprojection QA overlay

For each selected camera the node finds the buffered `image_raw` frame matching
the landmark timestamp, draws:

- the **reprojected** 3D skeleton (solid, blue=Left / orange=Right),
- the upstream **2D detection** (green hollow dots, `draw_detected`),
- the **mean per-joint reprojection error** in pixels (text, top-left),

and republishes it on `<cam>/image_raw/depth/reprojected`. Small, tight overlap
= good calibration + triangulation.

## Run

```bash
# upstream (separate terminals / launch files):
#   multi_cam_stream, calibration_multi_cam publish.launch.py,
#   mediapie_landmarks_extraction (enable_undistortion=true, enable_landmark_msg=true)

ros2 launch handpose_depth_estimation handpose_depth_estimation.launch.py
# without RViz:
ros2 launch handpose_depth_estimation handpose_depth_estimation.launch.py rviz:=false
```

Pick a different pair by editing `config/handpose_depth_estimation.yaml`
(`camera_names`), e.g. `["camera0", "camera1"]`.
