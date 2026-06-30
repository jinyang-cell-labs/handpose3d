# single_cam_pose_estimation

**Monocular** model-based 6-DoF hand-pose estimation — one camera per estimate,
any number of cameras.

Where `handpose_depth_estimation` triangulates the 21 joints from **two**
cameras, this node recovers the hand from a **single** view: it fits the rigid
6-DoF transform `T_world_hand` that places MediaPipe's hand-local 3D model
(`hand_world_landmarks`) so it reprojects onto the detected 2D landmarks
(PnP / reprojection minimisation, with a cheirality penalty to break the
front/back mirror). The maths lives in [`pose_estimation.py`](single_cam_pose_estimation/pose_estimation.py)
— a copy of `tools/statistics_evaluation/pose_estimation.py`, kept local so the
package has no cross-tree import.

## Many camera sources

`camera_names` is a **list of any length**. Each camera is estimated
**independently** — its own landmarks, its own `camera_info`, its own pose —
and every result is expressed in the shared world frame. Running several at once
lets you eyeball how well the independent monocular estimates agree, which is a
calibration sanity check (each camera draws its skeleton in its own colour).

## Data flow

```
multi_cam_stream              ──>  <cam>/image_raw
calibration_multi_cam         ──>  <cam>/camera_info   (K + distortion, NO R/P)
                              ──>  TF world->cam + extrinsics.yaml (T_world_cam)
mediapie_landmarks_extraction ──>  <cam>/image_raw/landmarks/hands  (HandLandmarks
                                   with landmarks_image AND landmarks_world)

single_cam_pose_estimation, for each camera, each hand:
  PnP fit T_world_hand on (image landmarks, hand-local world model)
  place the 21-joint model into the world
    -> single_cam_pose/markers                       (RViz skeleton, 1 colour/cam)
    -> single_cam_pose/<cam>/joints_{left,right}      (PoseArray, 21 joints, world)
    -> single_cam_pose/<cam>/hand_pose_{left,right}   (PoseStamped, 6-DoF pose)
    -> TF world -> <cam>_hand_{Left,Right}            (hand-frame axes in RViz)
    -> <cam>/image_raw/pose/reprojected               (reprojection QA overlay)
```

The upstream `landmarks_world` field **must** be populated — i.e.
`mediapie_landmarks_extraction` running with `enable_landmark_msg: true` (it is
by default; the model returns the world landmarks). Hands missing the world
model are skipped.

## Algorithm (per camera, per hand)

1. `estimate_hand_pose(K, T_world_cam, landmarks_image, landmarks_world)` solves
   for `T_world_hand` (6 DoF) by minimising `sum_i ||uv_i - uv_detected_i||^2`
   plus a one-sided **cheirality penalty** `w * relu(margin - z_cam)` that keeps
   every joint in front of the lens (the pure reprojection cost is blind to the
   front/back flip). Seed: a constant pose `SEED_DEPTH_M` in front of the camera.
2. Place the model: `X_world = X_hand @ R^T + t` → 21 world joints.
3. Publish markers / PoseArray / PoseStamped / TF, and reproject for QA.

Because every camera's pose lands in the world frame, the independent estimates
are directly comparable — agreement across cameras checks the calibration.

## Distortion / undistortion

`landmarks_undistorted` (default **true**) assumes the upstream landmarks are in
the undistorted pinhole image. The PnP cost is pinhole regardless; this flag only
controls the reprojection overlay: when true the joints reproject with the
pinhole `P` and `image_raw` is undistorted first so detection and reprojection
share coordinates; when false the joints reproject with the full distortion
model (`cv2.projectPoints`) onto the raw image.

## Reprojection QA overlay

Per camera the node finds the buffered `image_raw` frame matching the landmark
timestamp and draws the **reprojected** posed skeleton (camera colour) over the
upstream **2D detection** (green hollow dots), with the **mean per-joint
reprojection error** (px) in the corner, on `<cam>/image_raw/pose/reprojected`.

## Run

```bash
# upstream (separate terminals / launch files):
#   multi_cam_stream, calibration_multi_cam publish.launch.py,
#   mediapie_landmarks_extraction (enable_undistortion=true, enable_landmark_msg=true)

# build from the workspace root (NOT the repo root):
cd /workspace/ros2_ws && colcon build --packages-select single_cam_pose_estimation
source install/setup.bash

ros2 launch single_cam_pose_estimation single_cam_pose_estimation.launch.py
# without RViz:
ros2 launch single_cam_pose_estimation single_cam_pose_estimation.launch.py rviz:=false
```

Change which cameras are used by editing `config/single_cam_pose_estimation.yaml`
(`camera_names`) — one, two, or all of the rig's cameras.
