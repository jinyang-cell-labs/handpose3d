# stereo_handpose_estimation

Recovers an accurate **3D world hand pose** by combining stereo triangulation
with MediaPipe's metric hand shape. It consumes the landmark messages published
by `mediapie_landmarks_extraction` — it does **not** run MediaPipe itself.

## Why this works

MediaPipe's `hand_world_landmarks` give the hand's **shape** in metres but in a
hand-local frame with **no absolute position** (and a single view can't give
true world placement). Two calibrated views can. So:

1. **Centroid feature** — for each hand, take the centroid of the 21 2D image
   landmarks (mean x, y) in each camera. One well-averaged point is far more
   robust to triangulate than 21 noisy per-joint correspondences, and the
   cross-view match is trivial (same handedness label).
2. **Triangulate** the two centroids → the hand's **3D position** in the world
   frame (DLT; stereo-from-`camera_info` or `extrinsics.yaml` + raw `K`).
3. **Place the shape** at that position:

   ```
   final_landmark[i] = centroid_world + R_(world<-cam) @ hand_world[i]
   ```

   `hand_world_landmarks` are offsets in the source camera's optical frame, so
   they're rotated into the world frame by that camera's camera→world rotation.
   With identity extrinsics this reduces to `centroid + hand_world`
   (toggle: `apply_camera_rotation`).

## Topics

**Subscribes**

| Topic | Type |
|---|---|
| `landmark_topics` (×2) | `handpose3d_msgs/HandLandmarks` |
| `camera_info_topics` (×2) | `sensor_msgs/CameraInfo` |

**Publishes**

| Topic | Type | Contents |
|---|---|---|
| `stereo_handpose/markers` | `visualization_msgs/MarkerArray` | per hand: placed 21-joint skeleton + bones + centroid sphere |
| `stereo_handpose/hand_left` / `_right` | `geometry_msgs/PoseWithCovarianceStamped` | triangulated centroid position + 3×3 position covariance (world frame) |
| `stereo_handpose/cameras` + TF | `MarkerArray` / static TF | camera frustums + `world→camera` transforms |

### Position covariance

Each hand pose carries a 3×3 position covariance (top-left of the 6×6 block).
It is the linearized propagation of pixel noise into the 3D point,

```
Cov = sigma_px^2 * (J^T J)^-1
```

where `J` is the stacked reprojection Jacobian of both cameras at the
triangulated point (`J^T J` is the reprojection cost's curvature / Fisher
information). `sigma_px` is `centroid_pixel_sigma`. For a stereo pair the
covariance ellipsoid is a cigar along the viewing ray, so **depth (Z) is the
least-certain axis**, scaling as `~Z² / (focal·baseline)`. Orientation is not
estimated and is flagged with a large variance. Per-hand `sigma(x,y,z)` is also
logged (mm, throttled).

## Configuration

See [`config/stereo_handpose_estimation.yaml`](config/stereo_handpose_estimation.yaml).
Key parameters: `use_camera_info_extrinsics` (stereo-from-`camera_info` vs
`extrinsics.yaml`), `apply_camera_rotation`, `world_landmark_sign` (per-axis flip
if the skeleton renders mirrored), `scale` (calibration units → metres in
extrinsics mode), `min_score`.

## Build & run

```bash
# from /workspace/ros2_ws
colcon build --packages-select handpose3d_msgs mediapie_landmarks_extraction stereo_handpose_estimation
source install/setup.bash

# 1) extract landmarks (with the data message enabled)
ros2 launch mediapie_landmarks_extraction mediapie_landmarks_extraction.launch.py rviz:=false
# 2) stereo-place them in 3D
ros2 launch stereo_handpose_estimation stereo_handpose_estimation.launch.py
```

## Notes / caveats

- The 2D centroid of projected landmarks is only *approximately* the projection
  of the 3D centroid (perspective bias), but it is intentionally robust — the
  whole point is to avoid fragile per-joint stereo matching.
- The placed shape's metric scale comes from MediaPipe (a learned average-hand
  prior); the **position** is true stereo. So absolute placement is accurate;
  the hand's absolute *size* is approximate.
- `extrinsics.yaml` is copied from `handpose_estimation` and must match your
  actual rig for the extrinsics path to be correct.
