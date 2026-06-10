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
| pub | `handpose/wrist_left` | `geometry_msgs/PoseStamped` |
| pub | `handpose/wrist_right` | `geometry_msgs/PoseStamped` |
| pub | `<name>/handpose/annotated` | `sensor_msgs/Image` |

## Key parameters (`config/handpose_estimation.yaml`)

- `camera_names` — exactly two, must match the published topic namespaces.
- `extrinsics_file` — YAML with per-camera `rotation`/`translation`.
- `scale` — multiplies triangulated world units into metres for RViz sizing.
- `num_hands`, `min_*_confidence`, `sync_slop`, `publish_annotated`.

## Wrist-pose pipeline

On top of the raw per-joint triangulation, a model-based, temporally coupled
pipeline (design: `docs/estimation_guide_v1.md` at the repo root) estimates a
single smooth **6-DoF wrist pose per hand** and publishes it as
`geometry_msgs/PoseStamped` on `handpose/wrist_left` / `handpose/wrist_right`.
The legacy `MarkerArray` is kept; a fitted-template skeleton is drawn next to
the raw joints (`publish_fitted_skeleton`) for A/B comparison in RViz.

Stages (code: `wrist_pose_pipeline.py` + Stage A1/A2 in `triangulation.py`),
each toggleable via its `*.enabled` parameter:

| Stage | What it does |
|-------|--------------|
| A1 `weighted_dlt` | Per-view confidence-weighted DLT: each camera's two rows of the 2N×4 system are scaled by its handedness score, solved by SVD |
| A2 `reprojection_residual` | Per-joint pixel residual after triangulation; converted to a per-joint trust weight `1/(1+resid/scale)` |
| B1 `reachability_gate` | Rejects joints outside the arm-reach shell `d_min ≤ ‖p − shoulder‖ ≤ d_max` or behind the head |
| C `procrustes` | Weighted Kabsch/orthogonal-Procrustes fit of the canonical hand template to the observed joints, with the `det` reflection fix; the wrist pose is the fitted rigid transform |
| B2 `ransac` | RANSAC wrapper around the fit; occluded/mistracked joints fall out as outliers before the final weighted refit |
| D `kalman` | Constant-velocity Kalman filter on position with chi-square Mahalanobis gating (`11.345` = χ²(3 DoF, 0.99)), SLERP low-pass on orientation, predict-only coasting up to `max_coast_frames` |
| E `one_euro` | One-Euro filter (Casiez et al., CHI 2012) as final position+quaternion polish |

### Units

The pipeline runs in **metres**: triangulated joints are multiplied by
`effective_scale` first (1.0 in stereo mode, `scale` in extrinsics mode), so
every threshold in the config (`ransac.inlier_thresh`, the reachability shell,
Kalman noises) is physical and valid in both triangulation modes. Published
poses match the RViz markers.

### Per-joint confidence

The MediaPipe Tasks-API HandLandmarker never populates per-landmark
visibility/presence (always 0 — MediaPipe issues #5212/#4479). Per-joint
weights are therefore synthesised from the per-hand handedness score and the
Stage-A2 reprojection residual (`weight_source`: `uniform` / `handedness` /
`reprojection` / `product`). Note a 2-camera caveat: residuals only expose
**off-epipolar** disagreement; an error along the epipolar line just shifts
depth and reprojects perfectly — that failure mode is what B1/B2 catch.

### Calibration

1. **Hand template** (`config/hand_template.yaml`): defaults from Buryanov &
   Kotiuk (2010) anthropometry; metacarpal/thumb/palm values are placeholders
   (marked TODO). To personalise: hold the hand flat in clear view of both
   cameras, capture ~50 frames, rigidly align each to the template (Kabsch),
   average in the wrist frame, write back.
2. **Reachability shell** (`shell:` section + `reachability_gate.*` params):
   extend then retract the arm while recording `‖wrist − shoulder‖`; set
   `d_max`/`d_min` ~5 % beyond the observed extremes. Set shoulder anchors
   from the rig geometry (world = left rectified camera optical frame in
   stereo mode: x right, **y down**, z forward).

### Incremental bring-up (recommended order)

1. **A1+A2 only** (disable B1/C/B2/D/E) — raw markers look right; residuals
   spike on visibly bad joints (if mean residual stays >3–5 px on a clearly
   visible hand, fix intrinsics/extrinsics first).
2. **+B1** — out-of-shell poses are dropped, not garbage.
3. **+C** — compare raw vs fitted skeleton in RViz; divergence on a good hand
   means a wrong-sized template.
4. **+B2** — tune `ransac.inlier_thresh` until a finger occlusion no longer
   drags the wrist.
5. **+D** — raise `measurement_noise_pos` for smoothness; verify coasting
   bridges short dropouts and gating rejects teleports.
6. **+E** — hand still: lower `min_cutoff` until jitter is acceptable; then
   move fast and raise `beta` until lag is acceptable.

### Tests

The math stages are ROS-free and unit-tested:

```bash
python3 -m pytest test/test_wrist_pose_pipeline.py
```
