# mediapie_landmarks_extraction

A basic, standalone MediaPipe hand-landmark extraction node — the simpler cousin
of `handpose_estimation`. It does **not** triangulate or estimate 3D pose.

For each configured image topic it:

1. subscribes to a `sensor_msgs/Image` stream,
2. runs MediaPipe's HandLandmarker to get the 21 2D hand keypoints,
3. draws the landmarks + skeleton onto the frame, and
4. republishes the annotated frame as `sensor_msgs/Image` (bgr8).

Optionally it also publishes:

- (`enable_landmark_msg`) the landmarks as data — `handpose3d_msgs/HandLandmarks`
  with the 2D image landmarks, 3D world landmarks, handedness and confidence.
- (`enable_3d_estimation`) MediaPipe's `hand_world_landmarks` as a
  `visualization_msgs/MarkerArray` for RViz.

One detector is created per input topic so VIDEO-mode timestamps stay
independent across streams.

## Landmark data message (`handpose3d_msgs/HandLandmarks`)

With `enable_landmark_msg: true`, each input topic gets a companion topic
(`<input_topic>` + `landmarks_suffix`, default `/landmarks/hands`) publishing:

```
# HandLandmarks
std_msgs/Header header     # stamp + frame_id from the source image
string source_topic        # which input image topic
Hand[] hands

# Hand
string handedness                      # "Left" / "Right"
float32 score                          # per-hand confidence [0,1]
geometry_msgs/Point[] landmarks_image  # 21 pts: x,y pixels, z relative depth
geometry_msgs/Point[] landmarks_world  # 21 pts: metres, hand-local (empty if N/A)
```

The messages are defined in the separate `handpose3d_msgs` package (an
`ament_cmake` interface package — `.msg` generation isn't possible from this
`ament_python` package). Build `handpose3d_msgs` first. Per-landmark confidence
is not provided by MediaPipe's Tasks API, so `score` is per hand.

## 3D estimation (`hand_world_landmarks`)

Set `enable_3d_estimation: true` to publish the model's metric (metres) 3D hand
landmarks on `markers_3d_topic` (default `/landmarks/markers_3d`).

**No `camera_info` / calibration is required.** These points come straight from
the MediaPipe model and live in a **hand-local frame** (origin ≈ the hand's
geometric center, MediaPipe's own axis convention). So you get the hand's 3D
*shape* in metres, but **not** its absolute position or orientation in any
world/camera frame — that's the difference from `handpose_estimation`, which
uses two views + calibration to place hands in a real `world` frame. Because
every hand is centered at its own origin, each `(camera, hand)` skeleton is
shifted by `camera_spacing` (x) / `hand_spacing` (y) so they don't overlap in
RViz. The node also broadcasts an identity static TF so `world_frame` exists as
the RViz fixed frame.

## Configuration

All behavior is driven by
[`config/mediapie_landmarks_extraction.yaml`](config/mediapie_landmarks_extraction.yaml):

| Parameter | Meaning |
|---|---|
| `image_topics` | list of input image topics |
| `annotated_topics` | optional explicit 1:1 output topics; `[""]` = auto-derive |
| `annotated_suffix` | suffix appended to each input topic when no explicit outputs given |
| `enable_annotation` | publish annotated images (false = detect + log only) |
| `enable_landmark_msg` | publish `handpose3d_msgs/HandLandmarks` (2D+3D data) |
| `landmarks_suffix` | suffix appended to each input topic for the data topic |
| `enable_3d_estimation` | publish `hand_world_landmarks` as an RViz `MarkerArray` |
| `markers_3d_topic` | output topic for the 3D markers |
| `world_frame` | frame for the 3D markers (broadcast as a static TF) |
| `joint_size` / `line_width` | 3D marker sphere diameter / bone thickness (m) |
| `camera_spacing` / `hand_spacing` | layout offsets so hands don't overlap (m) |
| `model_path` | path to `hand_landmarker.task` |
| `num_hands` | max hands to detect |
| `min_hand_detection_confidence` / `min_hand_presence_confidence` / `min_tracking_confidence` | MediaPipe thresholds |
| `running_mode` | `video` (stateful) or `image` (stateless per frame) |
| `line_thickness` / `point_radius` | overlay drawing |
| `enable_logging` | create the start/stop log services + capture calibration |
| `log_dir` | directory the JSONL takes are written to |
| `extrinsics_file` | `T_world_cam` source (calibration_multi_cam) for the meta |
| `log_camera_names` | per-stream camera name; `[""]` = derive from topic namespace |

By default the output topic for `camera0/image_raw` is
`camera0/image_raw/landmarks/annotated`.

## Session logging (service-driven JSONL)

With `enable_logging: true`, the node records the hand landmarks **and** the
active session calibration to a self-contained file, started/stopped on demand
via `std_srvs/Trigger` services:

```bash
ros2 service call /mediapie_landmarks_node/start_log std_srvs/srv/Trigger {}
# ... perform the motion you want to capture ...
ros2 service call /mediapie_landmarks_node/stop_log  std_srvs/srv/Trigger {}
```

Each take is one **JSONL** file (`<log_dir>/handpose_log_<timestamp>.jsonl`):

- **line 1 — `meta`**: schema version, world frame, `joint_names`,
  `landmarks_undistorted` (true iff `enable_undistortion`), and per camera the
  `intrinsics` (`K`, `distortion`, `model`, `resolution`, from each
  `camera_info`) and `T_world_cam` (4×4 cam→world, from `extrinsics_file`).
  Intrinsics/extrinsics are `null` if their source wasn't available when the
  take started (a warning is returned in the service response).
- **lines 2+ — `frame`**: one record per processed image — `camera`,
  `stamp_ns`, and `hands[]` each carrying `handedness`, `score`,
  `landmarks_image` (21×[x_px, y_px, z_rel]) and `landmarks_world` (21×metres,
  hand-local, or `null`). Frames with no hands are still recorded so detection
  gaps are visible in the timeline.

Writes are flushed per line, so a take survives a crash / `Ctrl-C` (the node
also closes the file cleanly on shutdown). Records are per-(camera, frame): the
node detects each stream independently, so re-correlate cameras offline by
`stamp_ns`.

### Loading in Python

[`scripts/load_handpose_log.py`](scripts/load_handpose_log.py) parses a take and
stacks one (camera, handedness) into dense NaN-filled arrays:

```python
from load_handpose_log import load_log, stack_camera

log = load_log("handpose_log_20260626_120000.jsonl")
K = log.meta["cameras"]["camera0"]["intrinsics"]["K"]      # row-major 3x3
T = log.meta["cameras"]["camera0"]["T_world_cam"]          # 4x4 cam->world

s = stack_camera(log, "camera0", "Left")
s["stamp_ns"]  # (T,)        s["image"]  # (T, 21, 3)
s["score"]     # (T,)        s["world"]  # (T, 21, 3)   NaN where the hand was absent
```

Or `python3 scripts/load_handpose_log.py <file.jsonl>` prints a summary.

## Build & run

```bash
# from /workspace/ros2_ws
colcon build --packages-select mediapie_landmarks_extraction
source install/setup.bash
ros2 launch mediapie_landmarks_extraction mediapie_landmarks_extraction.launch.py
```

The model bundle is shipped in `models/hand_landmarker.task`. To re-fetch it:

```bash
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```
