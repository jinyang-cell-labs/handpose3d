# mediapie_landmarks_extraction

A basic, standalone MediaPipe hand-landmark extraction node — the simpler cousin
of `handpose_estimation`. It does **not** triangulate or estimate 3D pose.

For each configured image topic it:

1. subscribes to a `sensor_msgs/Image` stream,
2. runs MediaPipe's HandLandmarker to get the 21 2D hand keypoints,
3. draws the landmarks + skeleton onto the frame, and
4. republishes the annotated frame as `sensor_msgs/Image` (bgr8).

One detector is created per input topic so VIDEO-mode timestamps stay
independent across streams.

## Configuration

All behavior is driven by
[`config/mediapie_landmarks_extraction.yaml`](config/mediapie_landmarks_extraction.yaml):

| Parameter | Meaning |
|---|---|
| `image_topics` | list of input image topics |
| `annotated_topics` | optional explicit 1:1 output topics; `[""]` = auto-derive |
| `annotated_suffix` | suffix appended to each input topic when no explicit outputs given |
| `enable_annotation` | publish annotated images (false = detect + log only) |
| `model_path` | path to `hand_landmarker.task` |
| `num_hands` | max hands to detect |
| `min_hand_detection_confidence` / `min_hand_presence_confidence` / `min_tracking_confidence` | MediaPipe thresholds |
| `running_mode` | `video` (stateful) or `image` (stateless per frame) |
| `line_thickness` / `point_radius` | overlay drawing |

By default the output topic for `camera0/image_raw` is
`camera0/image_raw/landmarks/annotated`.

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
