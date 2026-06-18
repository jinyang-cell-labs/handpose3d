# mediapipe_estimation

Quick-look tool for judging how well MediaPipe's HandLandmarker extracts the
hand pose from a given camera or recording. No calibration, no triangulation —
it plays one video file, runs the landmarker on every frame, draws the
21-landmark skeleton and republishes the annotated frames for RViz.

## Run

```bash
# inside the container (repo bind-mounted at /workspace)
cd /workspace/ros2_ws
cbs mediapipe_estimation && sw          # colcon build + source overlay
ros2 launch mediapipe_estimation mediapipe_estimation.launch.py
# headless (no RViz), e.g. just to check the topic:
ros2 launch mediapipe_estimation mediapipe_estimation.launch.py rviz:=false
```

The annotated stream is published on `/mediapipe/annotated`
(`sensor_msgs/Image`, `bgr8`) and shown in the RViz **Annotated** image panel.

## Testing a different camera / recording

Everything is config-driven — edit
[config/mediapipe_estimation.yaml](config/mediapipe_estimation.yaml) and
relaunch:

- `video_path` — recording to play (container path; `recordings/` is gitignored)
- `frame_width` / `frame_height` + `resize_to_config` — set the source
  resolution and enable resizing to normalise across cameras (e.g. 1920x1080)
- `fps` — playback rate (`0.0` = use the video's own FPS)
- `loop` — replay the clip when it ends
- `num_hands`, `min_hand_*_confidence` — detector knobs
- `annotated_topic`, `frame_id` — output topic / image frame

The MediaPipe model bundle is reused from `handpose_estimation/models/`
(fetched by `scripts/download_model.sh`).
