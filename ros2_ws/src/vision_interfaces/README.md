# vision_interfaces

Camera frontend for the handpose3d stack. Opens one OpenCV `VideoCapture` per
camera — backed by a **live device** or a **recorded video file** — and
publishes, for each camera:

| Topic | Type |
|-------|------|
| `<name>/image_raw` | `sensor_msgs/Image` (`bgr8`) |
| `<name>/camera_info` | `sensor_msgs/CameraInfo` |

All cameras are sampled on one timer tick and stamped with the same timestamp,
so a downstream `ApproximateTimeSynchronizer` pairs the frames cleanly.

## Run

```bash
ros2 launch vision_interfaces vision_interfaces.launch.py
```

## Key parameters (`config/vision_interfaces.yaml`)

- `source_type` — `"video"` (replay files) or `"camera"` (live V4L devices).
- `video_paths` / `camera_devices` — per-camera sources.
- `camera_names` — topic namespaces (e.g. `camera0`, `camera1`).
- `camera_info_urls` — `package://` URLs to per-camera calibration YAML.
- `crop_square` / `output_size` — centre-crop + resize so the published frame
  matches the calibrated intrinsics (the original handpose3d calibration was
  done on a 720×720 square crop).
- `frame_rate`, `loop`.

Calibration files live in `config/camera_info/` and follow the standard
`camera_info_manager` YAML format.
