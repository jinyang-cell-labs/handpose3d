# multi_cam_stream

Minimal multi-USB-camera frontend for ROS 2. Opens one OpenCV `VideoCapture`
per V4L device and publishes a `sensor_msgs/Image` (bgr8) per camera:

```
<name>/image_raw     sensor_msgs/Image   (bgr8)
```

No `camera_info`, no video-file replay — just the live USB streams. All cameras
are grabbed on one timer tick and share a single timestamp, so a downstream
`ApproximateTimeSynchronizer` (e.g. `calibration_multi_cam`) pairs them cleanly.

## Usage

```bash
# find your devices first
v4l2-ctl --list-devices        # or: ls /dev/video*

# edit config/multi_cam_stream.yaml: camera_names, camera_devices, resolution
ros2 launch multi_cam_stream multi_cam_stream.launch.py
```

RViz2 opens by default with one Image display per camera so you can see all
three streams as soon as the node is up. Disable it (e.g. headless) with:

```bash
ros2 launch multi_cam_stream multi_cam_stream.launch.py rviz:=false
```

Default topics are `/cam0/image_raw`, `/cam1/image_raw` — matching
`calibration_multi_cam`'s default `camera_names`, so the two packages compose
directly:

```bash
ros2 launch multi_cam_stream multi_cam_stream.launch.py        # publish USB cams
ros2 launch calibration_multi_cam calibrate.launch.py          # collect + calibrate
```

## Notes

- `fourcc: "MJPG"` (default) is usually required for full resolution/fps on UVC
  webcams; the node logs the actual mode and warns on a resolution fallback.
- Several high-res MJPG streams can saturate a single USB bus/hub — if frames
  stall, lower `capture_width`/`capture_height` or spread cameras across buses.
