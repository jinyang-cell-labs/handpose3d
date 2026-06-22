# hot3_dataset_interface

Replays a **HOT3D** (Meta Project Aria) clip as ROS 2 camera streams, mirroring
the topic/format contract of `vision_interfaces` / `cityu_data_interface` so any
downstream consumer (e.g. the handpose stack) can subscribe uniformly.

## Published topics

For each published camera `<name>`:

| Topic                  | Type                     | Notes                          |
| ---------------------- | ------------------------ | ------------------------------ |
| `<name>/image_raw`     | `sensor_msgs/Image`      | `mono8` (fisheye) or `bgr8`    |
| `<name>/camera_info`   | `sensor_msgs/CameraInfo` | FISHEYE624 intrinsics          |
| `/tf`                  | `tf2_msgs/TFMessage`     | `world -> <frame_id>` per frame |

All streams on a given tick share one timestamp, so a downstream
`ApproximateTimeSynchronizer` pairs them.

## The three Aria cameras

| Label    | Camera              | Size      | Encoding |
| -------- | ------------------- | --------- | -------- |
| `1201-1` | left SLAM fisheye   | 640×480   | `mono8`  |
| `1201-2` | right SLAM fisheye  | 640×480   | `mono8`  |
| `214-1`  | middle RGB camera   | 1408×1408 | `bgr8`   |

By default they map to `camera0` / `camera1` / `camera2`.

## camera_info / FISHEYE624

The Aria cameras use Meta's **FISHEYE624** model (15 projection params: focal,
cx, cy, 6 radial + 2 tangential + 4 thin-prism). No standard ROS distortion
model captures this, so `camera_info` is published faithfully:

- `K` / `P` carry focal length + principal point,
- `distortion_model = "FISHEYE624"`,
- `D` holds the 12 distortion coefficients (params `[3:]`).

Consumers expecting `plumb_bob`/`equidistant` must be aware of the custom model
string (or rectify the fisheye images first).

## Extrinsics

Each frame's `cameras.json` gives `T_world_from_camera` (the headset moves, so
this changes per frame). It is broadcast as `world -> <frame_id>` on `/tf`.
Disable with `publish_tf: false`.

## Run

```bash
# from /workspace/ros2_ws inside the container
colcon build --packages-select hot3_dataset_interface
source install/setup.bash
ros2 launch hot3_dataset_interface hot3_dataset_interface.launch.py

# override the clip without editing config
ros2 launch hot3_dataset_interface hot3_dataset_interface.launch.py clip:=clip-001849
```

Edit `config/hot3_dataset_interface.yaml` to choose the clip, trim which streams
are published, set the frame rate, or restrict the playback window. The four
per-stream lists (`stream_labels`, `camera_names`, `encodings`, `frame_ids`) are
positionally aligned — trim all four together to publish fewer cameras.

## Not in scope

The per-frame `hands.json`, `objects.json`, and `hand_crops.json` annotation
sidecars are not published by this package (images + camera_info + TF only).
