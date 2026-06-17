# cityu_data_interface

Replays the **CityU stereo hand pose benchmark** (Zhang et al., *A hand pose
tracking benchmark from stereo matching*, ICIP 2017) as ROS 2 camera streams,
using the exact topic/format contract of [`vision_interfaces`](../vision_interfaces),
so the rest of the handpose stack runs unchanged on recorded dataset frames.

## Published topics

For each configured stream (default `camera0`, `camera1`):

| Topic | Type | Notes |
|-------|------|-------|
| `<name>/image_raw` | `sensor_msgs/Image` | `bgr8`, 640×480 |
| `<name>/camera_info` | `sensor_msgs/CameraInfo` | from `config/camera_info/*.yaml` |

All streams are published on one timer tick with a shared timestamp, so a
downstream `ApproximateTimeSynchronizer` pairs them.

## Dataset layout

Each sequence folder (e.g. `B1Counting`) holds **6000 PNGs** — 1500 frames ×
4 streams, prefixed `BB_left_`, `BB_right_` (Bumblebee2 rectified stereo pair)
and `SK_color_`, `SK_depth_` (RealSense F200). By default this package
publishes the Bumblebee stereo pair as `camera0`/`camera1`.

**Memory:** frames are *never* preloaded. Each tick reads only the current
frame of each published stream with `cv2.imread`, so RAM stays flat (~1 image
per stream) regardless of sequence length.

## Calibration

Intrinsics come from the dataset `readme.txt`: `fx=fy=822.79041`,
`cx=318.47345`, `cy=250.31296`. The images are already rectified, so distortion
is zero. The right camera's projection matrix encodes the stereo baseline as
`Tx = -fx * B` with **B in metres** (0.120054 m) — ROS `CameraInfo` P is metric
and `handpose_estimation` reads the baseline as `|P[0,3]| / fx` metres
(`effective_scale = 1.0`). So 3D points triangulated from these matrices come out
in **metres**. The dataset's `handPara` ground-truth labels are in **millimetres**
(see `../../data_set/.../load_labels.py`), so scale labels by `1e-3` to compare.

## Configuration

Edit [`config/cityu_data_interface.yaml`](config/cityu_data_interface.yaml).
The key knob is `sequence` — the folder to replay:

```yaml
cityu_data_publisher_node:
  ros__parameters:
    dataset_root: "/workspace/ros2_ws/data_set/stereo hand pose data set"
    sequence: "B1Counting"          # which folder to load + launch
    image_prefixes: ["BB_left", "BB_right"]
    camera_names: ["camera0", "camera1"]
    frame_rate: 30.0
    loop: true
    start_frame: 0
    num_frames: -1                  # -1 = whole 1500-frame sequence
```

To replay the RealSense color stream instead, set
`image_prefixes: ["SK_color"]`, `camera_names: ["camera0"]`, and a matching
single `camera_info_urls` entry.

## Run

```bash
# build (from the colcon workspace so artifacts land in ros2_ws/{build,install,log})
cd /workspace/ros2_ws && colcon build --packages-select cityu_data_interface   # or: cbs cityu_data_interface
source install/setup.bash                                                      # or: sw

# launch (sequence taken from the config yaml)
ros2 launch cityu_data_interface cityu_data_interface.launch.py

# or override the sequence without editing the config
ros2 launch cityu_data_interface cityu_data_interface.launch.py sequence:=B3Random
```
