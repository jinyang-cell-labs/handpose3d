# hot3_evaluation

Evaluates **stereo depth from the HOT3D (Project Aria) SLAM fisheye pair** so you
can *see how good it is* in RViz. Consumes the streams published by
`hot3_dataset_interface`, undistorts + rectifies the two FISHEYE624 mono views,
runs `StereoSGBM`, and publishes a depth image.

## Pipeline

1. **Extrinsics from `/tf`** — the two SLAM cameras are *not* a rectified rig
   (they sit ~75° apart), so `camera_info` carries no stereo baseline. The
   inter-camera pose is looked up from `/tf` (broadcast by
   `hot3_dataset_interface`) and fed to `cv2.fisheye.stereoRectify`.
2. **FISHEYE624 → OpenCV fisheye** — FISHEYE624 uses the same *equidistant*
   radial basis as OpenCV's fisheye model, so the first four radial coefficients
   from `camera_info.D` are used directly as `cv2.fisheye` k1–k4. The small
   higher-order radial + tangential + thin-prism terms are dropped (a faithful
   approximation here — they are ~1e-3 to 1e-4).
3. **Rectify → SGBM → depth** — both views are remapped to a common virtual
   pinhole pair, matched, and disparity is converted to metric depth
   (`Z = f_rect · B / disparity`).

## Published topics (rectified-left frame `camera0`)

| Topic                  | Type                     | Notes                          |
| ---------------------- | ------------------------ | ------------------------------ |
| `stereo/depth`         | `sensor_msgs/Image`      | `32FC1`, metres, `NaN` invalid |
| `stereo/depth_color`   | `sensor_msgs/Image`      | `bgr8` colorized (near=red)    |
| `stereo/camera_info`   | `sensor_msgs/CameraInfo` | rectified-left intrinsics      |
| `stereo/image_rect`    | `sensor_msgs/Image`      | `mono8` rectified-left view    |

## Run

```bash
# inside the container, from /workspace/ros2_ws
colcon build --packages-select hot3_evaluation
source install/setup.bash

# 1) publish the dataset
ros2 launch hot3_dataset_interface hot3_dataset_interface.launch.py
# 2) in another shell: depth eval + RViz
ros2 launch hot3_evaluation stereo_depth_eval.launch.py
```

The RViz layout shows `stereo/depth_color`, the rectified-left view, and a
DepthCloud. Pass `rviz:=false` to run headless.

## Expected result (honest)

The pipeline is geometrically correct — rectified epipolar lines are row-aligned
— **but depth is sparse and noisy** (~10–20% valid pixels). This is inherent:
the two cameras diverge ~75°, so the *overlapping* field of view after
rectification is small and the rectified focal collapses to ~59 px. The central
region (e.g. a hand) gets a usable depth gradient; the periphery does not. This
camera pair is a head-tracking rig, not a depth-stereo rig.

For reliable 3D from HOT3D, prefer the dataset's own 3D hand annotations
(`hands.json`) or triangulating 2D keypoints across cameras using the `/tf`
extrinsics, rather than dense block-matching on this pair.

## Tuning

Edit `config/stereo_depth_eval.yaml`:

- `balance` (0–1): `0` crops to the valid overlap (densest depth); `1` keeps the
  full source FOV.
- `fov_scale`: `>1` zooms out.
- `num_disparities` (mult. of 16), `block_size` (odd), `uniqueness_ratio`,
  speckle filters: standard SGBM knobs.
- `min_depth` / `max_depth`: validity clamp + color range.

## Not handled

The RGB camera (`214-1` / `camera2`) is not used — it isn't part of the stereo
pair. The dropped FISHEYE624 distortion terms are not modeled (would need
`projectaria_tools`, which isn't installed).
