# body_cam_teleop

Multi-camera hand-pose teleoperation pipeline in ONE package. Consolidates
(and replaces at runtime) the chain

    multi_cam_stream -> mediapie_landmarks_extraction
                     -> single_cam_pose_estimation -> handpose_teleop

One identical pipeline pair per camera, each in its own namespace, plus one
selector; no image traffic on DDS unless the reprojection overlay is on:

```
per camera namespace (/cam0, /cam1, ...)
hand_landmarks_node (Python)                hand_pose_node (C++)
┌──────────────────────────────┐            ┌───────────────────────────────────┐
│ cv2.VideoCapture (ONE camera)│ landmarks  │ undistort the 21 points (K, D)    │
│ MediaPipe HandLandmarker     │──────────▶ │ scale hand model ×1.3             │
│ (in-process)                 │  (tiny     │ Ceres PnP  T_cam_hand             │
│                              │   msg)     │ anatomical palm frame             │
│ [image for                   │            │ T_body_hand = T_body_cam·T_cam_hand│
│  enable_reprojection]        │            │ TeleopMessage @ 50 Hz ▶ <ns>/teleop│
└──────────────────────────────┘            └───────────────────────────────────┘

teleop_mux_node (Python): sticky per-hand selection over all <ns>/teleop
                          ▶ /teleop_converted (single stream to the arm)
```

The mux keeps a hand's current camera while that camera still offers a fresh
pose (trigger held) and only then fails over to another camera, so the small
pose offsets between the cameras' body-frame estimates don't toggle every
frame.

## Run

```bash
ros2 launch body_cam_teleop body_cam_teleop.launch.py
# subset of the configured cameras
ros2 launch body_cam_teleop body_cam_teleop.launch.py cameras:=cam0
# with rviz
ros2 launch body_cam_teleop body_cam_teleop.launch.py rviz:=true
```

All parameters: `config/body_cam_teleop.yaml`. Shared parameters live once under
the `/**/` wildcard blocks; the camera set is the `/camN:` blocks at the
bottom (per camera only `camera_name` — the calibration entry — and
`camera_device` — the V4L device). The launch file spawns one pipeline pair
per block; adding a camera = adding one block.

## Calibration

`config/intrinsics.yaml` is a copy of the calibration_multi_cam output and
must contain every configured `camera_name`. Per camera, `model` selects the
distortion model: `pinhole-radtan` ([k1,k2,p1,p2], the default when the key
is absent) or `pinhole-equi` (fisheye/equidistant, exactly [k1,k2,k3,k4] —
handled with the `cv::fisheye::` API). Any other model string is rejected at
startup. No extrinsics and no board detection: the camera is mounted at the
operator body center, so camera -> operator_body is the fixed
`operator_body_position/rotation` offset in `config/body_cam_teleop.yaml`
(identity by default; the rotation is only there to re-align the camera
optical axes with the body convention if needed). All instances share the
same offset, which is what makes their body frames coincide. Each node
broadcasts its `<ns>/operator_body -> <camera_name>` mount offset as
namespace-prefixed static TF so RViz can show it without frame collisions.

### Hand-size scale calibration (multi-camera)

Monocular PnP gets the hand's depth entirely from the assumed hand size, so a
wrong `hand_size_scaling_factor` slides each camera's estimate of the SAME
hand along its own camera ray — visible in RViz as a translational gap
between the per-camera skeletons. `hand_scale_calib_node` estimates the
correction from cross-camera agreement: scaling the factor by `x` moves a
joint `J` seen by camera `i` to `C_i + x (J - C_i)`, so the best `x` is the
scalar least-squares fit over all joints/hands/frames/camera pairs of a time
window (closed form), and the recommended factor is `current * x`.

```bash
# log-only: prints x, the recommended factor and the rms gap per window
ros2 launch body_cam_teleop body_cam_teleop.launch.py calibrate_hand_scale:=log
# closed loop: also live-updates every hand_pose_node (damped) until x ~ 1
ros2 launch body_cam_teleop body_cam_teleop.launch.py calibrate_hand_scale:=apply
```

Show ONE hand to all cameras and hold it still or move slowly (the position
filter lags motion, which inflates estimation noise). The converged value
must be persisted into `body_cam_teleop.yaml` by hand — runtime parameters die
with the session. The rms gap left after convergence is cross-camera
intrinsics/mount disagreement, which no scale factor can remove. Needs >= 2
cameras and forces `enable_reprojection` on (the node consumes the marker
topics); `hand_size_scaling_factor` is dynamically updatable on
hand_pose_node (`ros2 param set`) to support the apply loop.

## Performance profiling

Every pipeline node self-times its stages (`enable_perf: true`, on by
default, ~free: a 1 Hz JSON `std_msgs/String` on `<ns>/body_cam_teleop/perf`).
hand_landmarks_node reports `capture_ms` / `convert_ms` / `mediapipe_ms` /
`publish_*_ms` plus achieved-vs-target fps; hand_pose_node reports
`undistort_ms` / `solve_ms` (+ Ceres iterations) / `reproject_ms` /
`markers_ms` and
`latency_capture_to_pose_ms` (capture -> landmark-arrival age, i.e. the
upstream end-to-end latency).

To record a session, add `perf:=true` (or run perf_monitor_node.py alongside
an already-running pipeline):

```bash
ros2 launch body_cam_teleop body_cam_teleop.launch.py perf:=true
# reproduce the workload for a minute or two, Ctrl-C, then:
python3 src/teleoperation/body_cam_teleop/scripts/perf_report.py   # newest run
```

perf_monitor_node writes three CSVs per run under `perf_log_dir` (default
`/workspace/robot/ros2_ws/logs/perf`): `*_stages.csv` (the per-stage
timings), `*_topics.csv` (message rates, counted on raw subscriptions — the
image topics are deliberately not subscribed), `*_system.csv` (per-process
CPU/RSS/threads plus a system TOTAL, sampled from /proc). It also logs a
top-stages summary every 10 s while recording.

perf_report.py ranks every (node, stage) by **core%** — the share of one CPU
core the stage kept busy over the run — so the top rows are the expensive
steps; it also prints fps/latency/solver metrics, topic rates, per-process
CPU, and heuristic bottleneck hints. Compare two runs (e.g. `delegate: cpu`
vs `gpu`, or `enable_reprojection` on vs off) by passing each run's prefix.

## Diagnosing dropouts (which gate ate the hand?)

A detection must clear **five gates** before the arm controller sees it. When
`teleop_mux_node` logs `left hand source: /cam0/teleop -> None`, one of them
rejected every detection for `pose_timeout_sec`:

| Gate | Where | Rejects when | Knob |
| --- | --- | --- | --- |
| 1 detect | hand_landmarks_node | MediaPipe found no hand | `min_hand_detection_confidence`, `min_hand_presence_confidence`, `min_tracking_confidence` |
| 2 label/score | hand_landmarks_node | label not allowed, or handedness score too low | `hand_filter_mode`, `min_handedness_confidence` |
| 3 score/contract | hand_pose_node | `hand.score` below threshold, or not 21 landmarks | `min_score` |
| 4 solve | hand_pose_node | Ceres produced no usable pose | `max_solver_iterations`, `seed_depth_m`, `warm_start` |
| 5 freshness | hand_pose_node | no pose within the timeout → trigger released | `pose_timeout_sec` |

Each gate has its own log flag, and every reject line prints the **measured
value and the threshold that rejected it**. All flags are runtime-settable, so
a live pipeline can be instrumented without a restart.

Start with the funnel counts — they show which gate is eating hands without
any per-frame spam:

```bash
ros2 param set /cam0/hand_landmarks_node log_gate_summary true
ros2 param set /cam0/hand_pose_node      log_gate_summary true
ros2 param set /teleop_mux_node          log_gate_summary true
```

```
[gate funnel 5.0s] ticks=150 (no_frame=0) -> results=148 -> gate1 detected=131 hand(s)
  -> gate2 rejected: label=131 handedness_score=0 no_world_landmarks=0 -> published=0 hand(s) in 0 frame(s)
```

(that example: MediaPipe is detecting fine, `hand_filter_mode` is dropping
every hand — its labels assume an *unmirrored* image and flip when the hand
rotates palm↔back.)

Then turn on the detail flag for the guilty gate only:

```bash
ros2 param set /cam0/hand_landmarks_node log_handedness true   # gate 2, per hand
ros2 param set /cam0/hand_pose_node      log_score_gate  true   # gate 3
ros2 param set /cam0/hand_pose_node      log_solve       true   # gate 4 + Ceres detail
ros2 param set /cam0/hand_pose_node      log_trigger     true   # gate 5 engage/release
ros2 param set /teleop_mux_node          log_offers      true   # why no source offers
```

Full flag list: `log_capture` / `log_detection` / `log_handedness` /
`log_publish` (hand_landmarks_node), `log_input` / `log_score_gate` /
`log_solve` / `log_pose` / `log_trigger` (hand_pose_node), `log_offers`
(teleop_mux_node), plus `log_gate_summary`, `log_throttle_sec` (detail lines
are throttled per stage+reason+hand, default 2 s) and
`log_summary_period_sec` everywhere.

**`log_reproj_warn_px`** (hand_pose_node, default 30) is independent of the
flags above: it warns whenever a solve *converges* but the placed model still
misses the detected landmarks by more than that many pixels. Nothing gates on
reprojection error, so such a pose is published — trigger held, arm following a
meaningless target. A large rms points at `hand_size_scaling_factor` or the
camera intrinsics, not at detection sensitivity.

## What changed vs the old 4-package pipeline (performance)

* Camera capture and MediaPipe run in ONE process: raw 1280x720 frames are no
  longer serialized over DDS at 30 Hz (that alone was a full extra copy +
  pub/sub of ~2.7 MB per frame). With `enable_reprojection: false` (default)
  no image leaves the process at all.
* Full-frame undistortion (`cv2.remap` per frame) is gone: MediaPipe detects
  on the raw frame and only the 21 landmark points are undistorted
  (`cv::undistortPoints`) before the pinhole PnP. The reprojection overlay
  projects with the full distortion model back onto the raw image, so the QA
  view is equivalent.
* The nonlinear reprojection solve moved from scipy `least_squares` (Python,
  numerical Jacobian) to Ceres (C++, autodiff, DENSE_QR), warm-started from
  the previous frame's pose. Solve time is ~tens of microseconds; the C++
  node idles at ~2% of a core including the 50 Hz TeleopMessage timer.
* Dropped features: annotated 2D landmark images, MediaPipe world-landmark
  markers, JSONL session logging. The calibration workflow stays in
  calibration_multi_cam (multi-camera came back as N independent monocular
  pipelines + mux, not as joint multi-view estimation).

Behavior kept: hand_size_scaling_factor 1.3, cheirality penalty (margin 0.05 m,
weight 1000 px/m), constant 0.5 m seed, anatomical palm frame, min_score 0.5,
50 Hz best-effort `/teleop_converted` with BUTTON_TRIGGER held per fresh hand,
0.5 s pose timeout, 25 NaN hand joints, `hand_orientation_offset_rpy`.

## Topics

| topic | type | when |
|---|---|---|
| `/camN/body_cam_teleop/landmarks` | handpose3d_msgs/HandLandmarks | always |
| `/camN/teleop` | robot_interfaces/TeleopMessage (best-effort, 50 Hz) | always (per camera) |
| `/teleop_converted` | robot_interfaces/TeleopMessage (best-effort, 50 Hz) | always (mux output) |
| `/camN/body_cam_teleop/image_raw` | sensor_msgs/Image | enable_reprojection |
| `/camN/body_cam_teleop/image_reprojected` | sensor_msgs/Image (overlay + px error) | enable_reprojection |
| `/camN/body_cam_teleop/markers` | visualization_msgs/MarkerArray (camN/operator_body frame) | enable_reprojection |

## Dependencies

`libceres-dev` (apt / rosdep), `mediapipe` (pip, no rosdep key), plus the
in-repo `handpose3d_msgs` and `robot_interfaces`.
