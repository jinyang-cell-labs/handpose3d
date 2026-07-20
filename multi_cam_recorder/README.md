# Multi-Camera Recorder

PyQt GUI to preview, record, and replay multiple V4L2 webcams with
frame-locked synchronization.

## Quick start

```bash
./start_recorder.bash                # uses config.yaml next to the script
./start_recorder.bash my_setup.yaml  # custom config
```

The script bootstraps a local `.venv` and installs dependencies on first run.

## Configuration (`config.yaml`)

| key | default | meaning |
|---|---|---|
| `cameras` | `[0, 2, 4, 6]` | initial device per preview window (list length = number of windows) — int index, `/dev/videoN` path, stable `/dev/v4l/by-id/...` symlink (survives replug), or `null` for an empty window. Each window has a dropdown in the GUI to switch devices; "Rescan devices" picks up hot-plugged cameras. Only the even node of a UVC camera captures; the odd sibling is metadata. |
| `width` / `height` | `1280` / `720` | capture resolution requested per camera |
| `fps` | `30` | recording clock rate |
| `capture_fourcc` | `MJPG` | pixel format requested from the camera |
| `record_fourcc` / `record_extension` | `MJPG` / `avi` | codec + container written to disk |
| `output_dir` | `~/recordings/multi_cam` | where sessions are stored |
| `preview_fps` | `20` | GUI preview refresh rate |
| `grid_columns` | `2` | preview grid layout |

## How synchronization works

Each camera runs in its own capture thread that keeps only its **latest**
frame. A single recording clock ticks at `fps`; on every tick it writes the
current frame of *every* camera to that camera's file. Therefore:

- all files of a session have the **same frame count**, and frame *i* in
  every file belongs to the same clock tick;
- a slow/stalled camera never blocks the others — its previous frame is
  duplicated instead;
- `timestamps.csv` records, per tick, each camera's frame sequence number
  and true capture time (`time.monotonic()`), so residual skew (bounded by
  one camera frame interval) can be measured offline.

## Session layout

```
<output_dir>/session_20260717_143000/
    cam_video0.avi
    cam_video2.avi
    cam_video4.avi
    cam_video6.avi
    meta.yaml         # cameras, resolution, fps, start time
    timestamps.csv    # tick_idx, tick_time, per-camera seq + capture time
```

## Playback

The **Playback** tab lists all sessions in `output_dir`. Selecting one opens
every video of the session and plays them frame-locked (all views always show
the same tick), with play/pause and a seek slider.

## Calibrate

The **Calibrate** tab runs multi-camera calibration on a recorded session,
reusing the solver of `ros2_ws/src/calibration_multi_cam` (AprilGrid detection,
per-camera intrinsics, PnP + spanning-tree extrinsics, bundle adjustment) with
the ROS collection layer replaced by a reader for recorder sessions.

**Rigs.** Cameras are grouped into rigs — sets rigidly mounted together (e.g. a
fixed table pair and a head-mounted pair). Extrinsics are solved *independently
per rig*, because the pose between rigs is dynamic; the relative pose between
rigs is never estimated. Each rig has a **reference camera** whose frame is its
origin (0,0,0). Assign rig, projection model and reference per camera in the
tab (defaults come from `calibration:` in config.yaml).

**Workflow.** Hold the AprilGrid so it is visible to the cameras and record one
take (a single session covers both stages). Then in Calibrate: pick the
session, confirm the rig table, press **Run**. The stage label + progress bar
and the live detection preview show which stage is running (extract →
intrinsics → per-rig extrinsics → write); the log shows RMS and coverage.

**Synchronization.** Uses `timestamps.csv`: a rig view is kept only when every
rig camera has a *fresh* frame (not a slow camera's duplicate) with capture-time
skew below `sync_max_skew`. This matters for the moving head rig, where pairing
a fresh frame with a stale one injects pose error.

**Outputs** → `<session>/calibration/`:

```yaml
# intrinsics.yaml
cameras:
  video0: {model: pinhole-radtan, resolution: [1280,720],
           intrinsics: [fx,fy,cx,cy], distortion: [k1,k2,p1,p2],
           reproj_rms: 1.3, num_views: 48}
```
```yaml
# extrinsics.yaml — per rig, reference camera is identity
rigs:
  head:
    reference: video5
    cameras:
      video5: {T_ref_cam: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
      video8: {T_ref_cam: [[...]]}   # pose of video8 in video5's frame
```
`report.yaml` records views, coverage and per-camera/per-rig RMS.

**Requirements.** Needs `cv2.aruco` (opencv-contrib) and `scipy`, both in
requirements.txt. OpenCV is pinned `<5`: the solver targets OpenCV 4.x (ROS 2
Jazzy), and 5.0 relocated the `cv2.fisheye.CALIB_*` flags.

**Coverage matters.** Intrinsics need the board across the *whole* frame,
especially the edges for fisheye (`pinhole-equi`) lenses — the tab warns when a
camera's board coverage is below 50%. Extrinsics need many views where *both*
rig cameras see the board together; if a rig reports too few synchronized
views, record a take aiming the board so both its cameras see it at once.
