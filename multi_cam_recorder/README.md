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
