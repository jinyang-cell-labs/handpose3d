# statistics_evaluation

Host-side Python environment for analysing and visualising the handpose **JSONL
logs** produced by `ros2_ws/src/mediapie_landmarks_extraction` (the
service-driven logger). Used for 2D/3D plotting, triangulation, and non-linear
optimisation over the recorded landmarks + calibration.

## Quick start

```bash
cd tools/statistics_evaluation
./start_evaluation.bash
```

This will:
1. create `./.venv` if it doesn't exist (using `python3`, override with
   `PYTHON_BIN=python3.x`),
2. install [`requirements.txt`](requirements.txt) into it (only when that file
   changed since the last run),
3. activate the venv and drop you into a shell that **stays inside** it —
   `exit` leaves.

Prefer to activate in your *current* shell instead of a subshell?

```bash
source start_evaluation.bash
```

## What's available inside

- **numpy / scipy** — linear algebra, DLT triangulation, `scipy.optimize.least_squares`
  for non-linear refinement (bundle-adjustment-style problems).
- **matplotlib** (incl. `mpl_toolkits.mplot3d`) — 2D and 3D plots.
- **opencv-python** — `triangulatePoints`, `projectPoints`, `Rodrigues`.
- **pandas** — wrangling the per-frame records.

The log loader is on `PYTHONPATH`, so analysis scripts can do:

```python
from load_handpose_log import load_log, stack_camera

log = load_log("../../ros2_ws/logs/handpose_log_20260626_120000.jsonl")
K = log.meta["cameras"]["camera0"]["intrinsics"]["K"]       # row-major 3x3
T = log.meta["cameras"]["camera0"]["T_world_cam"]           # 4x4 cam->world
s = stack_camera(log, "camera0", "Left")                    # NaN-filled arrays
#   s["stamp_ns"] (T,)   s["image"] (T,21,3)   s["world"] (T,21,3)   s["score"] (T,)
```

Recorded logs live in `ros2_ws/logs/` (inside the container that's
`/workspace/ros2_ws/logs/`). The `.venv/` here is gitignored.
