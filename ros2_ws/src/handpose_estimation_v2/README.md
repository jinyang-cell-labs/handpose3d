# handpose_estimation_v2 — model-based fit

Experimental successor to `handpose_estimation`. The front-end (two calibrated
cameras, MediaPipe HandLandmarker, handedness matching) and the back-end
(B1 reachability gate, D Kalman, E One-Euro) are unchanged; the **estimation
core replaces stages A + C**:

| | v1 (triangulate-then-fit) | v2 (model-based fit) |
|---|---|---|
| A | per-joint weighted DLT (21 independent 2-view problems) | — |
| C | 3D Procrustes of template onto triangulated points | direct LM fit of template pose (R, t) minimising 2D reprojection error in **all views simultaneously** |
| B2 | RANSAC voting in 3D metres | RANSAC voting in **pixels** (worst per-view residual) |
| depth constraint | two-ray intersection only | template bone lengths/scale inside the solve |
| temporal coupling | output smoothing only (D/E) | fit **warm-started from previous pose** (+ D/E) |
| one-view dropout | coast | **monocular PnP continuation** (`single_view.enabled`) |

Cold start (no previous pose) still bootstraps from v1's DLT + Procrustes and
needs both views; `model_fit.enabled: false` runs that bootstrap alone as an
ablation (≈ v1 behaviour inside the v2 node).

## Layout

- `model_fit.py` — ROS-free math: Levenberg–Marquardt rigid fit on SE(3)
  (Huber-robust, confidence-weighted), pixel-domain RANSAC, DLT bootstrap.
- `model_pose_pipeline.py` — `ModelFitWristTracker`, subclasses v1's
  `WristTracker`; replaces only the measurement step, reuses D/E/coasting.
- `handpose_node.py` — `handpose_node_v2`; publishes the same topics as v1
  (`handpose/wrist_left|right`, `handpose/markers`, annotated views), so RViz
  configs and consumers work unchanged. **Don't run v1 and v2 nodes at the
  same time without remapping** — the topic names collide.

The MediaPipe model, hand template and extrinsics files are shared from the
v1 package source tree (see `config/handpose_estimation_v2.yaml`).

## Run (in the docker container)

```bash
colcon build --packages-select handpose_estimation handpose_estimation_v2
source install/setup.bash
ros2 launch handpose_estimation_v2 handpose_estimation_v2.launch.py
```

## Tests (host, repo venv)

```bash
cd ros2_ws/src/handpose_estimation_v2
/home/jinyang/repo/handpose3d/.venv/bin/python -m pytest test/ -q
```
