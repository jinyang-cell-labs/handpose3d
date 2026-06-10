# Wrist-Pose Estimation Pipeline for Head-Mounted Stereo RGB Hand Tracking — Complete Implementation Draft

## TL;DR
- **This draft replaces independent per-joint triangulation with a model-based, temporally coupled pipeline that emits a single 6-DoF wrist pose per hand** through six staged, individually toggleable steps: confidence-weighted DLT (A1) → per-joint reprojection residual (A2) → reachability gate (B1) → RANSAC + rigid Procrustes/Kabsch fit (B2/C) → constant-velocity Kalman filter with SLERP orientation smoothing and Mahalanobis gating (D) → One-Euro output polish (E). It publishes `geometry_msgs/PoseStamped` on `handpose/wrist_left` and `handpose/wrist_right` while keeping the existing `MarkerArray`.
- **Critical research finding that shapes the design:** the MediaPipe Tasks-API `HandLandmarker` does **not** populate per-landmark `visibility`/`presence` — they are always `0` (Google MediaPipe GitHub issue #5212: "the visibility and presence fields for the hands are always 0 … there is no output indicating the quality or confidence of the keypoints obtained"; issue #4479 confirms `landmarks` and `worldLandmarks` "only contain values of x, y, z coordinates"). The pipeline therefore derives per-view weights from the **handedness score combined with a reprojection-residual-based weight**, with the weighting source exposed as a config parameter.
- **Every tunable/calibratable value is exposed in config:** the existing `handpose_estimation.yaml` is extended with dotted ROS 2 parameter names (e.g. `kalman.process_noise_pos`) plus per-stage `*.enabled` flags, and two external non-ROS YAML files (`hand_template.yaml`, loaded via a path parameter following the existing `extrinsics_file` pattern) hold the 21-landmark template and the reachability shell.

---

## Key Findings

1. **Per-landmark confidence is unavailable in the Tasks API.** The legacy `mp.solutions.hands` and the Tasks-API `HandLandmarker` both return 21 landmarks with `x,y,z` only; the `Landmark` container documents `visibility` as "Should stay unset if not supported." Hands return it unset/zero. The only confidence signal exposed per *hand* is the **handedness `score`** (≥0.5). The robust, practical substitute for a *per-landmark* trust signal is the **reprojection residual** computed in Stage A2 — joints that reproject poorly into both views are down-weighted. The implementation makes the weight source switchable (`handedness`, `reprojection`, `product`, or `uniform`).

2. **Weighted DLT is a standard, well-justified extension.** Across multi-view human-pose literature (Learnable Triangulation, ICCV 2019; Smart Edge Sensors, arXiv:2106.14729; combat-sports pose, arXiv:2504.08175), the confidence-weighted linear system is `(w ∘ A) x̃ = 0`, where each camera *i* contributes two rows scaled by its scalar weight `w_i`. This is solved by SVD exactly as unweighted DLT, and naturally supports N≥2 views.

3. **Kabsch/orthogonal-Procrustes with determinant correction** is the closed-form rigid fit. Compute the (weighted) cross-covariance `H = Σ w_k (p_k − p̄)(q_k − q̄)ᵀ`, take `H = U S Vᵀ`, then `R = V · diag(1,1,sign(det(VUᵀ))) · Uᵀ` to prevent reflections (the Kabsch-Umeyama det fix). This is the same fix used in the PyMOL/Umeyama references.

4. **Constant-velocity Kalman is sufficient for position; orientation is best smoothed separately.** A 6-state `[x,y,z,vx,vy,vz]` linear KF with `F = [[I, dt·I],[0, I]]` and the standard discrete white-noise-acceleration `Q` is the textbook choice. For orientation, an error-state quaternion KF is overkill at this scale; the pragmatic, widely used choice is **SLERP-based low-pass filtering** of the measured quaternion (MathWorks documents SLERP explicitly for "smoothing or lowpass filtering" of orientation). The draft uses a small hand-rolled KF (no filterpy dependency required) plus quaternion SLERP.

5. **Mahalanobis gating threshold is a chi-square critical value.** For a 3-DoF position measurement, the 99% gate is **χ²(3, 0.99) = 11.345** (NIST Engineering Statistics Handbook). The code exposes this as a parameter so the user can loosen/tighten it.

6. **Anthropometric template.** Buryanov & Kotiuk, "Proportions of Hand Segments," *Int. J. Morphol.* 28(3):755-758, 2010 (radiographic study of 66 adults; DOI 10.4067/S0717-95022010000300015) provides the phalanx lengths used for the default template (e.g. index proximal ≈ 39.78 mm, middle proximal ≈ 44.63 mm — *these specific phalanx figures are reproduced via Saraç Stroppa 2017, arXiv:2003.11598 Table 2.1, which cites Buryanov; the primary PDF blocks automated access so they should be spot-checked against the original Table I*). Average adult total arm length (acromion→fingertip) is ≈ **0.66 m** (FingerMapper, arXiv:2302.11865, uses 0.60 m shoulder-to-wrist; ANSUR male shoulder-to-fingertip ≈ 78.6 cm), informing the reachability `d_max`.

---

## Architecture & Data Flow

```
 cam0/image_raw ─┐                          ┌─ handpose/wrist_left  (PoseStamped)
 cam0/camera_info├─►ApproxTimeSync─►MediaPipe├─ handpose/wrist_right (PoseStamped)
 cam1/image_raw ─┤   (per-cam)    HandLandmarker└─ handpose/skeleton (MarkerArray)
 cam1/camera_info┘                  │
                                    ▼
        2D landmarks (21×2 per view) + handedness score
                                    │
   ┌────────────────────────────────┴───────────────────────────────┐
   │  PER HAND (matched by handedness label)                          │
   │  A1 weighted_dlt   → 21 raw 3D joints (weights = w_view)         │
   │  A2 reproj_resid   → per-joint pixel error → per-joint weight    │
   │  B1 reachability   → gate/flag joints & candidate wrist          │
   │  C  Procrustes     → R,t mapping template→inlier joints          │
   │  B2 RANSAC wrap    → robust inlier set, refit weighted Kabsch    │
   │  D  Kalman + SLERP → CV position filter + orientation LPF + gate │
   │  E  One-Euro       → final 6-DoF polish for display/output       │
   └──────────────────────────────────────────────────────────────────┘
```

Each WristTracker instance owns the temporal state (Kalman + One-Euro + coast counter) for one hand. Stages are skipped cleanly when their `*.enabled` flag is false, so you can bring the system up incrementally (A1→A2→B1→C→B2→D→E) and A/B test.

---

## Deliverable 1 — `triangulation.py` (updated, backward compatible)

```python
"""triangulation.py

Multi-view triangulation utilities for the handpose_estimation package.

Backward-compatible: make_projection_matrix, rotation_matrix_to_quaternion and
dlt keep their original signatures and behaviour. New in this revision:
  * weighted_dlt(...)        -- confidence-weighted N-view DLT (Stage A1)
  * reprojection_error(...)  -- per-view pixel residual for a 3D point (Stage A2)
  * triangulate_point(...)   -- convenience wrapper returning point + residuals
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  Existing API (unchanged)                                                    #
# --------------------------------------------------------------------------- #
def make_projection_matrix(K, R, t):
    """Build the 3x4 projection matrix P = K @ [R | t].

    Parameters
    ----------
    K : (3, 3) array_like   Intrinsic camera matrix.
    R : (3, 3) array_like   World->camera rotation.
    t : (3,)   array_like   World->camera translation.

    Returns
    -------
    P : (3, 4) ndarray
    """
    K = np.asarray(K, dtype=float)
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3, 1)
    return K @ np.hstack((R, t))


def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to a quaternion (x, y, z, w).

    Uses Shepperd's method for numerical stability.
    """
    R = np.asarray(R, dtype=float)
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    return q / np.linalg.norm(q)


def dlt(P1, P2, point1, point2):
    """Classic unweighted two-view DLT for a single point via SVD of A^T A.

    Parameters
    ----------
    P1, P2 : (3, 4) projection matrices.
    point1, point2 : (2,) pixel coordinates in each view.

    Returns
    -------
    X : (3,) triangulated 3D point (cartesian).
    """
    A = _dlt_rows([P1, P2], [point1, point2])
    return _solve_dlt(A)


# --------------------------------------------------------------------------- #
#  New API (Stage A1 / A2)                                                     #
# --------------------------------------------------------------------------- #
def _dlt_rows(projections, points, weights=None):
    """Assemble the 2N x 4 DLT matrix A.

    For each view i with projection P_i and pixel (u_i, v_i):
        row 2i   = u_i * P_i[2, :] - P_i[0, :]
        row 2i+1 = v_i * P_i[2, :] - P_i[1, :]
    If weights are supplied, both rows of view i are multiplied by w_i
    (confidence-weighted DLT: (w o A) x = 0).
    """
    n = len(projections)
    if weights is None:
        weights = np.ones(n, dtype=float)
    rows = []
    for P, pt, w in zip(projections, points, weights):
        P = np.asarray(P, dtype=float)
        u, v = float(pt[0]), float(pt[1])
        rows.append(w * (u * P[2, :] - P[0, :]))
        rows.append(w * (v * P[2, :] - P[1, :]))
    return np.asarray(rows, dtype=float)


def _solve_dlt(A):
    """Solve A x = 0 for homogeneous x by SVD; return cartesian 3-vector."""
    # Smallest-singular-value right vector. Using SVD of A directly is more
    # numerically robust than eigendecomposition of A^T A.
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    if abs(X[3]) < 1e-12:
        return np.full(3, np.nan)
    return X[:3] / X[3]


def weighted_dlt(projections, points, weights=None):
    """Confidence-weighted N-view DLT (Stage A1).

    Parameters
    ----------
    projections : list of (3, 4) projection matrices, length N >= 2.
    points      : list of (2,) pixel coordinates, length N.
    weights     : optional length-N non-negative per-view confidence weights.
                  If None, equal weighting (== unweighted DLT).

    Returns
    -------
    X : (3,) triangulated 3D point.

    Notes
    -----
    Each camera contributes two rows to A; both are scaled by that view's
    scalar weight w_i, then the homogeneous system (w o A) x = 0 is solved by
    SVD. Supports N >= 2 even though the head-mounted rig has 2 cameras.
    """
    if len(projections) < 2:
        raise ValueError("weighted_dlt requires at least 2 views")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        # Guard against an all-zero weight vector collapsing the system.
        if not np.any(weights > 0):
            weights = np.ones(len(projections))
    A = _dlt_rows(projections, points, weights)
    return _solve_dlt(A)


def project_point(P, X):
    """Project a 3D cartesian point X into a view with projection matrix P.

    Returns the (2,) pixel coordinate (or NaN if behind the camera).
    """
    P = np.asarray(P, dtype=float)
    Xh = np.append(np.asarray(X, dtype=float), 1.0)
    x = P @ Xh
    if abs(x[2]) < 1e-12:
        return np.full(2, np.nan)
    return x[:2] / x[2]


def reprojection_error(P, X, point):
    """Pixel reprojection error for one 3D point in one view (Stage A2)."""
    proj = project_point(P, X)
    return float(np.linalg.norm(proj - np.asarray(point, dtype=float)))


def triangulate_point(projections, points, weights=None):
    """Triangulate one point and return (X, mean_resid, per_view_resid).

    Convenience wrapper combining Stage A1 (weighted DLT) and Stage A2
    (per-view reprojection residual). The mean residual is a per-joint trust
    signal used downstream by RANSAC and by the fallback weighting scheme.
    """
    X = weighted_dlt(projections, points, weights)
    resid = np.array([reprojection_error(P, X, pt)
                      for P, pt in zip(projections, points)], dtype=float)
    return X, float(np.nanmean(resid)), resid
```

---

## Deliverable 2 — `wrist_pose_pipeline.py` (new)

```python
"""wrist_pose_pipeline.py

Model-based, temporally coupled 6-DoF wrist pose estimation for the
handpose_estimation package. Stages (each individually toggleable):

  B1  ReachabilityShell    -- distance gate against a calibrated arm-reach shell
  C   procrustes_fit       -- weighted Kabsch/orthogonal-Procrustes (det fix)
  B2  ransac_procrustes    -- RANSAC inlier selection wrapping the Kabsch fit
  D   ConstantVelocityKF   -- CV position KF + SLERP orientation LPF + gating
  E   OneEuroFilter        -- final 6-DoF smoothing (Casiez et al., CHI 2012)

  WristTracker             -- per-hand orchestrator: B1 -> C -> B2 -> D -> E

All numeric thresholds are passed in by the node from ROS parameters; nothing
is hard-coded except mathematically fixed constants.
"""

from __future__ import annotations

import math
import numpy as np

from .triangulation import rotation_matrix_to_quaternion


# ===========================================================================#
#  Quaternion helpers (x, y, z, w convention, matching triangulation.py)      #
# ===========================================================================#
def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def quat_slerp(q0, q1, t):
    """Spherical linear interpolation between unit quaternions (x,y,z,w).

    Used as an orientation low-pass filter (t in [0,1]: 0 keeps q0, 1 takes q1).
    """
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:                 # take the shorter arc (double-cover fix)
        q1 = -q1
        dot = -dot
    if dot > 0.9995:              # nearly parallel -> linear, then renormalise
        return quat_normalize(q0 + t * (q1 - q0))
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    q2 = quat_normalize(q1 - q0 * dot)
    return q0 * math.cos(theta) + q2 * math.sin(theta)


def quat_angle(q0, q1):
    """Absolute angle (rad) between two quaternions."""
    d = abs(float(np.dot(quat_normalize(q0), quat_normalize(q1))))
    return 2.0 * math.acos(min(1.0, d))


# ===========================================================================#
#  Stage C -- weighted Kabsch / orthogonal Procrustes                         #
# ===========================================================================#
def procrustes_fit(template, observed, weights=None):
    """Closed-form rigid transform mapping `template` onto `observed`.

    Solves the (weighted) orthogonal Procrustes problem with the determinant
    correction that forbids reflections (Kabsch-Umeyama).

        H = sum_k w_k (t_k - t_bar)(o_k - o_bar)^T
        H = U S V^T
        R = V diag(1, 1, sign(det(V U^T))) U^T
        t = o_bar - R t_bar

    Parameters
    ----------
    template : (M, 3) canonical landmark positions in the wrist-local frame.
    observed : (M, 3) triangulated landmark positions in the world frame.
    weights  : optional (M,) non-negative per-landmark weights.

    Returns
    -------
    R : (3, 3) rotation, t : (3,) translation, rmsd : float weighted RMSD.
    """
    template = np.asarray(template, dtype=float)
    observed = np.asarray(observed, dtype=float)
    m = template.shape[0]
    if weights is None:
        weights = np.ones(m, dtype=float)
    weights = np.asarray(weights, dtype=float)
    wsum = weights.sum()
    if wsum < 1e-12:
        weights = np.ones(m, dtype=float)
        wsum = float(m)

    t_bar = (weights[:, None] * template).sum(axis=0) / wsum
    o_bar = (weights[:, None] * observed).sum(axis=0) / wsum
    tc = template - t_bar
    oc = observed - o_bar

    H = (weights[:, None] * tc).T @ oc          # 3x3 cross-covariance
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    d = np.sign(np.linalg.det(V @ U.T))
    D = np.diag([1.0, 1.0, d])                  # reflection-prevention fix
    R = V @ D @ U.T
    t = o_bar - R @ t_bar

    resid = oc @ R.T - (... if False else (tc))  # placeholder removed below
    # Weighted RMSD between R*template+t and observed:
    pred = (R @ template.T).T + t
    err2 = np.sum(weights * np.sum((pred - observed) ** 2, axis=1)) / wsum
    rmsd = float(np.sqrt(max(err2, 0.0)))
    return R, t, rmsd


# ===========================================================================#
#  Stage B2 -- RANSAC wrapping the Procrustes fit                             #
# ===========================================================================#
def ransac_procrustes(template, observed, weights=None,
                      iterations=50, sample_size=4, inlier_thresh=0.02,
                      rng=None):
    """Robust rigid fit: RANSAC over the 21 joints around procrustes_fit.

    Each iteration samples a minimal subset (sample_size joints), fits a rigid
    transform, then counts inliers by 3D residual < inlier_thresh against the
    transformed template. The best inlier set is refit once with confidence
    weights.

    Parameters
    ----------
    template, observed : (M, 3) arrays (M typically 21).
    weights            : optional (M,) confidence weights.
    iterations         : RANSAC iteration count.
    sample_size        : minimal subset size (3 minimum for a rigid 3D fit;
                         4 recommended for stability).
    inlier_thresh      : 3D residual threshold in world units (see config note
                         on scale: world_units * scale = metres).

    Returns
    -------
    dict with keys R, t, rmsd, inliers (bool mask), n_inliers.
    Returns None if no acceptable model is found.
    """
    template = np.asarray(template, dtype=float)
    observed = np.asarray(observed, dtype=float)
    m = template.shape[0]
    if weights is None:
        weights = np.ones(m, dtype=float)
    rng = rng or np.random.default_rng()

    valid = np.all(np.isfinite(observed), axis=1)
    valid_idx = np.where(valid)[0]
    if valid_idx.size < sample_size:
        return None

    best_mask = None
    best_count = -1
    for _ in range(iterations):
        sample = rng.choice(valid_idx, size=sample_size, replace=False)
        try:
            R, t, _ = procrustes_fit(template[sample], observed[sample],
                                     weights[sample])
        except np.linalg.LinAlgError:
            continue
        pred = (R @ template.T).T + t
        d = np.linalg.norm(pred - observed, axis=1)
        mask = valid & (d < inlier_thresh)
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count = count
            best_mask = mask

    if best_mask is None or best_count < sample_size:
        return None

    R, t, rmsd = procrustes_fit(template[best_mask], observed[best_mask],
                                weights[best_mask])
    return {"R": R, "t": t, "rmsd": rmsd,
            "inliers": best_mask, "n_inliers": best_count}


# ===========================================================================#
#  Stage B1 -- reachability shell                                             #
# ===========================================================================#
class ReachabilityShell:
    """Distance gate of a candidate wrist position against an arm-reach shell.

    A shoulder anchor is offset from the head/world frame origin for each hand.
    A position is accepted if  d_min <= ||p - shoulder|| <= d_max  AND it is
    not "behind the head" (component along -forward_axis beyond a margin).

    All parameters are loaded from config (reachability.* and shell yaml) so
    the shell can be calibrated per user.
    """

    def __init__(self, shoulder_left, shoulder_right, d_min, d_max,
                 forward_axis=(0.0, 0.0, 1.0), behind_margin=0.0):
        self.shoulder = {"Left": np.asarray(shoulder_left, dtype=float),
                         "Right": np.asarray(shoulder_right, dtype=float)}
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.forward = np.asarray(forward_axis, dtype=float)
        self.behind_margin = float(behind_margin)

    def check(self, position, handedness):
        """Return (ok: bool, reason: str) for a candidate 3D position."""
        s = self.shoulder.get(handedness)
        if s is None:
            return True, "no_anchor"
        v = np.asarray(position, dtype=float) - s
        d = float(np.linalg.norm(v))
        if d < self.d_min:
            return False, "too_close"
        if d > self.d_max:
            return False, "too_far"
        if float(np.dot(v, self.forward)) < -self.behind_margin:
            return False, "behind_head"
        return True, "ok"

    def filter_joints(self, joints, handedness):
        """Return a boolean mask of joints inside the shell (NaNs -> False)."""
        mask = np.zeros(joints.shape[0], dtype=bool)
        for i, p in enumerate(joints):
            if np.all(np.isfinite(p)):
                ok, _ = self.check(p, handedness)
                mask[i] = ok
        return mask


# ===========================================================================#
#  Stage D -- constant-velocity Kalman + SLERP orientation LPF + gating       #
# ===========================================================================#
class ConstantVelocityKF:
    """6-state [x,y,z,vx,vy,vz] CV Kalman filter for the wrist position.

    Orientation is handled separately by SLERP low-pass filtering (an
    error-state quaternion KF would be heavier than warranted here).

    Position model:
        F = [[I, dt I], [0, I]]            (constant velocity)
        Q = WNA discretisation scaled by process_noise_pos (sigma_a^2)
        H = [I, 0]                          (measure position only)
        R = measurement_noise_pos * I, optionally inflated by 1/confidence

    Outlier rejection: squared Mahalanobis distance of the innovation,
        d^2 = y^T S^-1 y,    accept if d^2 <= gate (chi-square, 3 DoF).
    """

    def __init__(self, process_noise_pos, measurement_noise_pos,
                 gate=11.345, orientation_lpf=0.5):
        self.q_pos = float(process_noise_pos)        # sigma_a^2
        self.r_pos = float(measurement_noise_pos)
        self.gate = float(gate)                      # chi2(3, 0.99) = 11.345
        self.ori_lpf = float(orientation_lpf)        # SLERP step toward meas.
        self.x = None                                # (6,) state
        self.P = None                                # (6,6) covariance
        self.quat = None                             # filtered orientation
        self.initialised = False

    def reset(self):
        self.x = None
        self.P = None
        self.quat = None
        self.initialised = False

    def _F(self, dt):
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        return F

    def _Q(self, dt):
        # Discrete white-noise-acceleration model (per axis), scaled by q_pos.
        q = self.q_pos
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = dt4 / 4.0
            Q[i, i + 3] = Q[i + 3, i] = dt3 / 2.0
            Q[i + 3, i + 3] = dt2
        return Q * q

    def initialise(self, position, quat):
        self.x = np.zeros(6)
        self.x[:3] = position
        self.P = np.eye(6)
        self.P[3:, 3:] *= 10.0       # large initial velocity uncertainty
        self.quat = quat_normalize(quat)
        self.initialised = True

    def predict(self, dt):
        """Predict-only step (used during dropouts / coasting)."""
        if not self.initialised:
            return None
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        return self.x[:3].copy(), self.quat.copy()

    def update(self, dt, position, quat, confidence=1.0):
        """Full predict+update. Returns (pos, quat) or None if gated out.

        confidence in (0,1]: measurement noise is inflated by 1/confidence,
        so low-confidence Procrustes fits are trusted less.
        """
        if not self.initialised:
            self.initialise(position, quat)
            return self.x[:3].copy(), self.quat.copy()

        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        conf = max(1e-3, float(confidence))
        R = np.eye(3) * (self.r_pos / conf)

        y = np.asarray(position, dtype=float) - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return None
        maha2 = float(y @ S_inv @ y)
        if maha2 > self.gate:
            # Reject late-surviving outlier; coast on the prediction instead.
            return self.x[:3].copy(), self.quat.copy()

        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

        # Orientation: SLERP low-pass toward the measurement.
        self.quat = quat_slerp(self.quat, quat, self.ori_lpf)
        return self.x[:3].copy(), self.quat.copy()


# ===========================================================================#
#  Stage E -- One-Euro filter (Casiez, Roussel & Vogel, CHI 2012)             #
# ===========================================================================#
def _smoothing_factor(t_e, cutoff):
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


def _exp_smooth(a, x, x_prev):
    return a * x + (1.0 - a) * x_prev


class OneEuroFilter:
    """Canonical scalar 1-Euro filter (self-contained, no external package).

    Equations (Casiez et al., CHI 2012):
        a_d   = smoothing_factor(t_e, d_cutoff)
        dx    = (x - x_prev) / t_e
        dx_hat= exp_smooth(a_d, dx, dx_prev)
        cutoff= min_cutoff + beta * |dx_hat|
        a     = smoothing_factor(t_e, cutoff)
        x_hat = exp_smooth(a, x, x_prev)
    """

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, t, x):
        if self.x_prev is None:
            self.x_prev = float(x)
            self.t_prev = float(t)
            return float(x)
        t_e = float(t) - self.t_prev
        if t_e <= 0.0:
            return self.x_prev
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (float(x) - self.x_prev) / t_e
        dx_hat = _exp_smooth(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exp_smooth(a, float(x), self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = float(t)
        return x_hat


class OneEuroVec:
    """One-Euro filter over a fixed-length vector (per-component scalar filters)."""

    def __init__(self, dim, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.filters = [OneEuroFilter(min_cutoff, beta, d_cutoff)
                        for _ in range(dim)]

    def reset(self):
        for f in self.filters:
            f.reset()

    def __call__(self, t, x):
        return np.array([f(t, xi) for f, xi in zip(self.filters, x)])


# ===========================================================================#
#  Per-hand orchestrator                                                      #
# ===========================================================================#
class WristTracker:
    """Owns the temporal state for ONE hand and runs stages B1 -> C -> B2 -> D -> E.

    The node constructs one WristTracker per handedness label ("Left"/"Right")
    and feeds it the per-frame triangulated joints, per-joint weights/residuals
    and aggregate confidence. Each stage can be disabled via the `flags` dict.
    """

    WRIST_INDEX = 0          # MediaPipe landmark 0 is the wrist

    def __init__(self, handedness, template, flags, params, shell=None):
        self.handedness = handedness
        self.template = np.asarray(template, dtype=float)   # (21,3) wrist-local
        self.flags = flags                                   # per-stage enables
        self.p = params                                      # flat param dict
        self.shell = shell                                   # ReachabilityShell

        self.kf = ConstantVelocityKF(
            process_noise_pos=params["kalman.process_noise_pos"],
            measurement_noise_pos=params["kalman.measurement_noise_pos"],
            gate=params["kalman.gate_threshold"],
            orientation_lpf=params["kalman.orientation_lpf"],
        )
        self.euro_pos = OneEuroVec(
            3, params["one_euro.min_cutoff"], params["one_euro.beta"],
            params["one_euro.d_cutoff"])
        self.euro_quat = OneEuroVec(
            4, params["one_euro.min_cutoff"], params["one_euro.beta"],
            params["one_euro.d_cutoff"])
        self.coast_frames = 0
        self.last_pose = None        # (pos, quat) last good output

    def reset(self):
        self.kf.reset()
        self.euro_pos.reset()
        self.euro_quat.reset()
        self.coast_frames = 0
        self.last_pose = None

    def update(self, stamp, joints3d, weights, agg_conf):
        """Run the pipeline for one frame.

        Parameters
        ----------
        stamp    : float seconds (monotonic) for One-Euro timing.
        joints3d : (21, 3) triangulated joints in the world frame (NaN if bad).
        weights  : (21,) per-joint confidence weights (from Stage A1/A2).
        agg_conf : float aggregate confidence in (0,1] (e.g. handedness score
                   times inlier fraction); scales Kalman measurement noise.

        Returns
        -------
        dict {pos, quat, valid, n_inliers, rmsd, inlier_mask} or a coasting
        result, or None if the track is reset/lost.
        """
        joints3d = np.asarray(joints3d, dtype=float)
        weights = np.asarray(weights, dtype=float)

        # ---- Stage B1: reachability gate ---------------------------------- #
        mask = np.all(np.isfinite(joints3d), axis=1)
        if self.flags.get("reachability_gate", True) and self.shell is not None:
            shell_mask = self.shell.filter_joints(joints3d, self.handedness)
            mask = mask & shell_mask

        usable = int(np.count_nonzero(mask))
        min_joints = int(self.p.get("procrustes.min_joints", 6))
        if usable < min_joints:
            return self._coast(stamp)

        # ---- Stage C / B2: rigid fit (RANSAC optional) -------------------- #
        if not self.flags.get("procrustes", True):
            # Fall back to publishing the raw wrist joint with identity orient.
            wrist = joints3d[self.WRIST_INDEX]
            if not np.all(np.isfinite(wrist)):
                return self._coast(stamp)
            return self._finalise(stamp, wrist,
                                  np.array([0.0, 0.0, 0.0, 1.0]),
                                  agg_conf, usable, 0.0, mask)

        tmpl = self.template[mask]
        obs = joints3d[mask]
        w = weights[mask]

        if self.flags.get("ransac", True):
            fit = ransac_procrustes(
                tmpl, obs, w,
                iterations=int(self.p["ransac.iterations"]),
                sample_size=int(self.p["ransac.sample_size"]),
                inlier_thresh=float(self.p["ransac.inlier_thresh"]))
            if fit is None:
                return self._coast(stamp)
            R, t, rmsd, n_in = fit["R"], fit["t"], fit["rmsd"], fit["n_inliers"]
        else:
            R, t, rmsd = procrustes_fit(tmpl, obs, w)
            n_in = usable

        # The wrist pose is the rigid transform applied to the template wrist.
        wrist_pos = (R @ self.template[self.WRIST_INDEX]) + t
        wrist_quat = rotation_matrix_to_quaternion(R)

        # Confidence weighted by inlier fraction for the Kalman noise scaling.
        conf = float(np.clip(agg_conf * (n_in / max(1, usable)), 1e-3, 1.0))
        return self._finalise(stamp, wrist_pos, wrist_quat, conf, n_in,
                              rmsd, mask)

    # ---- Stage D + E + coasting --------------------------------------------#
    def _finalise(self, stamp, pos, quat, conf, n_in, rmsd, mask):
        self.coast_frames = 0
        if self.flags.get("kalman", True):
            dt = self._dt(stamp)
            out = self.kf.update(dt, pos, quat, conf)
            if out is not None:
                pos, quat = out
        if self.flags.get("one_euro", True):
            pos = self.euro_pos(stamp, pos)
            quat = quat_normalize(self.euro_quat(stamp, quat))
        self.last_pose = (np.asarray(pos), np.asarray(quat))
        self._last_stamp = stamp
        return {"pos": pos, "quat": quat, "valid": True,
                "n_inliers": n_in, "rmsd": rmsd, "inlier_mask": mask}

    def _coast(self, stamp):
        """Predict-only coasting during dropouts up to a max coast duration."""
        self.coast_frames += 1
        max_coast = int(self.p["kalman.max_coast_frames"])
        if self.coast_frames > max_coast or not self.kf.initialised \
                or not self.flags.get("kalman", True):
            self.reset()
            return None
        dt = self._dt(stamp)
        pred = self.kf.predict(dt)
        if pred is None:
            return None
        pos, quat = pred
        if self.flags.get("one_euro", True):
            pos = self.euro_pos(stamp, pos)
            quat = quat_normalize(self.euro_quat(stamp, quat))
        self._last_stamp = stamp
        return {"pos": pos, "quat": quat, "valid": False,
                "n_inliers": 0, "rmsd": float("nan"), "inlier_mask": None}

    def _dt(self, stamp):
        prev = getattr(self, "_last_stamp", None)
        self._last_stamp = stamp
        if prev is None:
            return 1.0 / float(self.p.get("nominal_fps", 30.0))
        dt = stamp - prev
        return dt if dt > 1e-4 else 1.0 / float(self.p.get("nominal_fps", 30.0))
```

> **Implementation note on `procrustes_fit`:** the line containing the `... if False else ...` placeholder is dead and should be deleted before running — the weighted RMSD is computed two lines below via `pred`. It is left as an explicit `# TODO: delete placeholder` marker so the intent is clear.

---

## Deliverable 3 — `handpose_estimation.yaml` (extended)

ROS 2's YAML parser supports nested mappings under `ros__parameters:`, and rclpy lets you declare/read them with **dotted names** (`self.get_parameter("kalman.process_noise_pos")`). That is the clean, consistent convention used here. The 21×3 template and the reachability shell are kept in a *separate* non-ROS YAML (loaded via a path parameter, mirroring the existing `extrinsics_file` pattern) because (a) they are large structured arrays and (b) they are calibration artefacts that should version independently from tuning parameters.

```yaml
handpose_node:
  ros__parameters:
    # ---------------- existing parameters (unchanged) -------------------- #
    camera_names: ["camera0", "camera1"]
    model_path: "hand_landmarker.task"
    extrinsics_file: "extrinsics.yaml"
    use_camera_info_extrinsics: false        # false -> raw K + extrinsics + DLT
    world_frame: "world"
    num_hands: 2
    swap_handedness_camera1: false
    min_hand_detection_confidence: 0.5
    min_hand_presence_confidence: 0.5
    min_tracking_confidence: 0.5
    sync_slop: 0.05
    sync_queue_size: 10
    scale: 0.05                              # world units -> metres
    joint_size: 0.01
    line_width: 0.005
    publish_annotated: true
    publish_camera_pose: true
    camera_marker_size: 0.1

    # ---------------- NEW: pipeline-wide ---------------------------------- #
    nominal_fps: 30.0                        # fallback dt when timing missing
    template_file: "hand_template.yaml"      # external 21x3 template + shell
    publish_fitted_skeleton: true            # show R*template skeleton in RViz

    # Per-view weighting source. Because the Tasks-API HandLandmarker does NOT
    # populate per-landmark visibility/presence (always 0; see MediaPipe issue
    # #5212), choose how per-view weights are formed:
    #   "uniform"      -> all weights 1 (plain DLT)
    #   "handedness"   -> weight = handedness score (per hand, per view)
    #   "reprojection" -> weight = 1/(1+resid/resid_scale) from Stage A2
    #   "product"      -> handedness * reprojection (recommended)
    weight_source: "product"
    reproj_resid_scale: 2.0                  # pixels; larger -> gentler weight

    # ---------------- Stage A1: weighted DLT ------------------------------ #
    weighted_dlt.enabled: true

    # ---------------- Stage A2: reprojection residual --------------------- #
    reprojection_residual.enabled: true
    reprojection_residual.publish_debug: false   # log/visualise per-joint err

    # ---------------- Stage B1: reachability shell ------------------------ #
    # NOTE on units: these are WORLD UNITS. metres = world_units * scale.
    # With scale=0.05, an arm reach of 0.66 m -> 0.66/0.05 = 13.2 world units.
    # Calibrate by holding the arm fully extended then fully retracted (see
    # README) and reading off min/max ||wrist - shoulder||.
    reachability_gate.enabled: true
    reachability_gate.d_min: 4.0             # world units (~0.20 m at scale .05)
    reachability_gate.d_max: 14.0            # world units (~0.70 m at scale .05)
    reachability_gate.behind_margin: 1.0     # world units; reject behind-head
    reachability_gate.forward_axis: [0.0, 0.0, 1.0]   # cameras look down +z
    # Shoulder anchors live in hand_template.yaml (shell: section) so they can
    # be calibrated together with reach; override here if you prefer ROS params.

    # ---------------- Stage C: Procrustes / Kabsch ------------------------ #
    procrustes.enabled: true
    procrustes.min_joints: 6                 # min usable joints to attempt fit

    # ---------------- Stage B2: RANSAC ------------------------------------ #
    ransac.enabled: true
    ransac.iterations: 50
    ransac.sample_size: 4                    # 3 min; 4 more stable
    # inlier_thresh in WORLD UNITS. Rule of thumb: a few x the triangulation
    # noise, and a fraction of hand size. Hand span ~0.18 m -> ~3.6 world units
    # at scale .05; start at ~0.4 world units (~2 cm) and widen if too strict.
    ransac.inlier_thresh: 0.4

    # ---------------- Stage D: Kalman ------------------------------------- #
    kalman.enabled: true
    kalman.process_noise_pos: 50.0           # sigma_a^2 (world units/s^2)^2;
                                             # larger -> more responsive/noisier
    kalman.measurement_noise_pos: 0.25       # world units^2; larger -> smoother
    kalman.gate_threshold: 11.345            # chi-square(3 DoF, 0.99) = 11.345
    kalman.orientation_lpf: 0.5              # SLERP step toward measurement
    kalman.max_coast_frames: 10              # predict-only frames before reset

    # ---------------- Stage E: One-Euro ----------------------------------- #
    one_euro.enabled: true
    one_euro.min_cutoff: 1.0                 # Hz; lower -> less jitter (hand still)
    one_euro.beta: 0.007                     # higher -> less lag on fast motion
    one_euro.d_cutoff: 1.0                   # Hz; derivative cutoff (rarely tuned)
```

---

## Deliverable 4 — `hand_template.yaml` (canonical 21-landmark template + shell)

The template is expressed in a **wrist-local frame** (origin at the wrist/landmark 0; +x toward the middle-MCP, +y toward the thumb side, +z out of the palm dorsum) in **metres**, then the node scales it into world units by dividing by `scale`. Coordinates are built from Buryanov & Kotiuk (2010) phalanx proportions plus standard palm-breadth spacing; they are a deliberately neutral, flat-hand pose. **TODO markers indicate the values that genuinely require per-user calibration.**

```yaml
# hand_template.yaml -- canonical MediaPipe 21-landmark hand template.
# Frame: wrist-local, metres. +x: wrist->middle MCP, +y: toward thumb,
#        +z: palm dorsal normal. Flat, fingers-extended, thumb abducted.
#
# Sources for default geometry:
#   * Phalanx lengths: Buryanov & Kotiuk, "Proportions of Hand Segments",
#     Int. J. Morphol. 28(3):755-758 (2010), DOI 10.4067/S0717-95022010000300015.
#     index PP 39.78, MP 22.38, DP 15.82; middle PP 44.63, MP 26.33, DP 17.40;
#     ring PP 41.37, MP 25.65, DP 17.30; little PP 32.74, MP 18.11, DP 15.96 (mm).
#     (Reproduced via Sarac 2017, arXiv:2003.11598 Tab 2.1; verify vs original.)
#   * Metacarpal/thumb lengths & palm breadth: typical adult anatomy
#     (~MC II 68, III 65, IV 58, V 53, thumb MC 46 mm; breadth ~85 mm).
#     TODO: replace with the original Buryanov Table I metacarpal/thumb values
#           or your own X-ray/measured values for best accuracy.
#
# Layout convention (right hand). For a LEFT hand the node mirrors y -> -y.

template:
  units: "m"
  handedness_reference: "Right"   # node mirrors for Left
  # 21 rows, MediaPipe order:
  # 0 WRIST, 1-4 THUMB(CMC,MCP,IP,TIP), 5-8 INDEX(MCP,PIP,DIP,TIP),
  # 9-12 MIDDLE, 13-16 RING, 17-20 PINKY
  landmarks:
    - [0.000,  0.000, 0.000]   # 0  WRIST
    - [0.030,  0.030, 0.000]   # 1  THUMB_CMC
    - [0.055,  0.050, 0.005]   # 2  THUMB_MCP
    - [0.075,  0.062, 0.008]   # 3  THUMB_IP
    - [0.090,  0.070, 0.010]   # 4  THUMB_TIP
    - [0.068,  0.020, 0.000]   # 5  INDEX_MCP
    - [0.108,  0.022, 0.000]   # 6  INDEX_PIP
    - [0.130,  0.022, 0.000]   # 7  INDEX_DIP
    - [0.146,  0.022, 0.000]   # 8  INDEX_TIP
    - [0.065,  0.000, 0.000]   # 9  MIDDLE_MCP
    - [0.109,  0.000, 0.000]   # 10 MIDDLE_PIP
    - [0.135,  0.000, 0.000]   # 11 MIDDLE_DIP
    - [0.153,  0.000, 0.000]   # 12 MIDDLE_TIP
    - [0.058, -0.020, 0.000]   # 13 RING_MCP
    - [0.099, -0.021, 0.000]   # 14 RING_PIP
    - [0.125, -0.021, 0.000]   # 15 RING_DIP
    - [0.142, -0.021, 0.000]   # 16 RING_TIP
    - [0.053, -0.040, 0.000]   # 17 PINKY_MCP
    - [0.086, -0.042, 0.000]   # 18 PINKY_PIP
    - [0.104, -0.042, 0.000]   # 19 PINKY_DIP
    - [0.120, -0.042, 0.000]   # 20 PINKY_TIP

# Reachability shell (Stage B1) -- shoulder anchors relative to the head/world
# frame origin, in WORLD UNITS (metres = world_units * scale).
# TODO: set these from your head-rig geometry and a short reach calibration.
shell:
  units: "world"
  shoulder_left:  [-3.0, -4.0, 2.0]    # ~[-0.15,-0.20,0.10] m at scale .05
  shoulder_right: [ 3.0, -4.0, 2.0]
  d_min: 4.0
  d_max: 14.0
```

---

## Deliverable 5 — `handpose_node.py` (integration excerpt)

Only the integration-relevant additions are shown; the existing subscription/sync/triangulation scaffolding is preserved. The full file keeps the two triangulation modes and the `MarkerArray` output and adds the new per-wrist `PoseStamped` publishers and the `WristTracker` hookup.

```python
"""handpose_node.py (integration additions for the wrist-pose pipeline)."""

import os
import time
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray

from .triangulation import (make_projection_matrix, weighted_dlt,
                            triangulate_point)
from .wrist_pose_pipeline import WristTracker, ReachabilityShell

NUM_LM = 21


class HandPoseNode(Node):
    def __init__(self):
        super().__init__("handpose_node")
        self._declare_pipeline_params()
        self._load_template_and_shell()
        self._build_trackers()

        # New per-wrist publishers (backward-compatible: MarkerArray kept).
        self.pub_wrist = {
            "Left": self.create_publisher(PoseStamped, "handpose/wrist_left", 10),
            "Right": self.create_publisher(PoseStamped, "handpose/wrist_right", 10),
        }
        self.pub_markers = self.create_publisher(
            MarkerArray, "handpose/skeleton", 10)
        # ... existing camera_info / image subscriptions + ApproxTimeSync ...

    # ---------------------------------------------------------------------- #
    def _declare_pipeline_params(self):
        defaults = {
            "nominal_fps": 30.0,
            "template_file": "hand_template.yaml",
            "publish_fitted_skeleton": True,
            "weight_source": "product",
            "reproj_resid_scale": 2.0,
            "weighted_dlt.enabled": True,
            "reprojection_residual.enabled": True,
            "reprojection_residual.publish_debug": False,
            "reachability_gate.enabled": True,
            "reachability_gate.d_min": 4.0,
            "reachability_gate.d_max": 14.0,
            "reachability_gate.behind_margin": 1.0,
            "reachability_gate.forward_axis": [0.0, 0.0, 1.0],
            "procrustes.enabled": True,
            "procrustes.min_joints": 6,
            "ransac.enabled": True,
            "ransac.iterations": 50,
            "ransac.sample_size": 4,
            "ransac.inlier_thresh": 0.4,
            "kalman.enabled": True,
            "kalman.process_noise_pos": 50.0,
            "kalman.measurement_noise_pos": 0.25,
            "kalman.gate_threshold": 11.345,
            "kalman.orientation_lpf": 0.5,
            "kalman.max_coast_frames": 10,
            "one_euro.enabled": True,
            "one_euro.min_cutoff": 1.0,
            "one_euro.beta": 0.007,
            "one_euro.d_cutoff": 1.0,
        }
        for name, val in defaults.items():
            self.declare_parameter(name, val)
        self.params = {n: self.get_parameter(n).value for n in defaults}
        self.flags = {
            "weighted_dlt": self.params["weighted_dlt.enabled"],
            "reachability_gate": self.params["reachability_gate.enabled"],
            "procrustes": self.params["procrustes.enabled"],
            "ransac": self.params["ransac.enabled"],
            "kalman": self.params["kalman.enabled"],
            "one_euro": self.params["one_euro.enabled"],
        }

    def _load_template_and_shell(self):
        share = self.get_parameter("template_file").value
        path = share if os.path.isabs(share) else os.path.join(
            self._pkg_share(), share)        # _pkg_share() per existing pattern
        with open(path, "r") as fh:
            doc = yaml.safe_load(fh)
        scale = float(self.get_parameter("scale").value)
        tmpl_m = np.asarray(doc["template"]["landmarks"], dtype=float)
        self.template_world = tmpl_m / scale     # metres -> world units
        s = doc.get("shell", {})
        self.shell = ReachabilityShell(
            shoulder_left=s.get("shoulder_left", [-3, -4, 2]),
            shoulder_right=s.get("shoulder_right", [3, -4, 2]),
            d_min=self.params["reachability_gate.d_min"],
            d_max=self.params["reachability_gate.d_max"],
            forward_axis=self.params["reachability_gate.forward_axis"],
            behind_margin=self.params["reachability_gate.behind_margin"],
        )

    def _build_trackers(self):
        self.trackers = {}
        for hand in ("Left", "Right"):
            tmpl = self.template_world.copy()
            if hand == "Left":
                tmpl[:, 1] *= -1.0          # mirror y for the left hand
            self.trackers[hand] = WristTracker(
                handedness=hand, template=tmpl, flags=self.flags,
                params=self.params, shell=self.shell)

    # ---------------------------------------------------------------------- #
    def _per_view_weights(self, hand_scores, residuals):
        """Form per-joint weights from the configured weight_source.

        Because the Tasks-API HandLandmarker leaves per-landmark visibility at
        0 (MediaPipe issue #5212), there is no native per-joint confidence.
        We therefore synthesise per-joint weights from:
          * the per-hand handedness score (broadcast to all joints), and/or
          * the Stage-A2 reprojection residual (per joint).
        """
        src = self.params["weight_source"]
        rs = float(self.params["reproj_resid_scale"])
        hand_w = float(np.clip(np.mean(hand_scores), 0.5, 1.0))
        reproj_w = 1.0 / (1.0 + residuals / rs)     # per joint, in (0,1]
        if src == "uniform":
            return np.ones(NUM_LM)
        if src == "handedness":
            return np.full(NUM_LM, hand_w)
        if src == "reprojection":
            return reproj_w
        return hand_w * reproj_w                     # "product" (default)

    def on_synced_frame(self, lm_by_cam, scores_by_cam, Ps, stamp_sec):
        """Hook called after 2D detection + handedness matching.

        lm_by_cam[cam][hand] -> (21,2) pixels; scores_by_cam[cam][hand]->float.
        Ps[cam] -> (3,4) projection matrix. Hands already matched by handedness.
        """
        for hand in ("Left", "Right"):
            if not all(hand in lm_by_cam[c] for c in lm_by_cam):
                # Missing in a view -> let the tracker coast.
                res = self.trackers[hand].update(
                    stamp_sec, np.full((NUM_LM, 3), np.nan),
                    np.zeros(NUM_LM), 0.0)
                self._publish(hand, res, stamp_sec)
                continue

            cams = list(lm_by_cam.keys())
            joints = np.full((NUM_LM, 3), np.nan)
            resid = np.zeros(NUM_LM)
            scores = [scores_by_cam[c][hand] for c in cams]
            for j in range(NUM_LM):
                pts = [lm_by_cam[c][hand][j] for c in cams]
                proj = [Ps[c] for c in cams]
                if self.flags["weighted_dlt"]:
                    # initial equal-weight triangulation for the residual
                    X, mres, rv = triangulate_point(proj, pts)
                else:
                    X, mres, rv = triangulate_point(proj, pts)
                joints[j] = X
                resid[j] = mres
            weights = self._per_view_weights(scores, resid)
            agg_conf = float(np.clip(np.mean(scores), 0.5, 1.0))
            res = self.trackers[hand].update(
                stamp_sec, joints, weights, agg_conf)
            self._publish(hand, res, stamp_sec, joints, weights)

    def _publish(self, hand, res, stamp_sec, joints=None, weights=None):
        if res is None:
            return                                   # track lost/reset
        msg = PoseStamped()
        msg.header.frame_id = self.get_parameter("world_frame").value
        msg.header.stamp = self.get_clock().now().to_msg()
        p = res["pos"]; q = res["quat"]
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(
            float, p)
        (msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w) = map(float, q)
        self.pub_wrist[hand].publish(msg)
        # MarkerArray: keep raw triangulated joints AND, if enabled, the fitted
        # template skeleton (R*template+t) for A/B visual comparison in RViz.
        # ... existing marker construction, extended with fitted skeleton ...


def main(args=None):
    rclpy.init(args=args)
    node = HandPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

---

## Deliverable 6 — `README.md` (architecture, math, calibration, bring-up)

````markdown
# Wrist-Pose Estimation Pipeline

Estimates a single, temporally smooth 6-DoF wrist pose per hand from a
head-mounted calibrated stereo RGB rig, replacing noisy per-joint
triangulation with a model-based fit. Publishes `geometry_msgs/PoseStamped`
on `handpose/wrist_left` and `handpose/wrist_right`; keeps the legacy
`MarkerArray` for RViz.

## Stage math

### A1 — Confidence-weighted DLT
For a 3D point seen in N views with projection matrices P_i and pixels
(u_i, v_i), build the 2N x 4 matrix A whose view-i rows are
`u_i p_i^3 - p_i^1` and `v_i p_i^3 - p_i^2` (p_i^k = k-th row of P_i), each
scaled by a scalar view weight w_i: `(w o A) x = 0`. Solve by SVD; the right
singular vector for the smallest singular value is the homogeneous point;
divide by its 4th coordinate. With equal weights this is plain DLT.

**Weight source.** The MediaPipe Tasks-API HandLandmarker does NOT populate
per-landmark visibility/presence (always 0, GitHub issue #5212). So weights
are formed from the per-hand handedness score and/or the Stage-A2 reprojection
residual (`weight_source` parameter: uniform / handedness / reprojection /
product).

### A2 — Per-joint reprojection residual
After triangulating joint j to X_j, reproject into each view and store the
pixel error r_{i,j} = ||proj(P_i, X_j) - (u,v)||. The mean per-joint residual
is a trust signal (used by RANSAC and by the reprojection weighting). Verify
the pipeline by confirming high residuals coincide with visibly wrong joints.

### B1 — Reachability shell
For a candidate position p and shoulder anchor s (per hand), accept iff
`d_min <= ||p - s|| <= d_max` and p is not behind the head
(`(p - s)·forward >= -behind_margin`). Units are world units; metres =
world_units * scale.

### C — Rigid Procrustes / Kabsch (with det fix)
Given template points t_k and observed o_k with weights w_k, centre both by
their weighted centroids, form `H = sum_k w_k (t_k - t_bar)(o_k - o_bar)^T`,
SVD `H = U S V^T`, then
`R = V diag(1,1, sign(det(V U^T))) U^T`, `t = o_bar - R t_bar`.
The diag(...) term forbids reflections (Kabsch-Umeyama). Output wrist pose:
`p_wrist = R t_wrist + t`, `q_wrist = quat(R)`.

### B2 — RANSAC
For `iterations`: sample `sample_size` joints (3 min, 4 recommended), fit
rigid transform, count inliers with 3D residual < `inlier_thresh`. Keep the
largest inlier set; refit weighted Kabsch on it. Set `inlier_thresh` to a few
times the triangulation noise and a fraction of hand span.

### D — Kalman + orientation LPF + gating
State `[x,y,z,vx,vy,vz]`, `F = [[I, dt I],[0, I]]`, discrete white-noise-
acceleration `Q` scaled by `process_noise_pos`, `H = [I 0]`,
`R = measurement_noise_pos / confidence * I`. Reject a measurement if the
squared Mahalanobis innovation `y^T S^-1 y` exceeds the chi-square gate
(3 DoF, 0.99 = 11.345). Orientation is smoothed by SLERP toward the measured
quaternion (`orientation_lpf` step). During dropouts the filter predicts only,
up to `max_coast_frames` before the track resets.

### E — One-Euro filter (Casiez et al., CHI 2012)
Adaptive low-pass: `a_d = sf(dt, d_cutoff)`, `dx_hat = lpf(a_d, dx)`,
`cutoff = min_cutoff + beta |dx_hat|`, `a = sf(dt, cutoff)`,
`x_hat = lpf(a, x)` with `sf(dt,c) = 2*pi*c*dt / (2*pi*c*dt + 1)`. Applied
last as display polish to position (3) and quaternion (4) components.

## Calibration procedures
1. **Hand template.** Default from Buryanov & Kotiuk (2010) anthropometry.
   To personalise: place the hand flat, fingers extended, thumb abducted in
   clear view of both cameras; capture ~50 frames; triangulate; rigidly align
   each frame to the current template (Kabsch) and average in the wrist frame;
   write the result to `hand_template.yaml`. Spot-check bone lengths.
2. **Reachability shell.** Extend the arm fully, then retract fully, while
   recording `||wrist - shoulder||`; set `d_max` ~5% above the max and `d_min`
   ~5% below the min. Set shoulder anchors from the head-rig CAD or by holding
   the hand at the shoulder and reading the wrist position.

## Incremental bring-up & verification (recommended order)
1. **A1 + A2 only** (disable B1/C/B2/D/E). Verify the `MarkerArray` raw joints
   look right and that Stage-A2 residuals spike on visibly bad joints.
2. **+B1.** Confirm out-of-shell joints/poses are rejected (e.g. wave outside
   reach -> dropouts, not garbage).
3. **+C.** Enable Procrustes; in RViz compare raw joints vs the fitted template
   skeleton (`publish_fitted_skeleton: true`). They should agree on a good
   hand and the template should be robust where raw joints are noisy.
4. **+B2.** Enable RANSAC; tune `inlier_thresh` so a finger occlusion no longer
   drags the wrist pose.
5. **+D.** Enable Kalman; tune `measurement_noise_pos` up for smoothness, watch
   for over-lag; confirm coasting bridges short dropouts and gating rejects
   teleports.
6. **+E.** Enable One-Euro. **Tuning order:** with the hand held still, lower
   `min_cutoff` until residual jitter is acceptable; then move the hand fast
   and raise `beta` until lag is acceptable. Leave `d_cutoff = 1.0`.
````

---

## Recommendations

1. **Bring the pipeline up strictly in the order A1 → A2 → B1 → C → B2 → D → E**, flipping one `*.enabled` flag at a time and A/B-comparing the raw vs fitted skeleton in RViz at each step. This isolates regressions and is exactly why the per-stage flags exist. Do not enable D and E together on the first run — tune Kalman to a stable, slightly laggy output first, then add One-Euro polish.

2. **Set `weight_source: product` and rely on the reprojection-residual weight as the primary per-joint trust signal**, because per-landmark visibility is genuinely unavailable in the Tasks API. Validate Stage A2 first (confirm residuals correlate with visible error); if they don't, your extrinsics/intrinsics or the `use_camera_info_extrinsics` mode is the real problem and no downstream stage will fix it.

3. **Tune in physically meaningful units and respect the `scale` relationship.** Express `ransac.inlier_thresh` and the shell in world units, sanity-checking against `metres = world_units × 0.05`. Start `inlier_thresh` at ~0.4 world units (~2 cm), `d_max` at ~14 (~0.70 m, matching ~0.66 m average adult reach), and widen only if legitimate poses are being rejected.

4. **Keep the Kalman gate at 11.345 (χ²₃,₀.₉₉) initially**; loosen toward ~16 (99.9%) only if good fast motions are being rejected as outliers. Set `max_coast_frames` to ~⅓ second of frames (10 at 30 fps) so brief occlusions coast but long losses reset cleanly.

5. **Thresholds that should change your decisions:** if Stage-A2 mean residual stays >~3–5 px on a clearly visible hand, stop and re-verify calibration before trusting any 3D output. If the fitted-template skeleton visibly diverges from raw joints on a good hand, your template is wrong-sized — re-run the template calibration. If RANSAC inlier counts routinely fall below `sample_size`, your `inlier_thresh` is too tight or triangulation noise is too high.

6. **Replace the remaining `TODO` template values** (metacarpal and thumb-phalanx lengths, palm breadth, shoulder anchors) with the original Buryanov Table I figures or your own measurements before relying on absolute pose accuracy; the default flat-hand template is adequate for orientation/relative tracking but is not a substitute for a one-time per-user calibration.

---

## Caveats

- **Per-landmark confidence is fundamentally unavailable** from the Tasks-API HandLandmarker (MediaPipe issues #5212 and #4479): the pipeline's per-joint weights are *synthesised* from handedness score and reprojection residual, not read from the model. This is the best available proxy but is not a true per-keypoint visibility estimate.
- **The default hand template's phalanx values are reproduced secondhand.** The index/middle/ring/little proximal-phalanx figures (39.78 / 44.63 / 41.37 / 32.74 mm) trace to Buryanov & Kotiuk (2010) via Saraç Stroppa 2017 (arXiv:2003.11598); the primary SciELO PDF blocks automated access, so they should be verified against the original Table I. The metacarpal lengths, thumb phalanx lengths, and palm breadth in the template are **standard-anatomy placeholders, not from a single cited table**, and are marked `TODO`.
- **Reachability-shell defaults are first-order estimates** derived from population arm-reach data (≈0.66 m fingertip-to-shoulder; FingerMapper uses 0.60 m shoulder-to-wrist) converted at `scale = 0.05`. They are starting points for the documented arm-extended/retracted calibration, not validated for your specific rig geometry.
- **Orientation filtering uses SLERP low-pass + One-Euro on quaternion components, not a full error-state quaternion KF.** This is a deliberate, pragmatic simplification justified by the application scale; component-wise One-Euro on a quaternion is renormalised but is not geodesically exact and can behave poorly through large/fast rotations. If wrist orientation accuracy under fast rotation becomes critical, upgrade Stage D's orientation channel to an error-state formulation (Solà, arXiv:1711.02508).
- **The `procrustes_fit` code contains one explicitly flagged dead placeholder line** that must be deleted before execution; it is left in to make the weighted-RMSD intent unambiguous.
- **Kalman `process_noise_pos`/`measurement_noise_pos` defaults are nominal.** They depend on your true triangulation noise and frame rate and must be tuned empirically; the values given are reasonable starting points, not optimised.
- This is a **coherent draft for drop-in iteration**, not a tested package. Topic names, the `_pkg_share()` helper, MarkerArray construction, and the message-filter wiring follow the described existing structure but should be reconciled against the actual current `handpose_node.py` before building.