"""Model-based, temporally coupled 6-DoF wrist pose estimation.

Stages (each individually toggleable via the node's ``*.enabled`` parameters):

  B1  ReachabilityShell    -- distance gate against a calibrated arm-reach shell
  C   procrustes_fit       -- weighted Kabsch/orthogonal-Procrustes (det fix)
  B2  ransac_procrustes    -- RANSAC inlier selection wrapping the Kabsch fit
  D   ConstantVelocityKF   -- CV position KF + SLERP orientation LPF + gating
  E   OneEuroFilter        -- final 6-DoF smoothing (Casiez et al., CHI 2012)

  WristTracker             -- per-hand orchestrator: B1 -> C -> B2 -> D -> E

All lengths are METRES (the node converts triangulated world units to metres
via ``effective_scale`` before calling into this module). All numeric
thresholds are passed in from ROS parameters; nothing is hard-coded except
mathematically fixed constants.

This module is deliberately ROS-free so it can be unit-tested standalone.
"""

from __future__ import annotations

import math

import numpy as np

from handpose_estimation.triangulation import rotation_matrix_to_quaternion


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
    if dot < 0.0:  # take the shorter arc (double-cover fix)
        q1 = -q1
        dot = -dot
    if dot > 0.9995:  # nearly parallel -> linear, then renormalise
        return quat_normalize(q0 + t * (q1 - q0))
    theta_0 = math.acos(min(1.0, dot))
    theta = theta_0 * t
    q2 = quat_normalize(q1 - q0 * dot)
    return q0 * math.cos(theta) + q2 * math.sin(theta)


def quat_angle(q0, q1):
    """Absolute rotation angle (rad) between two quaternions."""
    d = abs(float(np.dot(quat_normalize(q0), quat_normalize(q1))))
    return 2.0 * math.acos(min(1.0, d))


# ===========================================================================#
#  Stage C -- weighted Kabsch / orthogonal Procrustes                         #
# ===========================================================================#
def procrustes_fit(template, observed, weights=None):
    """Closed-form rigid transform mapping ``template`` onto ``observed``.

    Solves the (weighted) orthogonal Procrustes problem with the determinant
    correction that forbids reflections (Kabsch-Umeyama):

        H = sum_k w_k (t_k - t_bar)(o_k - o_bar)^T
        H = U S V^T
        R = V diag(1, 1, sign(det(V U^T))) U^T
        t = o_bar - R t_bar

    Args:
        template: (M, 3) canonical landmark positions, wrist-local frame.
        observed: (M, 3) triangulated landmark positions, world frame.
        weights: optional (M,) non-negative per-landmark weights.

    Returns:
        (R, t, rmsd): (3,3) rotation, (3,) translation, weighted RMSD.
    """
    template = np.asarray(template, dtype=float)
    observed = np.asarray(observed, dtype=float)
    m = template.shape[0]
    if weights is None:
        weights = np.ones(m, dtype=float)
    weights = np.asarray(weights, dtype=float)
    wsum = float(weights.sum())
    if wsum < 1e-12:
        weights = np.ones(m, dtype=float)
        wsum = float(m)

    t_bar = (weights[:, None] * template).sum(axis=0) / wsum
    o_bar = (weights[:, None] * observed).sum(axis=0) / wsum
    tc = template - t_bar
    oc = observed - o_bar

    H = (weights[:, None] * tc).T @ oc  # 3x3 cross-covariance
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T
    d = np.sign(np.linalg.det(V @ U.T))
    D = np.diag([1.0, 1.0, d])  # reflection-prevention fix
    R = V @ D @ U.T
    t = o_bar - R @ t_bar

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
    """Robust rigid fit: RANSAC over the joints around :func:`procrustes_fit`.

    Each iteration samples a minimal subset (``sample_size`` joints), fits a
    rigid transform, then counts inliers by 3D residual < ``inlier_thresh``
    against the transformed template. The best inlier set is refit once with
    confidence weights.

    Args:
        template, observed: (M, 3) arrays (M typically 21).
        weights: optional (M,) confidence weights.
        iterations: RANSAC iteration count.
        sample_size: minimal subset size (3 minimum for a rigid 3D fit;
            4 recommended for stability).
        inlier_thresh: 3D residual threshold in metres.
        rng: optional numpy Generator (pass one for reproducibility).

    Returns:
        dict with keys R, t, rmsd, inliers (bool mask over M), n_inliers;
        or None if no acceptable model is found.
    """
    template = np.asarray(template, dtype=float)
    observed = np.asarray(observed, dtype=float)
    m = template.shape[0]
    if weights is None:
        weights = np.ones(m, dtype=float)
    weights = np.asarray(weights, dtype=float)
    rng = rng if rng is not None else np.random.default_rng()

    valid = np.all(np.isfinite(observed), axis=1)
    valid_idx = np.where(valid)[0]
    if valid_idx.size < sample_size:
        return None

    best_mask = None
    best_count = -1
    for _ in range(int(iterations)):
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
    """Distance gate of candidate joint positions against an arm-reach shell.

    A shoulder anchor is offset from the head/world frame origin for each hand.
    A position is accepted if ``d_min <= ||p - shoulder|| <= d_max`` AND it is
    not "behind the head" (component along the forward axis below
    ``-behind_margin``).

    All parameters come from config (``reachability_gate.*`` + the ``shell:``
    section of hand_template.yaml) so the shell can be calibrated per user.
    Units: metres.
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
        joints = np.asarray(joints, dtype=float)
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
    """6-state ``[x,y,z,vx,vy,vz]`` CV Kalman filter for the wrist position.

    Orientation is handled separately by SLERP low-pass filtering (an
    error-state quaternion KF would be heavier than warranted here).

    Position model:
        F = [[I, dt I], [0, I]]            (constant velocity)
        Q = WNA discretisation scaled by process_noise_pos (sigma_a^2)
        H = [I, 0]                          (measure position only)
        R = measurement_noise_pos * I, inflated by 1/confidence

    Outlier rejection: squared Mahalanobis distance of the innovation,
        d^2 = y^T S^-1 y,    accept if d^2 <= gate (chi-square, 3 DoF).
    """

    def __init__(self, process_noise_pos, measurement_noise_pos,
                 gate=11.345, orientation_lpf=0.5):
        self.q_pos = float(process_noise_pos)      # sigma_a^2
        self.r_pos = float(measurement_noise_pos)
        self.gate = float(gate)                    # chi2(3, 0.99) = 11.345
        self.ori_lpf = float(orientation_lpf)      # SLERP step toward meas.
        self.x = None                              # (6,) state
        self.P = None                              # (6,6) covariance
        self.quat = None                           # filtered orientation
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
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = dt4 / 4.0
            Q[i, i + 3] = Q[i + 3, i] = dt3 / 2.0
            Q[i + 3, i + 3] = dt2
        return Q * self.q_pos

    def initialise(self, position, quat):
        self.x = np.zeros(6)
        self.x[:3] = position
        self.P = np.eye(6)
        self.P[3:, 3:] *= 10.0  # large initial velocity uncertainty
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
        """Full predict+update. Returns (pos, quat); coasts if gated out.

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
        a_d    = smoothing_factor(t_e, d_cutoff)
        dx     = (x - x_prev) / t_e
        dx_hat = exp_smooth(a_d, dx, dx_prev)
        cutoff = min_cutoff + beta * |dx_hat|
        a      = smoothing_factor(t_e, cutoff)
        x_hat  = exp_smooth(a, x, x_prev)
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
    """One-Euro filter over a fixed-length vector (per-component scalars)."""

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
    """Owns the temporal state for ONE hand; runs stages B1 -> C -> B2 -> D -> E.

    The node constructs one WristTracker per handedness label ("Left"/"Right")
    and feeds it the per-frame triangulated joints (metres), per-joint weights
    and aggregate confidence. Each stage can be disabled via the ``flags`` dict.
    """

    WRIST_INDEX = 0  # MediaPipe landmark 0 is the wrist

    def __init__(self, handedness, template, flags, params, shell=None,
                 rng_seed=0):
        self.handedness = handedness
        self.template = np.asarray(template, dtype=float)  # (21,3) wrist-local
        self.flags = flags                                  # per-stage enables
        self.p = params                                     # flat param dict
        self.shell = shell                                  # ReachabilityShell
        self.rng = np.random.default_rng(rng_seed)          # for RANSAC

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
        self.last_pose = None       # (pos, quat) last good output
        self.last_fit = None        # (R, t) of the last accepted rigid fit
        self._last_stamp = None

    def reset(self):
        self.kf.reset()
        self.euro_pos.reset()
        self.euro_quat.reset()
        self.coast_frames = 0
        self.last_pose = None
        self.last_fit = None
        self._last_stamp = None

    def update(self, stamp, joints3d, weights, agg_conf):
        """Run the pipeline for one frame.

        Args:
            stamp: float seconds (monotonic) for filter timing.
            joints3d: (21, 3) triangulated joints in METRES (NaN if bad).
            weights: (21,) per-joint confidence weights (from Stage A1/A2).
            agg_conf: float aggregate confidence in (0,1] (e.g. handedness
                score times inlier fraction); scales Kalman measurement noise.

        Returns:
            dict {pos, quat, valid, n_inliers, rmsd, inlier_mask} (coasting
            results have valid=False), or None if the track is reset/lost.
        """
        joints3d = np.asarray(joints3d, dtype=float)
        weights = np.asarray(weights, dtype=float)

        # ---- Stage B1: reachability gate ---------------------------------- #
        mask = np.all(np.isfinite(joints3d), axis=1)
        if self.flags.get("reachability_gate", True) and self.shell is not None:
            mask = mask & self.shell.filter_joints(joints3d, self.handedness)

        usable = int(np.count_nonzero(mask))
        min_joints = int(self.p.get("procrustes.min_joints", 6))
        if usable < min_joints:
            return self._coast(stamp)

        # ---- Stage C / B2: rigid fit (RANSAC optional) -------------------- #
        if not self.flags.get("procrustes", True):
            # Fall back to publishing the raw wrist joint with identity orient.
            wrist = joints3d[self.WRIST_INDEX]
            if not np.all(np.isfinite(wrist)) or not mask[self.WRIST_INDEX]:
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
                inlier_thresh=float(self.p["ransac.inlier_thresh"]),
                rng=self.rng)
            if fit is None:
                return self._coast(stamp)
            R, t, rmsd, n_in = fit["R"], fit["t"], fit["rmsd"], fit["n_inliers"]
        else:
            R, t, rmsd = procrustes_fit(tmpl, obs, w)
            n_in = usable

        # The wrist pose is the rigid transform applied to the template wrist.
        wrist_pos = (R @ self.template[self.WRIST_INDEX]) + t
        wrist_quat = rotation_matrix_to_quaternion(R)
        self.last_fit = (R, t)

        # Confidence weighted by inlier fraction for the Kalman noise scaling.
        conf = float(np.clip(agg_conf * (n_in / max(1, usable)), 1e-3, 1.0))
        return self._finalise(stamp, wrist_pos, wrist_quat, conf, n_in,
                              rmsd, mask)

    # ---- Stage D + E + coasting ------------------------------------------ #
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
        self._last_stamp = stamp
        self.last_pose = (np.asarray(pos), np.asarray(quat))
        return {"pos": np.asarray(pos), "quat": np.asarray(quat),
                "valid": True, "n_inliers": n_in, "rmsd": rmsd,
                "inlier_mask": mask}

    def _coast(self, stamp):
        """Predict-only coasting during dropouts, up to a max coast duration."""
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
        return {"pos": np.asarray(pos), "quat": np.asarray(quat),
                "valid": False, "n_inliers": 0, "rmsd": float("nan"),
                "inlier_mask": None}

    def _dt(self, stamp):
        prev = self._last_stamp
        nominal = 1.0 / float(self.p.get("nominal_fps", 30.0))
        if prev is None:
            return nominal
        dt = stamp - prev
        return dt if dt > 1e-4 else nominal
