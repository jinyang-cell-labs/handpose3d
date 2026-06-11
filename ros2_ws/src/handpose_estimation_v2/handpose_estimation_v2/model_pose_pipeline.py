"""Per-hand orchestrator for the model-based (v2) wrist-pose pipeline.

v2 stage layout (vs. docs/estimation_guide_v1.md):

  A+C  rigid_fit_reprojection -- direct multi-view template fit on 2D
                                 reprojection error (replaces DLT + Procrustes)
  B2   ransac_rigid_reprojection -- RANSAC over joints, voting in pixels
  B1   ReachabilityShell      -- gate applied to the FITTED wrist position
  D    ConstantVelocityKF     -- unchanged (handpose_estimation)
  E    OneEuroFilter          -- unchanged (handpose_estimation)

Temporal coupling: the fit is warm-started from the previous frame's accepted
pose; a cold start falls back to v1's DLT + Procrustes bootstrap (>= 2 views).
With a live track, a hand visible in only ONE view is still fit (monocular
PnP) instead of coasting — toggleable via ``single_view.enabled``.

ROS-free; units are METRES at the tracker boundary. Internally the fit runs
in the projection matrices' world units (template / world_scale), and poses
are converted back to metres for the B1/D/E stages.
"""

from __future__ import annotations

import numpy as np

from handpose_estimation.triangulation import rotation_matrix_to_quaternion
from handpose_estimation.wrist_pose_pipeline import WristTracker

from handpose_estimation_v2.model_fit import (
    bootstrap_pose,
    ransac_rigid_reprojection,
    reprojection_residuals,
    rigid_fit_reprojection,
)


class ModelFitWristTracker(WristTracker):
    """WristTracker whose measurement is a model fit, not triangulated joints.

    Reuses the base class's Stage D (Kalman), Stage E (One-Euro), coasting and
    reset logic via ``_finalise`` / ``_coast``; only the per-frame measurement
    step (``update``) is replaced. ``rmsd`` in the result dict is the fit's
    pixel RMSE (not metres as in v1).
    """

    def __init__(self, handedness, template, flags, params, shell=None,
                 rng_seed=0, world_scale=1.0):
        super().__init__(handedness, template, flags, params, shell=shell,
                         rng_seed=rng_seed)
        # Template in the projection matrices' world units for fitting;
        # self.template (metres) is kept for the fitted-skeleton viz.
        self.world_scale = float(world_scale)
        self.template_world = self.template / self.world_scale
        self.last_fit_world = None  # (R, t) in world units, warm start

    def reset(self):
        super().reset()
        self.last_fit_world = None

    def update(self, stamp, observations, projections, view_weights, agg_conf):
        """Run the v2 pipeline for one frame.

        Args:
            stamp: float seconds (monotonic) for filter timing.
            observations: list of V (21, 2) pixel keypoint arrays, NaN where a
                joint (or the whole view) is unobserved. Points must match the
                projections (i.e. already undistorted/rectified in stereo mode).
            projections: list of V (3, 4) projection matrices.
            view_weights: (V,) per-view confidence (handedness scores).
            agg_conf: float aggregate confidence in (0, 1].

        Returns:
            dict {pos, quat, valid, n_inliers, rmsd, inlier_mask} (rmsd is the
            fit RMSE in PIXELS), or None if the track is reset/lost.
        """
        observations = [np.asarray(o, dtype=float) for o in observations]
        view_weights = np.asarray(view_weights, dtype=float)
        seen = [i for i, o in enumerate(observations)
                if np.any(np.isfinite(o))]
        max_rmse = float(self.p.get("model_fit.max_rmse_px", 25.0))

        if len(seen) == 0:
            return self._coast(stamp)

        if not self.flags.get("model_fit", True):
            # Ablation path: raw DLT + Procrustes (v1's A + C) as the
            # measurement, no LM refinement and no warm start.
            boot = bootstrap_pose(observations, projections,
                                  self.template_world, view_weights)
            if boot is None:
                return self._coast(stamp)
            resid = reprojection_residuals(
                observations, projections, self.template_world, *boot)
            finite = resid[np.isfinite(resid)]
            fit = {"R": boot[0], "t": boot[1], "resid": resid,
                   "rmse_px": float(np.sqrt(np.mean(finite ** 2)))
                   if finite.size else float("inf")}
        else:
            if len(seen) == 1 and not (
                    self.flags.get("single_view", True)
                    and self.last_fit_world is not None):
                return self._coast(stamp)

            # ---- initialisation: warm start, else DLT+Procrustes boot ---- #
            init = self.last_fit_world
            if init is None:
                init = bootstrap_pose(observations, projections,
                                      self.template_world, view_weights)
                if init is None:
                    return self._coast(stamp)
            R0, t0 = init

            fit = self._fit(observations, projections, R0, t0, view_weights)

            # A stale warm start (track jumped while coasting) can trap LM in
            # a bad basin: if the fit looks poor and a cold start is possible,
            # re-bootstrap once and keep the better fit.
            if (fit is None or fit["rmse_px"] > max_rmse) and len(seen) >= 2 \
                    and self.last_fit_world is not None:
                boot = bootstrap_pose(observations, projections,
                                      self.template_world, view_weights)
                if boot is not None:
                    refit = self._fit(observations, projections, boot[0],
                                      boot[1], view_weights)
                    if refit is not None and (
                            fit is None or refit["rmse_px"] < fit["rmse_px"]):
                        fit = refit
            if fit is None:
                return self._coast(stamp)

        # ---- joint inlier accounting (per-joint trust, in pixels) -------- #
        inlier_px = float(self.p.get("ransac.inlier_thresh_px", 8.0))
        worst = np.nanmax(np.where(np.isnan(fit["resid"]), -np.inf,
                                   fit["resid"]), axis=0)
        inlier_mask = (worst >= 0) & (worst < inlier_px)
        n_in = fit.get("n_inliers", int(np.count_nonzero(inlier_mask)))
        observed = np.isfinite(worst) & (worst >= 0)
        n_observed = int(np.count_nonzero(observed))

        min_joints = int(self.p.get("model_fit.min_joints", 6))
        if n_in < min_joints or fit["rmse_px"] > max_rmse:
            return self._coast(stamp)

        R, t_world = fit["R"], fit["t"]
        wrist_pos = (R @ self.template[self.WRIST_INDEX]) \
            + t_world * self.world_scale  # template wrist is the origin
        wrist_quat = rotation_matrix_to_quaternion(R)

        # ---- Stage B1: reachability gate on the fitted wrist ------------- #
        if self.flags.get("reachability_gate", True) and self.shell is not None:
            ok, _reason = self.shell.check(wrist_pos, self.handedness)
            if not ok:
                return self._coast(stamp)

        self.last_fit_world = (R, t_world)
        self.last_fit = (R, t_world * self.world_scale)  # metres, for viz

        # Single-view fits have unconstrained-ish depth; trust them less.
        conf = float(agg_conf) * (n_in / max(1, n_observed))
        if len(seen) == 1:
            conf *= float(self.p.get("single_view.conf_scale", 0.5))
        conf = float(np.clip(conf, 1e-3, 1.0))

        return self._finalise(stamp, wrist_pos, wrist_quat, conf, n_in,
                              fit["rmse_px"], inlier_mask)

    # ------------------------------------------------------------------ fit
    def _fit(self, observations, projections, R0, t0, view_weights):
        huber_px = float(self.p.get("model_fit.huber_px", 5.0))
        max_iters = int(self.p.get("model_fit.max_iters", 20))
        if self.flags.get("ransac", True):
            return ransac_rigid_reprojection(
                observations, projections, self.template_world, R0, t0,
                view_weights=view_weights,
                iterations=int(self.p.get("ransac.iterations", 30)),
                sample_size=int(self.p.get("ransac.sample_size", 4)),
                inlier_thresh_px=float(
                    self.p.get("ransac.inlier_thresh_px", 8.0)),
                huber_px=huber_px,
                rng=self.rng)
        return rigid_fit_reprojection(
            observations, projections, self.template_world, R0, t0,
            view_weights=view_weights, max_iters=max_iters,
            huber_px=huber_px)

    # ---------------------------------------------------------------- debug
    def residuals(self, observations, projections):
        """Pixel residuals of the last accepted fit (for debug logging)."""
        if self.last_fit_world is None:
            return None
        R, t = self.last_fit_world
        return reprojection_residuals(observations, projections,
                                      self.template_world, R, t)
