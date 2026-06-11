"""Direct multi-view rigid fit of the hand template on 2D reprojection error.

This module replaces stages A (per-joint DLT triangulation) and C (3D
Procrustes) of handpose_estimation with a single optimisation: estimate the
6-DoF pose (R, t) of the rigid 21-landmark template by minimising the robust,
confidence-weighted pixel reprojection error of every template joint in every
view that observed it,

    E(R, t) = sum_i sum_j  w_i * rho_huber( || pi(P_i, R m_j + t) - x_ij || )

where P_i is view i's (3, 4) projection matrix, m_j the template joint in the
wrist-local frame and x_ij the detected 2D keypoint. Because the template's
bone lengths and scale enter the optimisation directly, depth is constrained
by hand structure instead of relying purely on two-ray intersection — and the
fit degenerates gracefully to monocular PnP when only one view sees the hand
(given a pose prior to initialise from).

Solver: Levenberg-Marquardt on SE(3) with a left-multiplied so(3) perturbation
(R <- exp([dtheta]x) R, t <- t + dt) and iteratively reweighted Huber loss.

Outlier handling (Stage B2 analog): :func:`ransac_rigid_reprojection` wraps
the fit with RANSAC over joints; inliers are voted on the max per-view pixel
residual rather than a 3D distance.

Initialisation: :func:`bootstrap_pose` reuses v1's weighted DLT + Procrustes
as a cold-start when no previous pose is available (requires >= 2 views).

This module is deliberately ROS-free so it can be unit-tested standalone.
Units: template and t are in the WORLD units of the projection matrices
(metres in stereo-rectified mode); residuals are in pixels.
"""

from __future__ import annotations

import numpy as np

from handpose_estimation.triangulation import weighted_dlt
from handpose_estimation.wrist_pose_pipeline import procrustes_fit


# ===========================================================================#
#  SO(3) helpers                                                              #
# ===========================================================================#
def skew(w):
    """(3,) vector -> (3, 3) cross-product matrix [w]x."""
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])


def so3_exp(w):
    """Rodrigues: so(3) vector -> rotation matrix."""
    w = np.asarray(w, dtype=float)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + skew(w)  # first-order approximation
    K = skew(w / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


# ===========================================================================#
#  Projection / residuals                                                     #
# ===========================================================================#
def project_points(P, X):
    """Project (M, 3) world points with a (3, 4) projection matrix.

    Returns:
        uv: (M, 2) pixel coordinates (NaN where depth is degenerate).
        depth: (M,) the homogeneous scale ``P[2] . [X 1]`` (sign tells
            front/behind for P = K [R|t] with det(K) > 0).
    """
    P = np.asarray(P, dtype=float)
    X = np.asarray(X, dtype=float)
    Xh = np.hstack([X, np.ones((X.shape[0], 1))])
    x = Xh @ P.T  # (M, 3)
    depth = x[:, 2]
    uv = np.full((X.shape[0], 2), np.nan)
    ok = np.abs(depth) > 1e-9
    uv[ok] = x[ok, :2] / depth[ok, None]
    return uv, depth


def reprojection_residuals(observations, projections, template, R, t):
    """Per-view, per-joint pixel residual norms for a template pose.

    Args:
        observations: list of V (M, 2) pixel arrays (NaN where unobserved).
        projections: list of V (3, 4) projection matrices.
        template: (M, 3) wrist-local template joints.
        R, t: rigid pose mapping template -> world.

    Returns:
        (V, M) residual norms; NaN where a joint is unobserved in that view
        or projects behind the camera.
    """
    template = np.asarray(template, dtype=float)
    X = (np.asarray(R) @ template.T).T + np.asarray(t)
    out = np.full((len(projections), template.shape[0]), np.nan)
    for i, (P, obs) in enumerate(zip(projections, observations)):
        obs = np.asarray(obs, dtype=float)
        uv, depth = project_points(P, X)
        valid = np.all(np.isfinite(obs), axis=1) & (depth > 1e-9)
        d = np.linalg.norm(uv - obs, axis=1)
        out[i, valid] = d[valid]
    return out


def _huber_row_weights(resid_norm, delta):
    """IRLS weight per observation: 1 inside the Huber radius, delta/e outside."""
    e = np.maximum(resid_norm, 1e-12)
    return np.where(e <= delta, 1.0, delta / e)


def _huber_cost(resid_norm, delta, conf):
    """Robust objective: conf-weighted Huber rho of the residual norms."""
    e = resid_norm
    rho = np.where(e <= delta, e * e, delta * (2.0 * e - delta))
    return float(np.sum(conf * rho))


def rigid_fit_reprojection(observations, projections, template, R0, t0,
                           view_weights=None, joint_mask=None,
                           max_iters=20, huber_px=5.0, lm_lambda0=1e-3,
                           tol=1e-8, min_obs=4):
    """Levenberg-Marquardt rigid fit of the template on reprojection error.

    Args:
        observations: list of V (M, 2) pixel arrays (NaN where unobserved).
        projections: list of V (3, 4) projection matrices.
        template: (M, 3) wrist-local template joints (world units).
        R0, t0: initial pose (from the previous frame or bootstrap_pose).
        view_weights: optional (V,) per-view confidence weights.
        joint_mask: optional (M,) bool mask restricting which joints
            contribute residuals (used by RANSAC).
        max_iters: outer LM iterations.
        huber_px: Huber radius in pixels (robust kernel knee).
        lm_lambda0: initial LM damping.
        tol: stop when the update step norm falls below this.
        min_obs: minimum joint-view observations required to attempt a fit.

    Returns:
        dict with keys R, t, resid ((V, M) pixel norms over ALL joints),
        rmse_px (over the fitted observations), n_obs, converged;
        or None if there are not enough observations.
    """
    template = np.asarray(template, dtype=float)
    V, M = len(projections), template.shape[0]
    if view_weights is None:
        view_weights = np.ones(V, dtype=float)
    view_weights = np.asarray(view_weights, dtype=float)
    if joint_mask is None:
        joint_mask = np.ones(M, dtype=bool)

    # Flatten the valid (view, joint) observations once.
    obs_view, obs_joint, obs_px, obs_conf = [], [], [], []
    for i in range(V):
        obs = np.asarray(observations[i], dtype=float)
        valid = np.all(np.isfinite(obs), axis=1) & joint_mask
        for j in np.where(valid)[0]:
            obs_view.append(i)
            obs_joint.append(j)
            obs_px.append(obs[j])
            obs_conf.append(max(view_weights[i], 1e-6))
    n_obs = len(obs_px)
    if n_obs < min_obs:
        return None
    obs_view = np.asarray(obs_view)
    obs_joint = np.asarray(obs_joint)
    obs_px = np.asarray(obs_px, dtype=float)
    obs_conf = np.asarray(obs_conf, dtype=float)
    Ps = [np.asarray(P, dtype=float) for P in projections]

    def residual_norms(R, t):
        """(n_obs,) pixel residual norms + the raw (n_obs, 2) residuals."""
        X = (R @ template.T).T + t
        r = np.full((n_obs, 2), np.nan)
        for i in range(V):
            sel = obs_view == i
            if not np.any(sel):
                continue
            uv, depth = project_points(Ps[i], X[obs_joint[sel]])
            sub = uv - obs_px[sel]
            # Behind-camera projections poison the linearisation; inflate
            # their residual so LM steps away from such poses.
            sub[depth <= 1e-9] = 1e6
            r[sel] = sub
        return np.linalg.norm(r, axis=1), r

    R = np.asarray(R0, dtype=float).copy()
    t = np.asarray(t0, dtype=float).copy()
    lam = float(lm_lambda0)
    converged = False

    e, r = residual_norms(R, t)
    cost = _huber_cost(e, huber_px, obs_conf)

    for _ in range(int(max_iters)):
        # IRLS row weights: confidence x Huber, shared by a joint's u/v rows.
        w_row = obs_conf * _huber_row_weights(e, huber_px)

        # Jacobian: d(residual)/d[dtheta dt] per observation (2 x 6).
        X = (R @ template.T).T + t
        Rm = (R @ template.T).T  # X - t
        J = np.zeros((2 * n_obs, 6))
        rvec = r.reshape(-1)
        wvec = np.repeat(w_row, 2)
        for k in range(n_obs):
            P = Ps[obs_view[k]]
            Xk = X[obs_joint[k]]
            Xh = np.append(Xk, 1.0)
            a, b, c = P[0] @ Xh, P[1] @ Xh, P[2] @ Xh
            if c <= 1e-9:
                continue  # behind camera: no useful gradient
            du_dX = (P[0, :3] * c - a * P[2, :3]) / (c * c)
            dv_dX = (P[1, :3] * c - b * P[2, :3]) / (c * c)
            dX = np.hstack([-skew(Rm[obs_joint[k]]), np.eye(3)])  # (3, 6)
            J[2 * k] = du_dX @ dX
            J[2 * k + 1] = dv_dX @ dX

        JtW = J.T * wvec
        H = JtW @ J
        g = JtW @ rvec

        # Marquardt damping on the diagonal; retry with more damping on a
        # rejected step instead of recomputing the Jacobian.
        accepted = False
        for _try in range(8):
            D = H + lam * np.diag(np.maximum(np.diag(H), 1e-12))
            try:
                delta = np.linalg.solve(D, -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            R_new = so3_exp(delta[:3]) @ R
            t_new = t + delta[3:]
            e_new, r_new = residual_norms(R_new, t_new)
            cost_new = _huber_cost(e_new, huber_px, obs_conf)
            if cost_new < cost:
                R, t, e, r, cost = R_new, t_new, e_new, r_new, cost_new
                lam = max(lam * 0.3, 1e-9)
                accepted = True
                break
            lam *= 10.0
        if not accepted:
            break
        if float(np.linalg.norm(delta)) < tol:
            converged = True
            break

    resid = reprojection_residuals(observations, projections, template, R, t)
    rmse = float(np.sqrt(np.mean(e[e < 1e5] ** 2))) if np.any(e < 1e5) else float("inf")
    return {"R": R, "t": t, "resid": resid, "rmse_px": rmse,
            "n_obs": n_obs, "converged": converged}


# ===========================================================================#
#  RANSAC over joints (Stage B2 analog, voting in pixels)                     #
# ===========================================================================#
def ransac_rigid_reprojection(observations, projections, template, R0, t0,
                              view_weights=None, iterations=30, sample_size=4,
                              inlier_thresh_px=8.0, huber_px=5.0,
                              sample_iters=8, rng=None):
    """RANSAC over joints wrapping :func:`rigid_fit_reprojection`.

    Each iteration fits the pose to a minimal joint subset (warm-started from
    R0/t0), then votes every joint by its WORST per-view pixel residual.
    The best inlier set is refit once with the full robust kernel.

    Returns dict with R, t, resid, rmse_px, inliers ((M,) bool), n_inliers;
    or None if no acceptable model is found.
    """
    template = np.asarray(template, dtype=float)
    M = template.shape[0]
    rng = rng if rng is not None else np.random.default_rng()

    observed_any = np.zeros(M, dtype=bool)
    for obs in observations:
        observed_any |= np.all(np.isfinite(np.asarray(obs, dtype=float)), axis=1)
    cand = np.where(observed_any)[0]
    if cand.size < sample_size:
        return None

    best_mask, best_count = None, -1
    for _ in range(int(iterations)):
        sample = rng.choice(cand, size=sample_size, replace=False)
        mask = np.zeros(M, dtype=bool)
        mask[sample] = True
        fit = rigid_fit_reprojection(
            observations, projections, template, R0, t0,
            view_weights=view_weights, joint_mask=mask,
            max_iters=sample_iters, huber_px=huber_px, min_obs=sample_size)
        if fit is None:
            continue
        worst = np.nanmax(np.where(np.isnan(fit["resid"]), -np.inf,
                                   fit["resid"]), axis=0)
        inliers = observed_any & (worst < inlier_thresh_px) & (worst >= 0)
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count, best_mask = count, inliers

    if best_mask is None or best_count < sample_size:
        return None

    fit = rigid_fit_reprojection(
        observations, projections, template, R0, t0,
        view_weights=view_weights, joint_mask=best_mask, huber_px=huber_px)
    if fit is None:
        return None
    fit["inliers"] = best_mask
    fit["n_inliers"] = best_count
    return fit


# ===========================================================================#
#  Cold-start initialisation (v1's A + C as a bootstrap)                      #
# ===========================================================================#
def bootstrap_pose(observations, projections, template, view_weights=None,
                   min_joints=4):
    """Initial (R0, t0) via weighted-DLT triangulation + Procrustes.

    Requires each bootstrap joint to be visible in >= 2 views (DLT needs two
    rays), so a cold start is impossible from a single view — by design the
    monocular path only continues an existing track.

    Returns (R0, t0) or None.
    """
    template = np.asarray(template, dtype=float)
    M = template.shape[0]
    obs = [np.asarray(o, dtype=float) for o in observations]
    finite = np.stack([np.all(np.isfinite(o), axis=1) for o in obs])  # (V, M)
    both = finite.sum(axis=0) >= 2
    if int(np.count_nonzero(both)) < min_joints:
        return None

    pts = np.full((M, 3), np.nan)
    for j in np.where(both)[0]:
        views = np.where(finite[:, j])[0]
        Ps = [projections[i] for i in views]
        xs = [obs[i][j] for i in views]
        ws = None
        if view_weights is not None:
            ws = np.asarray(view_weights, dtype=float)[views]
        pts[j] = weighted_dlt(Ps, xs, ws)

    ok = np.all(np.isfinite(pts), axis=1)
    if int(np.count_nonzero(ok)) < min_joints:
        return None
    R0, t0, _ = procrustes_fit(template[ok], pts[ok])
    return R0, t0
