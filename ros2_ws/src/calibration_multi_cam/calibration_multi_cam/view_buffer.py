"""Keep-most-informative view retention (maximin diversity thinning).

When a bounded buffer is full and a new view arrives, we drop whichever view is
*most redundant* — the one whose nearest neighbour (in an appearance-feature
space) is closest — so the retained set stays spread out and diverse. This is a
lightweight stand-in for kalibr's information-gain view selection: instead of
storing every frame, we keep a fixed-size, maximally-varied subset.

A view's "appearance feature" summarizes where/how the board was seen:
normalized centroid (where in the image), spread (scale / distance), and the
x-y correlation (tilt). Diversity in that space ≈ diverse calibration geometry.
"""
from __future__ import annotations

import numpy as np


def corner_features(pids, pts, width=None, height=None):
    """5-D appearance fingerprint of one camera's detection:
    [mean_x, mean_y, std_x, std_y, xy_corr], normalized by image size."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.zeros(5)
    w = float(width) if width else 1.0
    h = float(height) if height else 1.0
    nx = pts[:, 0] / w
    ny = pts[:, 1] / h
    mx, my = float(nx.mean()), float(ny.mean())
    sx, sy = float(nx.std()), float(ny.std())
    corr = float(np.mean((nx - mx) * (ny - my)) / (sx * sy)) if sx > 1e-9 and sy > 1e-9 else 0.0
    return np.array([mx, my, sx, sy, corr], dtype=np.float64)


def multicam_view_features(detections, camera_names, resolutions=None):
    """Per-camera [presence, *corner_features] concatenated over all cameras
    (length 6 * num_cameras). Captures which cameras saw the board AND how it
    appeared in each, so diversity covers both overlap pattern and geometry."""
    parts = []
    for name in camera_names:
        if name in detections:
            pids, pts = detections[name]
            res = resolutions.get(name) if resolutions else None
            w = res[0] if res else None
            h = res[1] if res else None
            parts.append(1.0)
            parts.extend(corner_features(pids, pts, w, h).tolist())
        else:
            parts.append(0.0)
            parts.extend([0.0] * 5)
    return np.array(parts, dtype=np.float64)


def most_redundant_index(feats, cand_feat):
    """Index of the member to drop from (feats + [cand_feat]) to best preserve
    diversity (maximize the set's minimum pairwise spacing).

    The drop must be one of the two endpoints of the globally closest pair —
    removing anything else would leave that pair and not raise the minimum
    spacing. Between the two endpoints we drop whichever leaves the larger
    minimum spacing (the more redundant one). Returns an index in
    [0, len(feats)]; if it equals len(feats) the candidate itself is dropped
    (i.e. rejected), otherwise feats[idx] is evicted."""
    F = np.vstack([np.asarray(f, dtype=np.float64) for f in feats]
                  + [np.asarray(cand_feat, dtype=np.float64)])
    n = F.shape[0]
    if n == 1:
        return 0
    diff = F[:, None, :] - F[None, :, :]
    D = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(D, np.inf)
    a, b = np.unravel_index(int(np.argmin(D)), D.shape)  # globally closest pair

    def min_spacing_without(idx):
        keep = [k for k in range(n) if k != idx]
        return float(D[np.ix_(keep, keep)].min())

    return int(a) if min_spacing_without(a) >= min_spacing_without(b) else int(b)


class MaximinViewBuffer:
    """Fixed-capacity buffer that retains a maximally-diverse subset of views.

    capacity <= 0 (or None) means unbounded (no thinning).
    """

    def __init__(self, capacity=None):
        self.capacity = int(capacity) if (capacity and capacity > 0) else None
        self.items = []
        self.feats = []

    def __len__(self):
        return len(self.items)

    def add(self, item, feat):
        """Insert (item, feat). Returns True if stored, False if rejected as the
        most-redundant of the full set."""
        feat = np.asarray(feat, dtype=np.float64)
        if self.capacity is None or len(self.items) < self.capacity:
            self.items.append(item)
            self.feats.append(feat)
            return True
        idx = most_redundant_index(self.feats, feat)
        if idx == len(self.feats):     # candidate is the most redundant -> reject
            return False
        self.items[idx] = item         # evict the redundant stored view
        self.feats[idx] = feat
        return True
