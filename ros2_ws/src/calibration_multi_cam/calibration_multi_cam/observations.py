"""Storage for synchronized multi-camera AprilGrid observations.

A *view* is one synchronized capture instant. It stores, per camera that saw
the target at that instant, the observed corner ``point_ids`` (indices into the
target's object points) and the matching subpixel ``image_points``.

This is the equivalent of kalibr's ObservationDatabase
(third_party/kalibr/.../kalibr_camera_calibration/ObsDb.py), trimmed to what
the OpenCV + scipy pipeline needs.
"""
from __future__ import annotations

import numpy as np


class ObservationDatabase:
    def __init__(self, camera_names):
        self.camera_names = list(camera_names)
        # views[i] : {camera_name: (point_ids (M,) int, image_points (M,2) float32)}
        self.views = []
        self.timestamps = []  # float seconds, one per view

    # ------------------------------------------------------------------ #
    # Building
    # ------------------------------------------------------------------ #
    def add_view(self, timestamp, detections):
        """Append a view. `detections` maps camera_name -> (point_ids, image_points)."""
        clean = {}
        for cam, (pids, pts) in detections.items():
            pids = np.asarray(pids, dtype=np.int64)
            pts = np.asarray(pts, dtype=np.float32)
            if pids.size == 0:
                continue
            clean[cam] = (pids, pts)
        if clean:
            self.views.append(clean)
            self.timestamps.append(float(timestamp))

    @property
    def num_views(self):
        return len(self.views)

    # ------------------------------------------------------------------ #
    # Queries used by the solver / status reporting
    # ------------------------------------------------------------------ #
    def per_camera_view_count(self):
        counts = {n: 0 for n in self.camera_names}
        for v in self.views:
            for cam in v:
                counts[cam] = counts.get(cam, 0) + 1
        return counts

    def pair_coobservation_count(self):
        """For every camera pair, how many views both cameras saw the target."""
        counts = {}
        for v in self.views:
            seen = [n for n in self.camera_names if n in v]
            for i in range(len(seen)):
                for j in range(i + 1, len(seen)):
                    key = tuple(sorted((seen[i], seen[j])))
                    counts[key] = counts.get(key, 0) + 1
        return counts

    def camera_observations(self, cam_name):
        """All (point_ids, image_points) tuples for a single camera (intrinsics)."""
        return [v[cam_name] for v in self.views if cam_name in v]

    def is_connected(self):
        """True if all cameras are linked through shared target co-observations."""
        edges = self.pair_coobservation_count()
        adj = {n: set() for n in self.camera_names}
        for (a, b), n in edges.items():
            if n > 0:
                adj[a].add(b)
                adj[b].add(a)
        if not self.camera_names:
            return False
        seen = set()
        stack = [self.camera_names[0]]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adj[node] - seen)
        return len(seen) == len(self.camera_names)

    # ------------------------------------------------------------------ #
    # Persistence (numpy, pickled object arrays for ragged data)
    # ------------------------------------------------------------------ #
    def save(self, path, meta=None):
        np.savez(
            path,
            camera_names=np.array(self.camera_names, dtype=object),
            timestamps=np.array(self.timestamps, dtype=np.float64),
            views=np.array(self.views, dtype=object),
            meta=np.array(meta if meta is not None else {}, dtype=object),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        db = cls([str(n) for n in data["camera_names"]])
        db.timestamps = [float(t) for t in data["timestamps"]]
        db.views = list(data["views"])
        meta = data["meta"].item() if data["meta"].shape == () else {}
        return db, meta
