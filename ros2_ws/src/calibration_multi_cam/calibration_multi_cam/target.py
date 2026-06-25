"""AprilGrid calibration target: 3D geometry + corner detection.

The 3D corner layout reproduces ethz-asl/kalibr's GridCalibrationTargetAprilgrid
*exactly*, so a board printed with kalibr_create_target_pdf is metrically
identical and the tag/corner indexing matches.

Reference:
  third_party/kalibr/aslam_cv/aslam_cameras_april/src/GridCalibrationTargetAprilgrid.cpp

Layout (2x2 tags shown). Tag id increases +x first, then +y, from the
bottom-left. A corner's point index is ``row * cols + col`` with
``cols = 2*tag_cols`` and ``rows = 2*tag_rows`` (4 corners per tag)::

      12----13  14----15
      | TAG2 |  | TAG3 |
      8-----9   10----11
      4-----5   6-----7
  y   | TAG0 |  | TAG1 |
  ^   0-----1   2-----3
  +-->x

For a tag, the four corners map to point indices [BL, BR, TR, TL] =
[base, base+1, base+2*tag_cols+1, base+2*tag_cols], where
base = (tag_id // tag_cols) * 4*tag_cols + (tag_id % tag_cols) * 2.

`tag_spacing` is the gap-to-size *ratio*: the gap between adjacent tags is
``tag_spacing * tag_size`` metres.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 always present at runtime
    cv2 = None

_FAMILY_TO_DICT = {
    "36h11": "DICT_APRILTAG_36h11",
    "25h9": "DICT_APRILTAG_25h9",
    "16h5": "DICT_APRILTAG_16h5",
}

# OpenCV aruco returns a marker's four corners clockwise from the marker's
# canonical top-left: [TL, TR, BR, BL]. Kalibr indexes a tag's points as
# [BL, BR, TR, TL]. This maps aruco corner slot -> kalibr corner slot:
#   aruco TL(0)->kalibr TL(3), TR(1)->TR(2), BR(2)->BR(1), BL(3)->BL(0)
# The mapping is fixed (AprilTag decoding makes corner order view-invariant).
# If a calibration shows large/structured reprojection residuals, this is the
# first thing to re-validate against the physical board.
_ARUCO_TO_KALIBR = (3, 2, 1, 0)


class AprilGridTarget:
    """A kalibr-compatible AprilGrid: 3D object points + a corner detector."""

    def __init__(self, tag_rows, tag_cols, tag_size, tag_spacing, family="36h11"):
        if int(tag_rows) < 1 or int(tag_cols) < 1:
            raise ValueError("tag_rows and tag_cols must be >= 1")
        if float(tag_size) <= 0.0 or float(tag_spacing) <= 0.0:
            raise ValueError("tag_size and tag_spacing must be positive")

        self.tag_rows = int(tag_rows)
        self.tag_cols = int(tag_cols)
        self.tag_size = float(tag_size)
        self.tag_spacing = float(tag_spacing)
        self.family = str(family)

        self.rows = 2 * self.tag_rows                 # corner rows
        self.cols = 2 * self.tag_cols                 # corner cols
        self.num_tags = self.tag_rows * self.tag_cols
        self.num_points = self.rows * self.cols       # 4 corners per tag

        self.object_points = self._build_object_points()  # (num_points, 3)
        self._detector = None  # built lazily on first detect()

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def _build_object_points(self):
        pts = np.zeros((self.num_points, 3), dtype=np.float64)
        step = (1.0 + self.tag_spacing) * self.tag_size
        for r in range(self.rows):
            for c in range(self.cols):
                x = (c // 2) * step + (c % 2) * self.tag_size
                y = (r // 2) * step + (r % 2) * self.tag_size
                pts[r * self.cols + c] = (x, y, 0.0)
        return pts

    def tag_point_indices(self, tag_id):
        """Return the four point indices [BL, BR, TR, TL] of a tag."""
        base = (tag_id // self.tag_cols) * self.cols * 2 + (tag_id % self.tag_cols) * 2
        return (base, base + 1, base + self.cols + 1, base + self.cols)

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #
    def _ensure_detector(self):
        if self._detector is not None:
            return
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for AprilGrid detection")
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "cv2.aruco is unavailable. Install opencv with contrib "
                "(python3-opencv on Ubuntu 24.04, or `pip install opencv-contrib-python`)."
            )
        dict_name = _FAMILY_TO_DICT.get(self.family)
        if dict_name is None or not hasattr(cv2.aruco, dict_name):
            raise RuntimeError(
                f"AprilTag family '{self.family}' is not available in this OpenCV build."
            )
        dict_id = getattr(cv2.aruco, dict_name)
        refine = getattr(
            cv2.aruco, "CORNER_REFINE_APRILTAG",
            getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 0),
        )
        # OpenCV >= 4.7 exposes the ArucoDetector class; older builds use the
        # functional API. Support both.
        if hasattr(cv2.aruco, "ArucoDetector"):
            dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = refine
            self._detector = ("new", cv2.aruco.ArucoDetector(dictionary, params))
        else:  # pragma: no cover - legacy OpenCV path
            dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters_create()
            params.cornerRefinementMethod = refine
            self._detector = ("old", (dictionary, params))

    def detect(self, gray):
        """Detect target corners in a grayscale image.

        Returns ``(point_ids, image_points)`` where ``point_ids`` is an (M,)
        int array indexing into ``self.object_points`` and ``image_points`` is
        the matching (M, 2) float32 array of subpixel corner locations.
        """
        self._ensure_detector()
        mode, det = self._detector
        if mode == "new":
            corners, ids, _ = det.detectMarkers(gray)
        else:  # pragma: no cover - legacy OpenCV path
            dictionary, params = det
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

        if ids is None or len(ids) == 0:
            return (np.empty((0,), dtype=np.int64),
                    np.empty((0, 2), dtype=np.float32))

        point_ids = []
        image_points = []
        for quad, tag_id in zip(corners, ids.flatten()):
            tag_id = int(tag_id)
            if tag_id < 0 or tag_id >= self.num_tags:
                continue  # a tag from a different board / spurious id
            pidx = self.tag_point_indices(tag_id)  # [BL, BR, TR, TL]
            q = np.asarray(quad, dtype=np.float32).reshape(4, 2)  # aruco [TL,TR,BR,BL]
            for slot in range(4):
                point_ids.append(pidx[_ARUCO_TO_KALIBR[slot]])
                image_points.append(q[slot])

        return (np.asarray(point_ids, dtype=np.int64),
                np.asarray(image_points, dtype=np.float32))

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_params(cls, params):
        """Build from a dict with keys family/tag_rows/tag_cols/tag_size/tag_spacing."""
        ttype = str(params.get("type", "aprilgrid")).lower()
        if ttype != "aprilgrid":
            raise NotImplementedError(
                f"target.type '{ttype}' is not supported yet (only 'aprilgrid')."
            )
        return cls(
            tag_rows=params["tag_rows"],
            tag_cols=params["tag_cols"],
            tag_size=params["tag_size"],
            tag_spacing=params["tag_spacing"],
            family=params.get("family", "36h11"),
        )

    def __repr__(self):
        return (
            f"AprilGridTarget({self.tag_rows}x{self.tag_cols} tags, "
            f"{self.num_points} corners, size={self.tag_size} m, "
            f"spacing={self.tag_spacing}, family={self.family})"
        )
