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

# Maps an aruco corner slot (0..3, as returned by detectMarkers) to an index
# into a tag's kalibr corner tuple [BL, BR, TR, TL] (see tag_point_indices).
# OpenCV's apriltag dictionary fixes each tag's canonical orientation
# differently from kalibr's MIT AprilTag detector, so the naive [TL,TR,BR,BL]
# -> [BL,BR,TR,TL] guess is wrong (gives ~76px reprojection RMS). This ordering
# was determined empirically with scripts/corr_debug.py, which brute-forces all
# 8 dihedral orderings against a near-frontal board: (1,0,3,2) wins at ~1.1px
# homography residual, every other ordering is >40px.
# Re-validate with corr_debug.py if a calibration shows large/structured RMS.
_ARUCO_TO_KALIBR = (1, 0, 3, 2)


class AprilGridTarget:
    """A kalibr-compatible AprilGrid: 3D object points + a corner detector."""

    def __init__(self, tag_rows, tag_cols, tag_size, tag_spacing, family="36h11",
                 border_bits=2, do_subpix=True, subpix_window=2,
                 max_subpix_displacement=1.5, min_border_distance=4.0):
        if int(tag_rows) < 1 or int(tag_cols) < 1:
            raise ValueError("tag_rows and tag_cols must be >= 1")
        if float(tag_size) <= 0.0 or float(tag_spacing) <= 0.0:
            raise ValueError("tag_size and tag_spacing must be positive")
        if int(border_bits) < 1:
            raise ValueError("border_bits must be >= 1")

        self.tag_rows = int(tag_rows)
        self.tag_cols = int(tag_cols)
        self.tag_size = float(tag_size)
        self.tag_spacing = float(tag_spacing)
        self.family = str(family)
        # Width of the tag's black border in bits. Boards from
        # kalibr_create_target_pdf use a 2-bit border (kalibr's default
        # blackTagBorder=2); OpenCV's aruco detector assumes 1 unless told
        # otherwise, and silently decodes nothing on a 2-bit board.
        self.border_bits = int(border_bits)

        # Corner-extraction quality filters, mirroring kalibr's
        # AprilgridOptions (GridCalibrationTargetAprilgrid). These matter for
        # reprojection RMS: the small subpix window avoids being pulled by
        # neighbouring tags in the dense grid, and the displacement/border
        # rejections drop unreliable corners.
        self.do_subpix = bool(do_subpix)
        self.subpix_window = int(subpix_window)           # kalibr cv::Size(2,2)
        self.max_subpix_displacement = float(max_subpix_displacement)  # px^2
        self.min_border_distance = float(min_border_distance)          # px

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
        # Leave refinement to our own cv::cornerSubPix in detect(), matching
        # kalibr exactly (small window + displacement rejection). aruco's
        # CORNER_REFINE_SUBPIX uses a large default window that gets pulled by
        # neighbouring tags in the dense grid; CORNER_REFINE_APRILTAG drops many
        # valid detections (measured 35->18 tags on a full-board frame).
        refine = getattr(cv2.aruco, "CORNER_REFINE_NONE", 0)

        def _configure(params):
            params.cornerRefinementMethod = refine
            # Match the printed board's black-border width (kalibr default 2).
            params.markerBorderBits = self.border_bits

        # OpenCV >= 4.7 exposes the ArucoDetector class; older builds use the
        # functional API. Support both.
        if hasattr(cv2.aruco, "ArucoDetector"):
            dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            _configure(params)
            self._detector = ("new", cv2.aruco.ArucoDetector(dictionary, params))
        else:  # pragma: no cover - legacy OpenCV path
            dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters_create()
            _configure(params)
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

        empty = (np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.float32))
        if ids is None or len(ids) == 0:
            return empty

        h, w = gray.shape[:2]
        d = self.min_border_distance
        point_ids = []
        image_points = []
        for quad, tag_id in zip(corners, ids.flatten()):
            tag_id = int(tag_id)
            if tag_id < 0 or tag_id >= self.num_tags:
                continue  # a tag from a different board / spurious id
            q = np.asarray(quad, dtype=np.float32).reshape(4, 2)  # aruco [TL,TR,BR,BL]
            # Drop the whole tag if any corner is too close to the image border
            # (kalibr minBorderDistance): such corners are extrapolated/unstable.
            if d > 0.0 and (
                np.any(q[:, 0] < d) or np.any(q[:, 0] > w - d)
                or np.any(q[:, 1] < d) or np.any(q[:, 1] > h - d)
            ):
                continue
            pidx = self.tag_point_indices(tag_id)  # [BL, BR, TR, TL]
            for slot in range(4):
                point_ids.append(pidx[_ARUCO_TO_KALIBR[slot]])
                image_points.append(q[slot])

        if not point_ids:
            return empty

        point_ids = np.asarray(point_ids, dtype=np.int64)
        image_points = np.asarray(image_points, dtype=np.float32)

        # Subpixel refinement + displacement rejection, matching kalibr's
        # cv::cornerSubPix(win=2x2) followed by the maxSubpixDisplacement2 gate.
        if self.do_subpix and cv2 is not None:
            raw = image_points.copy()
            refined = image_points.reshape(-1, 1, 2).copy()
            win = (self.subpix_window, self.subpix_window)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            cv2.cornerSubPix(gray, refined, win, (-1, -1), criteria)
            refined = refined.reshape(-1, 2)
            disp2 = np.sum((refined - raw) ** 2, axis=1)
            keep = disp2 <= self.max_subpix_displacement
            point_ids = point_ids[keep]
            image_points = refined[keep]

        return (point_ids, image_points.astype(np.float32))

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
            border_bits=params.get("border_bits", 2),
            do_subpix=params.get("do_subpix", True),
            subpix_window=params.get("subpix_window", 2),
            max_subpix_displacement=params.get("max_subpix_displacement", 1.5),
            min_border_distance=params.get("min_border_distance", 4.0),
        )

    def __repr__(self):
        return (
            f"AprilGridTarget({self.tag_rows}x{self.tag_cols} tags, "
            f"{self.num_points} corners, size={self.tag_size} m, "
            f"spacing={self.tag_spacing}, family={self.family})"
        )
