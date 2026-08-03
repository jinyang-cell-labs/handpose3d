#!/usr/bin/env python3
"""Egocentric Left/Right hand label tracker (body_cam_teleop).

MediaPipe's handedness classifier assumes a third-person, unmirrored view and
is unreliable on an egocentric body camera. This tracker replaces "trust the
per-frame label" with the constraints that hold for this rig: exactly one
operator, at most one Left and one Right hand, the left hand stays on the
image's left side and the right hand on the right (hands never cross), and
occasionally only one hand is in the scene.

Decision ladder, strongest cue first:

  1. TWO hands visible -> assign labels by wrist x-order. Deterministic; the
     MediaPipe labels are ignored (kept only for comparison/diagnostics).
  2. ONE hand matched to a track with a committed label -> keep the track's
     label. Labels are sticky per track: hands do not teleport between frames,
     so a lone right hand stays "Right" even if it drifts across the midline.
  3. ONE hand on a NEW (or still uncommitted) track -> accumulate weak cues
     over several frames before committing: the side-of-image prior (abstains
     inside a dead zone around the centreline), MediaPipe's own label as a
     score-weighted vote, and an exclusivity vote when the opposite label's
     track was seen recently. Until commit, the current best guess is returned
     and marked provisional.

Detections whose wrists are closer than duplicate_sep_frac of the image width
are treated as one physical hand (MediaPipe sometimes reports the same hand
twice); only the highest-score one is resolved.

Pure Python, no ROS dependencies: importable by both hand_landmarks_node and
mediapipe_detection_debug_node (installed into the same directory), and
testable standalone.
"""
import math
from dataclasses import dataclass

LEFT = "Left"
RIGHT = "Right"

# Cue weights for the lone-new-track case (step 3 above). Per-frame votes are
# clamped to [-1, 1] and folded into an EMA; positive = Right, negative = Left.
SIDE_WEIGHT = 1.0
MEDIAPIPE_WEIGHT = 0.5
EXCLUSIVITY_WEIGHT = 0.75
EVIDENCE_ALPHA = 0.4        # EMA update rate per frame
COMMIT_EVIDENCE = 0.35      # |EMA| required (together with commit_frames)

# Association tie-break: a track unseen for dt is penalized by dt * this many
# image-widths of distance, so a hand moving toward where the OTHER hand
# disappeared cannot steal the stale track (and its label) from its own
# fresh one.
STALE_PENALTY_WIDTHS_PER_SEC = 1.0


@dataclass
class Resolution:
    """Outcome for one detection, returned in input order.

    source: position  - two-hand x-order assignment (deterministic)
            track     - sticky label from an already-committed track
            cues      - fused weak cues (committed=False while accumulating)
            duplicate - same physical hand as a higher-score detection
            mediapipe - raw MediaPipe label passed through (tracker bypassed)
    """
    label: str          # resolved label ("" only for duplicates)
    source: str
    committed: bool
    track_id: int       # -1 when no track backs this resolution
    mp_label: str
    mp_score: float

    @property
    def corrected(self):
        """True when the tracker overrode MediaPipe's label."""
        return bool(self.label) and self.label != self.mp_label


class _Track:
    __slots__ = ("track_id", "wrist", "last_seen", "label", "evidence", "frames")

    def __init__(self, track_id, wrist, now):
        self.track_id = track_id
        self.wrist = wrist
        self.last_seen = now
        self.label = None       # committed label; None while accumulating
        self.evidence = 0.0     # EMA in [-1, 1]: + Right, - Left
        self.frames = 0


class HandLabelTracker:
    def __init__(self, left_is_image_left=True, side_dead_zone_frac=0.10,
                 commit_frames=5, max_gap_sec=0.5, max_jump_frac=0.35,
                 duplicate_sep_frac=0.06):
        # Pixel-side mapping: True = the operator's left hand appears on the
        # image's left (unmirrored egocentric camera). False for mirrored feeds.
        self.left_is_image_left = bool(left_is_image_left)
        # Half-width around the image centreline (fraction of width) where the
        # side-of-image prior abstains instead of voting.
        self.side_dead_zone_frac = float(side_dead_zone_frac)
        # Frames of accumulated evidence before a lone new track's label commits.
        self.commit_frames = int(commit_frames)
        # A track unseen for longer than this is dead (its label is forgotten).
        self.max_gap_sec = float(max_gap_sec)
        # Max wrist move between frames (fraction of width) to associate a
        # detection with an existing track.
        self.max_jump_frac = float(max_jump_frac)
        # Two wrists closer than this (fraction of width) are one physical hand.
        self.duplicate_sep_frac = float(duplicate_sep_frac)
        self._tracks = []
        self._next_id = 0

    def reset(self):
        self._tracks = []

    # ------------------------------------------------------------------ update
    def update(self, detections, image_width, now):
        """Resolve labels for one frame.

        detections: sequence of (wrist_x_px, wrist_y_px, mp_label, mp_score).
        now: monotonic seconds. Returns one Resolution per detection, in order.
        """
        w = float(max(image_width, 1))
        self._tracks = [
            t for t in self._tracks if now - t.last_seen <= self.max_gap_sec]

        dets = [(float(x), float(y), str(lbl), float(sc))
                for x, y, lbl, sc in detections]
        results = [None] * len(dets)

        active = self._drop_duplicates(dets, results, w)
        # More than two survivors cannot all be real under the one-operator
        # constraint: resolve the two highest-score ones, pass the rest through.
        if len(active) > 2:
            by_score = sorted(active, key=lambda i: -dets[i][3])
            for i in by_score[2:]:
                results[i] = self._passthrough(dets[i])
            active = sorted(by_score[:2])

        match = self._associate(active, dets, w, now)
        if len(active) == 2:
            self._resolve_two(active, dets, results, match, now)
        elif len(active) == 1:
            self._resolve_one(active[0], dets, results, match, w, now)

        for i, r in enumerate(results):
            if r is None:  # safety net; should not happen
                results[i] = self._passthrough(dets[i])
        return results

    # ---------------------------------------------------------------- internal
    @staticmethod
    def _passthrough(det):
        return Resolution(label=det[2], source="mediapipe", committed=False,
                          track_id=-1, mp_label=det[2], mp_score=det[3])

    def _drop_duplicates(self, dets, results, w):
        if len(dets) < 2:
            return list(range(len(dets)))
        min_sep = self.duplicate_sep_frac * w
        kept = []
        for i in sorted(range(len(dets)), key=lambda i: -dets[i][3]):
            near = any(
                math.hypot(dets[i][0] - dets[j][0], dets[i][1] - dets[j][1])
                < min_sep for j in kept)
            if near:
                results[i] = Resolution(
                    label="", source="duplicate", committed=False,
                    track_id=-1, mp_label=dets[i][2], mp_score=dets[i][3])
            else:
                kept.append(i)
        return sorted(kept)

    def _associate(self, active, dets, w, now):
        """Greedy nearest wrist-to-track matching, gated by max_jump_frac.

        Matching order uses distance plus a staleness penalty: a track updated
        last frame outranks one coasting unseen, even when the unseen one's
        last position happens to be closer.
        """
        max_jump = self.max_jump_frac * w
        pairs = []
        for i in active:
            for t in self._tracks:
                d = math.hypot(dets[i][0] - t.wrist[0], dets[i][1] - t.wrist[1])
                if d <= max_jump:
                    stale = (now - t.last_seen) * STALE_PENALTY_WIDTHS_PER_SEC * w
                    pairs.append((d + stale, i, t))
        pairs.sort(key=lambda p: p[0])
        match, used = {}, set()
        for _, i, t in pairs:
            if i in match or t.track_id in used:
                continue
            match[i] = t
            used.add(t.track_id)
        return match

    def _new_track(self, det, now):
        t = _Track(self._next_id, (det[0], det[1]), now)
        self._next_id += 1
        self._tracks.append(t)
        return t

    @staticmethod
    def _touch(track, det, now):
        track.wrist = (det[0], det[1])
        track.last_seen = now

    def _resolve_two(self, active, dets, results, match, now):
        """Both hands visible: x-order decides, MediaPipe's labels do not."""
        by_x = sorted(active, key=lambda i: dets[i][0])
        labels = (LEFT, RIGHT) if self.left_is_image_left else (RIGHT, LEFT)
        for i, label in zip(by_x, labels):
            track = match.get(i)
            if track is None:
                track = self._new_track(dets[i], now)
            self._touch(track, dets[i], now)
            track.label = label
            track.evidence = 1.0 if label == RIGHT else -1.0
            track.frames = max(track.frames, self.commit_frames)
            results[i] = Resolution(
                label=label, source="position", committed=True,
                track_id=track.track_id, mp_label=dets[i][2],
                mp_score=dets[i][3])

    def _resolve_one(self, i, dets, results, match, w, now):
        x, _, mp_label, mp_score = dets[i]
        track = match.get(i)
        if track is not None and track.label is not None:
            # Sticky label: the track already knows which hand it is.
            self._touch(track, dets[i], now)
            results[i] = Resolution(
                label=track.label, source="track", committed=True,
                track_id=track.track_id, mp_label=mp_label, mp_score=mp_score)
            return

        if track is None:
            track = self._new_track(dets[i], now)
        self._touch(track, dets[i], now)

        vote = 0.0
        x_frac = x / w
        if abs(x_frac - 0.5) > self.side_dead_zone_frac:
            on_right_half = x_frac > 0.5
            is_right = (on_right_half if self.left_is_image_left
                        else not on_right_half)
            vote += SIDE_WEIGHT * (1.0 if is_right else -1.0)
        vote += MEDIAPIPE_WEIGHT * (mp_score if mp_label == RIGHT else -mp_score)
        other = next(
            (t for t in self._tracks
             if t.track_id != track.track_id and t.label is not None), None)
        if other is not None:
            vote += EXCLUSIVITY_WEIGHT * (1.0 if other.label == LEFT else -1.0)

        vote = max(-1.0, min(1.0, vote))
        track.evidence += EVIDENCE_ALPHA * (vote - track.evidence)
        track.frames += 1

        if track.evidence > 0.0:
            guess = RIGHT
        elif track.evidence < 0.0:
            guess = LEFT
        else:
            guess = mp_label
        committed = (track.frames >= self.commit_frames
                     and abs(track.evidence) >= COMMIT_EVIDENCE)
        if committed:
            track.label = guess
        results[i] = Resolution(
            label=guess, source="cues", committed=committed,
            track_id=track.track_id, mp_label=mp_label, mp_score=mp_score)
