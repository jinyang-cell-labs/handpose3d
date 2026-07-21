#!/usr/bin/env python3
"""Synchronized, interpolated phone pose per camera tick.

For each tick in a recorded session's ``timestamps.csv`` this:

1. averages the per-camera ``*_capture_time`` values (of the cameras that
   captured a *fresh* frame that tick — duplicates from a slow camera are
   skipped via the ``*_seq`` columns), giving the tick's mean capture instant
   on the camera clock (``time.monotonic``);
2. maps that instant onto the laptop **wall clock** using ``meta.yaml``'s
   ``clock_anchor`` (``wall = monotonic + (anchor.wall_ns - anchor.monotonic_ns)``);
3. interpolates the phone's ARCore pose stream (``phone_pose.jsonl``) to that
   wall time.

Reference frame — auto-detected:

* If the stream carries the ARCore **calibration anchor** (``apx..aqw``, present
  once Calibrate has been pressed), each phone packet's pose is expressed in the
  anchor frame: ``anchor_T_phone = inv(world_T_anchor) @ world_T_phone`` (both
  come from the same packet/instant, so this cancels ARCore's world-frame drift
  and loop-closure jumps). Output -> ``anchor_T_phone_sync.jsonl``.
* If no anchor is present, the anchor is treated as the identity/origin (the
  ``0 0 0 0 0 0`` placeholder) and the pose is left in the ARCore world frame:
  ``world_T_phone``. Output -> ``world_T_phone_sync.jsonl``.

(``a_T_b`` = pose of frame ``b`` expressed in frame ``a``.)

Interpolation: the reference time axis is the laptop wall clock. Phone samples
are indexed by ``t_laptop`` (their frame timestamp mapped onto the laptop wall
clock) and interpolated against those real timestamps — linear on translation,
SLERP on rotation — so ARCore's jittery 20-40 Hz spacing is handled correctly.
Bracketing samples farther apart than ``max_gap`` (dropped UDP packets) are
flagged; ticks outside the phone data, or (anchor mode) straddling a
re-calibration onto a different anchor, are marked invalid. Every ``tick_idx``
gets exactly one output line, so the file stays index-aligned with the frames
(``p=[x,y,z]`` m, ``q=[x,y,z,w]``).
"""
import csv
import json
import os

import numpy as np
import yaml
from PyQt5.QtCore import QThread, pyqtSignal
from scipy.spatial.transform import Rotation, Slerp

PHONE_LOG = "phone_pose.jsonl"
ANCHOR_OUTPUT = "anchor_T_phone_sync.jsonl"
WORLD_OUTPUT = "world_T_phone_sync.jsonl"
OUTPUT_NAMES = (ANCHOR_OUTPUT, WORLD_OUTPUT)
DEFAULT_MAX_GAP_MS = 100.0


def load_clock_anchor(session_dir):
    """-> (monotonic_ns, wall_ns) from meta.yaml; raises if absent."""
    with open(os.path.join(session_dir, "meta.yaml")) as f:
        meta = yaml.safe_load(f) or {}
    ca = meta.get("clock_anchor")
    if not ca or "monotonic_ns" not in ca or "wall_ns" not in ca:
        raise ValueError(
            "meta.yaml has no clock_anchor — this session was recorded before "
            "the clock-anchor feature; camera time cannot be mapped to wall time")
    return int(ca["monotonic_ns"]), int(ca["wall_ns"])


def load_tick_times(session_dir):
    """Mean capture instant per tick, averaging only freshly-captured frames.

    -> (tick_idx[int64], tick_mono_ns[int64]). A camera counts as fresh on a
    tick when its ``*_seq`` advanced since the previous tick; if none advanced
    (all duplicates) we fall back to averaging all cameras, then to tick_time.
    """
    with open(os.path.join(session_dir, "timestamps.csv"), newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        cap_cols = [c for c in fields if c.endswith("_capture_time")]
        if not cap_cols:
            raise ValueError("timestamps.csv has no *_capture_time columns")
        labels = [c[: -len("_capture_time")] for c in cap_cols]

        idxs, monos = [], []
        prev_seq = {}
        for row in reader:
            fresh, allv = [], []
            for lab in labels:
                try:
                    cap = float(row[lab + "_capture_time"])
                    seq = int(row[lab + "_seq"])
                except (TypeError, ValueError, KeyError):
                    continue
                if cap > 0:
                    allv.append(cap)
                    if prev_seq.get(lab) != seq:
                        fresh.append(cap)
                prev_seq[lab] = seq
            pool = fresh or allv
            mean_s = (sum(pool) / len(pool)) if pool else float(row.get("tick_time") or 0)
            idxs.append(int(row["tick_idx"]))
            monos.append(int(round(mean_s * 1e9)))
    return np.array(idxs, dtype=np.int64), np.array(monos, dtype=np.int64)


def load_pose_samples(session_dir):
    """Parse phone_pose.jsonl into per-packet interpolation samples.

    Auto-detects the reference frame: 'anchor' if any TRACKING packet carries a
    calibration anchor (apx..aqw), else 'world'. Returns (mode, samples) where
    samples is a dict of arrays sorted by strictly-increasing t_laptop, or None
    if there are no usable (TRACKING) poses:
        t     int64 wall ns (t_laptop)
        p     (N,3) translation  (anchor_T_phone, or world_T_phone in world mode)
        q     (N,4) quaternion [x,y,z,w]
        calib (N,)  phone calibration counter at the sample
    """
    raw = []
    with open(os.path.join(session_dir, PHONE_LOG)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") == "pose" and r.get("state") == "TRACKING":
                raw.append(r)
    if not raw:
        return "world", None

    has_anchor = any(r.get("apx") is not None and r.get("aqw") is not None
                     for r in raw)
    mode = "anchor" if has_anchor else "world"

    ts, ps, qs, calibs = [], [], [], []
    for r in raw:
        q_phone = np.array([r["qx"], r["qy"], r["qz"], r["qw"]], float)
        p_phone = np.array([r["px"], r["py"], r["pz"]], float)
        if mode == "anchor":
            if r.get("apx") is None or r.get("aqw") is None:
                continue  # pre-Calibrate packet without an anchor yet
            q_anch = np.array([r["aqx"], r["aqy"], r["aqz"], r["aqw"]], float)
            p_anch = np.array([r["apx"], r["apy"], r["apz"]], float)
            R_anch = Rotation.from_quat(q_anch)
            p = R_anch.inv().apply(p_phone - p_anch)
            q = (R_anch.inv() * Rotation.from_quat(q_phone)).as_quat()
        else:  # anchor treated as identity -> pose stays in the ARCore world frame
            p, q = p_phone, q_phone
        ts.append(int(r["t_laptop"]))
        ps.append(p)
        qs.append(q)
        calibs.append(int(r.get("calib", 0)))
    if not ts:
        return mode, None

    t = np.array(ts, dtype=np.int64)
    order = np.argsort(t, kind="stable")
    t, p, q, calib = t[order], np.array(ps)[order], np.array(qs)[order], \
        np.array(calibs, dtype=np.int64)[order]
    keep = np.concatenate(([True], np.diff(t) > 0))  # strictly increasing for Slerp
    return mode, {"t": t[keep], "p": p[keep], "q": q[keep], "calib": calib[keep]}


def generate_sync(session_dir, max_gap_ns=DEFAULT_MAX_GAP_MS * 1e6,
                  progress_cb=None, log_cb=None):
    """Write the per-tick sync file for one session; returns a summary dict."""
    def log(m):
        if log_cb:
            log_cb(m)

    def prog(d, t):
        if progress_cb:
            progress_cb(int(d), int(t))

    mono0, wall0 = load_clock_anchor(session_dir)
    offset = wall0 - mono0
    tick_idx, tick_mono = load_tick_times(session_dir)
    tick_wall = tick_mono + offset
    n = len(tick_idx)
    log(f"{n} camera ticks; clock-anchor offset {offset / 1e9:+.6f} s")
    prog(0, n)

    mode, samples = load_pose_samples(session_dir)
    out_path = os.path.join(session_dir, ANCHOR_OUTPUT if mode == "anchor"
                            else WORLD_OUTPUT)
    frame = "anchor_T_phone" if mode == "anchor" else "world_T_phone"

    valid = np.zeros(n, bool)
    reason = ["no_pose_data"] * n
    p_out = np.full((n, 3), np.nan)
    q_out = np.full((n, 4), np.nan)
    calib_out = np.full(n, -1, dtype=np.int64)
    gap_ms = np.full(n, np.nan)

    if samples is None:
        log("WARNING: no TRACKING poses found in phone_pose.jsonl")
    else:
        t, P, Q, C = samples["t"], samples["p"], samples["q"], samples["calib"]
        if mode == "anchor":
            log(f"calibration anchor present -> {frame} (anchor frame); "
                f"{len(t)} samples, calib ids {sorted(set(C.tolist()))}")
        else:
            log(f"no calibration anchor in stream -> {frame} (ARCore world "
                f"frame, anchor = identity placeholder); {len(t)} samples")
        log(f"phone span {(t[-1] - t[0]) / 1e9:.1f} s")
        base = int(t[0])
        t_s = (t - base) / 1e9  # rebased seconds: keeps float64 precision on ns

        # In anchor mode, split into contiguous constant-calib runs so we never
        # interpolate across a re-calibration onto a different anchor. In world
        # mode the calib counter is irrelevant, so it is one continuous stream.
        if mode == "anchor":
            bounds = np.concatenate(([0], np.where(np.diff(C) != 0)[0] + 1, [len(t)]))
        else:
            bounds = np.array([0, len(t)])

        covered = np.zeros(n, bool)
        for si in range(len(bounds) - 1):
            a, b = int(bounds[si]), int(bounds[si + 1])
            if b - a < 2:
                continue  # need two samples to interpolate
            seg_t = t_s[a:b]
            slerp = Slerp(seg_t, Rotation.from_quat(Q[a:b]))
            lo, hi = int(t[a]), int(t[b - 1])
            sel = np.where((tick_wall >= lo) & (tick_wall <= hi))[0]
            if len(sel) == 0:
                continue
            query = np.clip((tick_wall[sel] - base) / 1e9, seg_t[0], seg_t[-1])
            ui = np.clip(np.searchsorted(seg_t, query, side="right"), 1, len(seg_t) - 1)
            li = ui - 1
            dt = seg_t[ui] - seg_t[li]
            alpha = np.where(dt > 0, (query - seg_t[li]) / dt, 0.0)
            seg_P = P[a:b]
            p_interp = seg_P[li] + alpha[:, None] * (seg_P[ui] - seg_P[li])
            q_interp = slerp(query).as_quat()
            g = dt * 1e3  # ms
            for k, ti in enumerate(sel):
                covered[ti] = True
                p_out[ti], q_out[ti] = p_interp[k], q_interp[k]
                calib_out[ti], gap_ms[ti] = int(C[a]), g[k]
                if g[k] <= max_gap_ns / 1e6:
                    valid[ti], reason[ti] = True, ""
                else:
                    reason[ti] = "large_gap"
            prog(int(covered.sum()), n)

        first_t, last_t = int(t[0]), int(t[-1])
        for ti in range(n):
            if covered[ti]:
                continue
            reason[ti] = ("before_data" if tick_wall[ti] < first_t
                          else "after_data" if tick_wall[ti] > last_t
                          else "calib_gap")

    n_valid = _write(out_path, session_dir, mode, frame, tick_idx, tick_wall,
                     tick_mono, valid, reason, p_out, q_out, calib_out, gap_ms,
                     0 if samples is None else len(samples["t"]), max_gap_ns)
    prog(n, n)
    log(f"wrote {n_valid}/{n} valid ticks -> {os.path.basename(out_path)}")
    return {"mode": mode, "n_ticks": n, "n_valid": n_valid, "out_path": out_path,
            "n_samples": 0 if samples is None else len(samples["t"]),
            "coverage": (n_valid / n) if n else 0.0}


def _write(out_path, session_dir, mode, frame, tick_idx, tick_wall, tick_mono,
           valid, reason, p_out, q_out, calib_out, gap_ms, n_samples, max_gap_ns):
    n = len(tick_idx)
    n_valid = int(np.count_nonzero(valid))
    calib_ids = sorted({int(c) for c in calib_out if c >= 0})
    frame_note = (
        "anchor_T_phone = inv(world_T_anchor) @ world_T_phone"
        if mode == "anchor" else
        "world_T_phone in the ARCore world frame (no calibration anchor in the "
        "stream; anchor treated as identity/origin)")
    with open(out_path, "w") as f:
        f.write(json.dumps({
            "type": "meta",
            "source": os.path.splitext(os.path.basename(out_path))[0],
            "mode": mode,
            "frame": frame,
            "session": os.path.basename(os.path.normpath(session_dir)),
            "n_ticks": n,
            "n_valid": n_valid,
            "n_pose_samples": n_samples,
            "max_gap_ms": round(max_gap_ns / 1e6, 3),
            "calib_ids": calib_ids,
            "reference_clock": ("laptop wall clock (unix ns); tick time = mean "
                                "fresh-camera capture_time (monotonic) mapped via "
                                "meta.yaml clock_anchor"),
            "frame_note": f"{frame_note}; p = translation [m], q = quaternion [x,y,z,w]",
        }) + "\n")
        for i in range(n):
            rec = {"type": "pose", "tick_idx": int(tick_idx[i]),
                   "t_wall_ns": int(tick_wall[i]), "t_mono_ns": int(tick_mono[i]),
                   "valid": bool(valid[i])}
            if not np.isnan(gap_ms[i]):
                rec["gap_ms"] = round(float(gap_ms[i]), 3)
            if not np.isnan(p_out[i][0]):  # interpolated (valid or large_gap)
                rec["calib"] = int(calib_out[i])
                rec["p"] = [round(float(x), 6) for x in p_out[i]]
                rec["q"] = [round(float(x), 8) for x in q_out[i]]
            if not valid[i]:
                rec["reason"] = reason[i] or "large_gap"
            f.write(json.dumps(rec) + "\n")
    return n_valid


class SyncWorker(QThread):
    """Runs generate_sync off the GUI thread, emitting progress/log/done."""

    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def __init__(self, session_dir, max_gap_ms, parent=None):
        super().__init__(parent)
        self.session_dir = session_dir
        self.max_gap_ms = max_gap_ms

    def run(self):
        try:
            s = generate_sync(
                self.session_dir, self.max_gap_ms * 1e6,
                progress_cb=lambda d, t: self.progress.emit(d, t),
                log_cb=self.log.emit)
            self.done.emit(True, (
                f"done ✓  {s['mode']} frame — {s['n_valid']}/{s['n_ticks']} ticks "
                f"valid ({100 * s['coverage']:.0f}% coverage), "
                f"{s['n_samples']} samples\n{s['out_path']}"))
        except Exception as e:  # surface the failure to the GUI
            import traceback
            traceback.print_exc()
            self.done.emit(False, f"{type(e).__name__}: {e}")
