#!/usr/bin/env python3
"""
Analyse a deviation log written by replay_logs.py (ENABLE_DEVIATION_LOG) and show
a one-window dashboard of how good — small and consistent — the triangulation
-vs-estimated-pose deviation is.

The stored metric per frame is the sum over the 21 joints of ||tri - pose||^2
(m^2). For readability everything here is shown as the **per-joint RMS in mm**,
``rms_mm = sqrt(SSD / 21) * 1000`` (the typical distance a joint is off by); the
raw m^2 stats are still printed in the summary.

Dashboard (per hand, overlaid where it makes sense):
  - config + summary panel  (what produced the data + accuracy/consistency stats)
  - histogram               (distribution; mean / median / p95 marked)
  - ECDF                    ("X% of frames are below Y mm"; p50/p90/p95 marked)
  - full-timeline series    (trend over the whole take; outlier line + gaps)
  - per-joint mean bar       (which joints drive the error; needs the per-joint
                              field, recorded by the updated logger)

Run from the evaluation venv (see start_evaluation.bash):
    python deviation_estimation.py [path/to/deviation_log.json]
"""
import glob
import importlib.util
import json
import os
import sys

import numpy as np

# ============================ USER CONFIG ==================================
# Path to the deviation log JSON. "" -> the newest file in ./deviation_logs.
# A command-line argument, if given, overrides this.
DEVIATION_LOG_PATH = ""
# A frame counts as an outlier (likely a single-view flip) when its per-joint
# RMS exceeds OUTLIER_FACTOR x the median.
OUTLIER_FACTOR = 3.0
# Histogram: clip the x-axis at this percentile so the heavy tail doesn't crush
# the informative bulk (clipped frames are still counted, just off-axis).
HIST_CLIP_PCTL = 98
HIST_BINS = 40
# ===========================================================================

N_JOINTS = 21
JOINT_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]
# Per-finger colour for the per-joint bar chart (wrist + 5 fingers).
FINGER_OF = ([0] + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4)
FINGER_COLORS = ["#555555", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
HAND_COLORS = {"Left": "#3399ff", "Right": "#ff8033"}

# Interactive backend (same rationale as replay_logs.py).
import matplotlib  # noqa: E402
for _mod, _be in (("PyQt5", "QtAgg"), ("PySide6", "QtAgg"), ("tkinter", "TkAgg")):
    if importlib.util.find_spec(_mod) is not None:
        matplotlib.use(_be)
        break
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402


def ssd_to_rms_mm(ssd_m2):
    """SSD (m^2 over 21 joints) -> per-joint RMS distance in mm."""
    return np.sqrt(np.asarray(ssd_m2, float) / N_JOINTS) * 1000.0


def resolve_path():
    if len(sys.argv) > 1:
        return sys.argv[1]
    if DEVIATION_LOG_PATH:
        return DEVIATION_LOG_PATH
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deviation_logs")
    files = sorted(glob.glob(os.path.join(here, "deviation_*.json")))
    if not files:
        sys.exit(f"no deviation log given and none found in {here}")
    return files[-1]


def parse(payload):
    """Extract per-hand arrays from the log payload.

    Returns dict {label: {t, rms_mm (present frames), t_all, rms_all (NaN gaps),
    per_joint_mm (F,21)}} plus the global n_frames.
    """
    frames = payload["frames"]
    n = len(frames)
    labels = sorted({lbl for f in frames for lbl in f.get("deviation", {})})
    out = {}
    for label in labels:
        t_all = np.array([f["t_sec"] for f in frames], float)
        rms_all = np.full(n, np.nan)
        per_joint = []
        for i, f in enumerate(frames):
            ssd = f.get("deviation", {}).get(label)
            if ssd is not None:
                rms_all[i] = ssd_to_rms_mm(ssd)
            pj = f.get("per_joint_dist_m", {}).get(label)
            if pj is not None:
                per_joint.append([np.nan if v is None else v for v in pj])
        present = np.isfinite(rms_all)
        out[label] = {
            "t_all": t_all,
            "rms_all": rms_all,
            "t": t_all[present],
            "rms": rms_all[present],
            "per_joint_mm": (np.array(per_joint, float) * 1000.0
                             if per_joint else None),
        }
    return out, n


def stats(rms, n_frames):
    """Accuracy + consistency stats on the per-joint RMS (mm) of a hand."""
    rms = np.asarray(rms, float)
    med = float(np.median(rms))
    q1, q3 = np.percentile(rms, [25, 75])
    thr = OUTLIER_FACTOR * med
    return {
        "count": int(rms.size),
        "coverage": rms.size / n_frames if n_frames else float("nan"),
        "mean": float(np.mean(rms)),
        "median": med,
        "std": float(np.std(rms)),
        "var": float(np.var(rms)),
        "cv": float(np.std(rms) / np.mean(rms)) if np.mean(rms) else float("nan"),
        "iqr": float(q3 - q1),
        "mad": float(np.median(np.abs(rms - med))),
        "p90": float(np.percentile(rms, 90)),
        "p95": float(np.percentile(rms, 95)),
        "p99": float(np.percentile(rms, 99)),
        "min": float(rms.min()),
        "max": float(rms.max()),
        "outlier_thr": float(thr),
        "outlier_rate": float(np.mean(rms > thr)),
    }


# -------------------------------------------------------------------- panels
def panel_text(ax, payload, per_hand_stats, path):
    ax.axis("off")
    cfg = payload.get("config", {})
    lines = [f"file: {os.path.basename(path)}",
             f"created: {payload.get('created', '?')}",
             f"metric: {payload.get('deviation_metric', '?')}",
             f"triangulate: {payload.get('triangulate_cameras')}   "
             f"pose<-{payload.get('pose_source')}",
             f"timeline frames: {payload.get('n_timeline_frames')}",
             "", "config:"]
    for k in ("LOG_PATH", "HANDEDNESS", "SYNC_TOLERANCE_S",
              "ENABLE_POSE_ESTIMATION_REPROJECT", "DEVIATION_LENGTH"):
        if k in cfg:
            v = cfg[k]
            if k == "LOG_PATH":
                v = os.path.basename(str(v))
            lines.append(f"  {k} = {v}")
    lines.append("")
    lines.append("per-joint RMS deviation (mm) — smaller & tighter = better:")
    for label, s in per_hand_stats.items():
        lines.append(f"  [{label}]  n={s['count']} ({s['coverage']*100:.0f}% cov)")
        lines.append(f"     median={s['median']:.1f}  mean={s['mean']:.1f}  "
                     f"min={s['min']:.1f}  max={s['max']:.1f}")
        lines.append(f"     std={s['std']:.1f}  var={s['var']:.1f}  "
                     f"IQR={s['iqr']:.1f}  MAD={s['mad']:.1f}  CV={s['cv']:.2f}")
        lines.append(f"     p90={s['p90']:.1f}  p95={s['p95']:.1f}  "
                     f"p99={s['p99']:.1f}")
        lines.append(f"     outliers >{s['outlier_thr']:.0f}mm: "
                     f"{s['outlier_rate']*100:.1f}%")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8,
            family="monospace", transform=ax.transAxes)
    ax.set_title("configuration & summary", fontsize=9)


def panel_hist(ax, data, per_hand_stats):
    hi = max((np.percentile(d["rms"], HIST_CLIP_PCTL) for d in data.values()),
             default=1.0)
    hi = max(hi, 1.0)
    for label, d in data.items():
        c = HAND_COLORS.get(label, "#888888")
        ax.hist(d["rms"], bins=HIST_BINS, range=(0, hi), alpha=0.55,
                color=c, label=label)
        s = per_hand_stats[label]
        ax.axvline(s["median"], color=c, ls="-", lw=1.5)
        ax.axvline(s["mean"], color=c, ls="--", lw=1.2)
        ax.axvline(s["p95"], color=c, ls=":", lw=1.2)
    ax.set_xlim(0, hi)
    ax.set_xlabel("per-joint RMS deviation (mm)")
    ax.set_ylabel("frame count")
    ax.set_title(f"distribution (x clipped at p{HIST_CLIP_PCTL}; "
                 "— median, -- mean, ·· p95)", fontsize=9)
    ax.legend(fontsize=8)


def panel_ecdf(ax, data):
    for label, d in data.items():
        x = np.sort(d["rms"])
        y = np.arange(1, x.size + 1) / x.size
        c = HAND_COLORS.get(label, "#888888")
        ax.plot(x, y, drawstyle="steps-post", color=c, label=label)
        for p, ls in ((0.5, "-"), (0.9, "--"), (0.95, ":")):
            xv = np.percentile(d["rms"], p * 100)
            ax.plot([xv, xv], [0, p], color=c, ls=ls, lw=1.0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("per-joint RMS deviation (mm)")
    ax.set_ylabel("fraction of frames ≤ x")
    ax.set_title("ECDF (— p50, -- p90, ·· p95)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def panel_timeline(ax, data, per_hand_stats):
    for label, d in data.items():
        c = HAND_COLORS.get(label, "#888888")
        ax.plot(d["t_all"], d["rms_all"], "-o", ms=2, lw=0.8, color=c,
                label=label)
        ax.axhline(per_hand_stats[label]["outlier_thr"], color=c, ls=":", lw=1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("per-joint RMS (mm)")
    ax.set_title("deviation over the whole take (gaps = no value; "
                 "·· outlier threshold)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def panel_per_joint(ax, data):
    have = {l: d for l, d in data.items() if d["per_joint_mm"] is not None}
    if not have:
        ax.axis("off")
        ax.text(0.5, 0.5, "no per-joint data in this log\n"
                "(re-record with the updated logger)", ha="center", va="center",
                fontsize=9, transform=ax.transAxes)
        ax.set_title("per-joint mean deviation", fontsize=9)
        return
    labels = list(have)
    width = 0.8 / len(labels)
    x = np.arange(N_JOINTS)
    for li, label in enumerate(labels):
        pj = have[label]["per_joint_mm"]
        # Median + IQR: robust to the flip frames that otherwise inflate every
        # joint's mean equally and hide the real per-joint pattern.
        med = np.nanmedian(pj, axis=0)
        q1 = np.nanpercentile(pj, 25, axis=0)
        q3 = np.nanpercentile(pj, 75, axis=0)
        yerr = np.vstack([np.clip(med - q1, 0, None), np.clip(q3 - med, 0, None)])
        colors = [FINGER_COLORS[FINGER_OF[j]] for j in range(N_JOINTS)]
        ax.bar(x + li * width, med, width, yerr=yerr, capsize=2,
               color=colors, edgecolor="black" if len(labels) > 1 else "none",
               linewidth=0.4, error_kw={"elinewidth": 0.6, "alpha": 0.5})
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=6)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("median deviation (mm)")
    ax.set_title("per-joint median deviation (coloured by finger; bars = IQR)"
                 + ("  [solid=Left]" if len(labels) > 1 else ""), fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)


def main():
    path = resolve_path()
    with open(path, "r") as fh:
        payload = json.load(fh)
    data, n_frames = parse(payload)
    if not data:
        sys.exit(f"{path}: no deviation values found")
    per_hand_stats = {l: stats(d["rms"], n_frames) for l, d in data.items()}

    # Console summary.
    print(f"deviation analysis: {path}")
    for label, s in per_hand_stats.items():
        print(f"  [{label}] n={s['count']} cov={s['coverage']*100:.0f}%  "
              f"median={s['median']:.1f}mm mean={s['mean']:.1f}mm "
              f"std={s['std']:.1f} CV={s['cv']:.2f} p95={s['p95']:.1f} "
              f"outliers={s['outlier_rate']*100:.1f}%")

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(f"Deviation analysis — {os.path.basename(path)}", fontsize=11)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1])
    panel_text(fig.add_subplot(gs[0, 0]), payload, per_hand_stats, path)
    panel_hist(fig.add_subplot(gs[0, 1]), data, per_hand_stats)
    panel_ecdf(fig.add_subplot(gs[0, 2]), data)
    panel_timeline(fig.add_subplot(gs[1, 0:2]), data, per_hand_stats)
    panel_per_joint(fig.add_subplot(gs[1, 2]), data)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


if __name__ == "__main__":
    main()
