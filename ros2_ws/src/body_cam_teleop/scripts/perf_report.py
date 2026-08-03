#!/usr/bin/env python3
"""Offline analysis of perf_monitor_node CSVs — where does the time go?

Usage:
  python3 perf_report.py                       # newest run in the default dir
  python3 perf_report.py /path/to/logs/perf    # newest run in that dir
  python3 perf_report.py /path/.../20260707_153000   # specific run prefix

No ROS required. Reads <prefix>_stages.csv / _topics.csv / _system.csv and
prints:
  * per-stage cost ranking: the key column is "core%" = total measured wall
    time in that stage divided by the run duration, i.e. how much of one CPU
    core the stage kept busy on average — the top rows ARE the expensive steps
    (per stage the nodes report 1 s windows of n/mean/max, so "p95(win)" is
    the 95th percentile of the per-window means, not of individual samples);
  * quality/limit metrics (fps vs target, solver iterations, latency);
  * topic rates and message ages;
  * per-process CPU / RSS;
  * heuristic bottleneck hints.
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

DEFAULT_DIR = "/workspace/robot/ros2_ws/logs/perf"


def find_run_prefix(path):
    if os.path.isfile(f"{path}_stages.csv") or path.endswith(
            ("_stages", "_topics", "_system")):
        return path.rsplit("_stages", 1)[0]
    directory = path if os.path.isdir(path) else os.path.dirname(path) or "."
    runs = sorted(glob.glob(os.path.join(directory, "*_stages.csv")))
    if not runs:
        sys.exit(f"no *_stages.csv found under {directory}")
    return runs[-1][: -len("_stages.csv")]


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def percentile(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(int(q * (len(values) - 1) + 0.5), len(values) - 1)
    return values[idx]


def fmt_table(headers, rows):
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    out = ["  ".join(h.ljust(w) for h, w in zip(headers, widths))]
    out.append("  ".join("-" * w for w in widths))
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", nargs="?", default=DEFAULT_DIR,
                    help="log dir or run prefix (default: %(default)s)")
    args = ap.parse_args()
    prefix = find_run_prefix(args.path)
    stages_rows = read_csv(f"{prefix}_stages.csv")
    topics_rows = read_csv(f"{prefix}_topics.csv")
    system_rows = read_csv(f"{prefix}_system.csv")
    if not stages_rows and not system_rows:
        sys.exit(f"run {prefix}: no data rows (did the pipeline publish perf?)")

    times = [float(r["wall_time"]) for r in stages_rows + topics_rows + system_rows]
    duration = max(times) - min(times) if len(times) > 1 else 0.0
    print(f"run:      {prefix}")
    print(f"duration: {duration:.1f} s\n")

    # ---- per-stage aggregation ---------------------------------------------
    stages = defaultdict(lambda: {"n": 0, "total": 0.0, "max": 0.0, "means": []})
    counters = defaultdict(list)
    for r in stages_rows:
        node, stage = r["node"], r["stage"]
        n, mean, peak = int(r["n"]), float(r["mean_ms"]), float(r["max_ms"])
        if stage.startswith("counter/"):
            counters[(node, stage[len("counter/"):])].append(mean)
            continue
        s = stages[(node, stage)]
        s["n"] += n
        s["total"] += n * mean
        s["max"] = max(s["max"], peak)
        s["means"].append(mean)

    timed = {k: v for k, v in stages.items() if k[1].endswith("_ms")}
    metrics = {k: v for k, v in stages.items() if not k[1].endswith("_ms")}

    if timed:
        print("== stage costs (sorted by core%: share of one CPU core) ==")
        rows = []
        for (node, stage), s in sorted(
                timed.items(), key=lambda kv: kv[1]["total"], reverse=True):
            mean = s["total"] / s["n"] if s["n"] else 0.0
            core = s["total"] / (duration * 1000.0) * 100.0 if duration else 0.0
            rows.append([node, stage, str(s["n"]), f"{mean:.2f}",
                         f"{percentile(s['means'], 0.95):.2f}",
                         f"{s['max']:.2f}", f"{core:.1f}"])
        print(fmt_table(
            ["node", "stage", "calls", "mean_ms", "p95(win)", "max_ms", "core%"],
            rows))
        print()

    if metrics or counters:
        print("== metrics / counters (mean over 1 s windows) ==")
        rows = []
        for (node, stage), s in sorted(metrics.items()):
            mean = s["total"] / s["n"] if s["n"] else 0.0
            rows.append([node, stage, f"{mean:.2f}", f"{s['max']:.2f}"])
        for (node, name), vals in sorted(counters.items()):
            rows.append([node, name, f"{sum(vals) / len(vals):.2f}",
                         f"{max(vals):.2f}"])
        print(fmt_table(["node", "metric", "mean", "max"], rows))
        print()

    # ---- topics --------------------------------------------------------------
    if topics_rows:
        agg = defaultdict(list)
        for r in topics_rows:
            agg[r["topic"]].append(float(r["hz"]))
        print("== topics ==")
        rows = [[topic, f"{sum(hz) / len(hz):.1f}", f"{min(hz):.1f}"]
                for topic, hz in sorted(agg.items())]
        print(fmt_table(["topic", "mean_hz", "min_hz"], rows))
        print()

    # ---- system ---------------------------------------------------------------
    proc_agg = defaultdict(lambda: {"cpu": [], "rss": 0.0})
    for r in system_rows:
        key = (r["process"], r["pid"])  # never average two instances together
        proc_agg[key]["cpu"].append(float(r["cpu_pct"]))
        proc_agg[key]["rss"] = max(proc_agg[key]["rss"], float(r["rss_mb"]))
    if proc_agg:
        print("== processes (cpu%: of one core; TOTAL: of all cores / used MB) ==")
        rows = []
        for (proc, pid), a in sorted(
                proc_agg.items(), key=lambda kv: -sum(kv[1]["cpu"])):
            rows.append([proc, pid, f"{sum(a['cpu']) / len(a['cpu']):.1f}",
                         f"{max(a['cpu']):.1f}", f"{a['rss']:.0f}"])
        print(fmt_table(["process", "pid", "cpu_mean", "cpu_max", "rss_mb"], rows))
        print()

    # ---- hints ------------------------------------------------------------------
    hints = []
    for (node, name), vals in counters.items():
        if name != "fps":
            continue
        fps = sum(vals) / len(vals)
        target = counters.get((node, "target_fps"))
        target = sum(target) / len(target) if target else 0.0
        capture = counters.get((node, "capture_fps"))
        capture = sum(capture) / len(capture) if capture else 0.0
        if capture and fps < 0.9 * capture:
            hints.append(
                f"{node}: capturing {capture:.1f} fps but publishing only "
                f"{fps:.1f} — MediaPipe's flow limiter is dropping frames; "
                "inference (mediapipe_ms) is the bottleneck.")
        elif target and fps < 0.9 * target:
            hints.append(
                f"{node}: achieved {fps:.1f} fps of {target:.0f} target — the "
                "capture loop cannot keep up; its biggest stage above is the "
                "bottleneck.")
    for (node, stage), s in timed.items():
        mean = s["total"] / s["n"] if s["n"] else 0.0
        if stage == "latency_capture_to_pose_ms" and mean > 100.0:
            hints.append(
                f"{node}: landmarks arrive {mean:.0f} ms after capture — "
                "upstream (camera/MediaPipe/DDS) dominates end-to-end latency.")
    for (proc, pid), a in proc_agg.items():
        cpu = sum(a["cpu"]) / len(a["cpu"])
        if proc == "TOTAL" and cpu > 85.0:
            hints.append(
                f"system TOTAL cpu {cpu:.0f}% — the whole box is saturated; "
                "per-stage numbers are inflated by scheduling delay.")
        elif proc != "TOTAL" and cpu > 90.0:
            hints.append(
                f"{proc} (pid {pid}): {cpu:.0f}% of a core — likely "
                "single-thread bound (a Python node cannot exceed ~100%).")
    print("== hints ==")
    print("\n".join(f"* {h}" for h in hints) if hints else "* nothing obviously saturated")


if __name__ == "__main__":
    main()
