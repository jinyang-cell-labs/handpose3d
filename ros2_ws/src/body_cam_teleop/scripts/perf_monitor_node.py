#!/usr/bin/env python3
"""Pipeline performance recorder for body_cam_teleop.

Records three views of the running pipeline to CSV files (one set per run)
so the expensive steps can be identified offline with perf_report.py:

  1. Per-stage wall times published by the pipeline nodes themselves on their
     ``body_cam_teleop/perf`` topics (JSON std_msgs/String, enable_perf: true —
     capture / convert / mediapipe in hand_landmarks_node; undistort / Ceres
     solve / reprojection in hand_pose_node).
  2. Message rate of the small pipeline topics (``*/body_cam_teleop/landmarks``,
     ``*/teleop``, ``/teleop_converted``), counted on RAW (serialized)
     subscriptions so the monitor itself stays cheap. The raw image topics
     are deliberately NOT subscribed: one extra 1280x720 bgr8 subscriber adds
     ~80 MB/s of DDS load per camera and would distort the measurement. The
     pose node's ``image_cb_ms`` stage count is the received-image rate, and
     its ``latency_capture_to_pose_ms`` stage covers end-to-end message age.
  3. Per-process CPU (percent of one core, like top) / RSS / thread count of
     every process whose cmdline matches ``process_patterns``, sampled from
     /proc, plus a TOTAL row (all-core busy %, system used memory).

Files (in ``log_dir``, timestamp prefix shared per run):
  <ts>_stages.csv   wall_time,node,stage,n,mean_ms,max_ms
  <ts>_topics.csv   wall_time,topic,hz
  <ts>_system.csv   wall_time,process,pid,cpu_pct,rss_mb,threads

Run standalone while the pipeline is up:
  ros2 run body_cam_teleop perf_monitor_node.py
or launch everything together:
  ros2 launch body_cam_teleop body_cam_teleop.launch.py perf:=true
"""
import csv
import json
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String

CLK_TCK = os.sysconf("SC_CLK_TCK")
N_CPUS = os.cpu_count() or 1


def _read_proc_stat_total():
    """(total_jiffies, idle_jiffies) across all cpus from /proc/stat."""
    with open("/proc/stat") as f:
        fields = [int(x) for x in f.readline().split()[1:]]
    return sum(fields), fields[3] + (fields[4] if len(fields) > 4 else 0)


def _read_pid_jiffies(pid):
    """utime+stime of a process; None if it exited."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            raw = f.read()
    except OSError:
        return None
    # comm (field 2) may contain spaces/parens: split after the closing ')'.
    fields = raw[raw.rfind(")") + 2:].split()
    return int(fields[11]) + int(fields[12])  # utime, stime (fields 14, 15)


def _read_pid_mem_threads(pid):
    rss_mb, threads = 0.0, 0
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_mb = int(line.split()[1]) / 1024.0
                elif line.startswith("Threads:"):
                    threads = int(line.split()[1])
    except OSError:
        pass
    return rss_mb, threads


def _read_mem_used_mb():
    total, avail = 0, 0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                total = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                avail = int(line.split()[1])
    return (total - avail) / 1024.0


class TopicRate:
    """Arrival counter for one topic (raw subscription: bytes, not parsed)."""

    def __init__(self):
        self.count = 0

    def snapshot_and_reset(self, window_sec):
        hz = self.count / window_sec if window_sec > 0 else 0.0
        self.count = 0
        return hz


class PerfMonitorNode(Node):
    def __init__(self):
        super().__init__("perf_monitor_node")
        log_dir = self.declare_parameter(
            "log_dir", "/workspace/robot/ros2_ws/logs/perf").value
        self.sample_period = float(
            self.declare_parameter("sample_period_sec", 1.0).value)
        self.summary_period = float(
            self.declare_parameter("summary_period_sec", 10.0).value)
        # Substrings matched against /proc/<pid>/cmdline to find the pipeline
        # processes (per-camera instances are told apart by their __ns remap).
        self.process_patterns = list(
            self.declare_parameter(
                "process_patterns",
                ["hand_landmarks_node", "hand_pose_node", "teleop_mux_node",
                 "perf_monitor_node", "rviz2"]).value)
        # Topics to rate-track, matched by suffix against the ROS graph.
        self.rate_topic_suffixes = list(
            self.declare_parameter(
                "rate_topic_suffixes",
                ["body_cam_teleop/landmarks", "teleop", "teleop_converted"]).value)

        os.makedirs(log_dir, exist_ok=True)
        prefix = time.strftime("%Y%m%d_%H%M%S")
        tag = str(self.declare_parameter("run_tag", "").value).strip()
        if tag:
            prefix += f"_{tag}"
        self.run_prefix = os.path.join(log_dir, prefix)

        self.stages_f = open(f"{self.run_prefix}_stages.csv", "w", newline="")
        self.stages_csv = csv.writer(self.stages_f)
        self.stages_csv.writerow(
            ["wall_time", "node", "stage", "n", "mean_ms", "max_ms"])
        self.topics_f = open(f"{self.run_prefix}_topics.csv", "w", newline="")
        self.topics_csv = csv.writer(self.topics_f)
        self.topics_csv.writerow(["wall_time", "topic", "hz"])
        self.system_f = open(f"{self.run_prefix}_system.csv", "w", newline="")
        self.system_csv = csv.writer(self.system_f)
        self.system_csv.writerow(
            ["wall_time", "process", "pid", "cpu_pct", "rss_mb", "threads"])

        self.perf_subs = {}    # topic -> subscription
        self.rate_subs = {}    # topic -> (subscription, TopicRate)
        self.pids = {}         # pid -> label
        self.pid_jiffies = {}  # pid -> last utime+stime
        self.last_total_jiffies, self.last_idle_jiffies = _read_proc_stat_total()
        self.last_sample_time = time.monotonic()
        self.latest_perf = {}  # node -> parsed perf payload (for the summary)

        self.discover_timer = self.create_timer(2.0, self._discover)
        self.sample_timer = self.create_timer(self.sample_period, self._sample)
        self.summary_timer = self.create_timer(self.summary_period, self._summary)
        self._discover()
        self.get_logger().info(
            f"recording to {self.run_prefix}_{{stages,topics,system}}.csv — "
            f"analyze with: python3 perf_report.py {self.run_prefix}")

    # ---------------------------------------------------------- discovery
    def _discover(self):
        """Subscribe to new perf/rate topics and (re)find pipeline PIDs."""
        for topic, types in self.get_topic_names_and_types():
            if (topic.endswith("/perf") and "std_msgs/msg/String" in types
                    and topic not in self.perf_subs):
                self.perf_subs[topic] = self.create_subscription(
                    String, topic, self._on_perf, 10)
                self.get_logger().info(f"tracking perf topic {topic}")
            elif (any(topic.endswith("/" + s) or topic == "/" + s
                      for s in self.rate_topic_suffixes)
                    and topic not in self.rate_subs):
                try:
                    msg_type = get_message(types[0])
                except (AttributeError, ModuleNotFoundError, ValueError) as exc:
                    self.get_logger().warn(f"cannot import {types[0]}: {exc}")
                    continue
                tracker = TopicRate()
                # Best-effort matches both reliable and best-effort publishers;
                # raw=True skips deserialization (we only count arrivals).
                qos = QoSProfile(
                    depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)
                sub = self.create_subscription(
                    msg_type, topic,
                    lambda _msg, t=tracker: self._on_rated(t), qos, raw=True)
                self.rate_subs[topic] = (sub, tracker)
                self.get_logger().info(f"tracking rate of {topic}")

        for pid in filter(str.isdigit, os.listdir("/proc")):
            pid = int(pid)
            if pid in self.pids:
                continue
            # Match on the executable name — comm (kernel-truncated to 15
            # chars) plus the basenames of the first two cmdline args (Python
            # nodes run as "python3 /path/to/node.py", so comm is just
            # "python3"). Deliberately NOT the full cmdline: wrappers like
            # "ros2 run ... hand_pose_node" or the launching shell carry the
            # node name in later args and would be double-counted.
            try:
                with open(f"/proc/{pid}/comm") as f:
                    comm = f.read().strip()
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    argv = [a.decode(errors="replace")
                            for a in f.read().split(b"\0") if a]
            except OSError:
                continue
            if not argv:
                continue  # kernel thread or zombie
            names = [os.path.basename(a) for a in argv[:2]]
            for pattern in self.process_patterns:
                if not any(
                        n.startswith(pattern)
                        or (len(comm) == 15 and pattern.startswith(comm) and n == comm)
                        for n in [comm] + names):
                    continue
                label = pattern
                for arg in argv:
                    if arg.startswith("__ns:="):
                        label += "@" + arg.split(":=", 1)[1]
                self.pids[pid] = label
                self.get_logger().info(f"tracking process {label} (pid {pid})")
                break

    # ---------------------------------------------------------- callbacks
    def _on_perf(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"bad perf JSON: {exc}")
            return
        wall = time.time()
        node = payload.get("node", "?")
        self.latest_perf[node] = payload
        for stage, s in payload.get("stages", {}).items():
            self.stages_csv.writerow(
                [f"{wall:.3f}", node, stage, s.get("n", 0),
                 s.get("mean_ms", 0.0), s.get("max_ms", 0.0)])
        for name, value in payload.get("counters", {}).items():
            self.stages_csv.writerow(
                [f"{wall:.3f}", node, f"counter/{name}", 1, value, value])
        self.stages_f.flush()

    def _on_rated(self, tracker):
        tracker.count += 1

    # ---------------------------------------------------------- sampling
    def _sample(self):
        now_mono = time.monotonic()
        window = now_mono - self.last_sample_time
        self.last_sample_time = now_mono
        wall = time.time()

        for topic, (_, tracker) in self.rate_subs.items():
            hz = tracker.snapshot_and_reset(window)
            self.topics_csv.writerow([f"{wall:.3f}", topic, f"{hz:.2f}"])
        self.topics_f.flush()

        total, idle = _read_proc_stat_total()
        d_total = max(total - self.last_total_jiffies, 1)
        d_idle = idle - self.last_idle_jiffies
        self.last_total_jiffies, self.last_idle_jiffies = total, idle

        for pid in list(self.pids):
            jiffies = _read_pid_jiffies(pid)
            if jiffies is None:
                self.get_logger().info(
                    f"process {self.pids[pid]} (pid {pid}) exited")
                del self.pids[pid]
                self.pid_jiffies.pop(pid, None)
                continue
            last = self.pid_jiffies.get(pid)
            self.pid_jiffies[pid] = jiffies
            if last is None:
                continue
            # Percent of ONE core, like top: (process jiffies)/(wall jiffies).
            cpu_pct = 100.0 * (jiffies - last) * N_CPUS / d_total
            rss_mb, threads = _read_pid_mem_threads(pid)
            self.system_csv.writerow(
                [f"{wall:.3f}", self.pids[pid], pid,
                 f"{cpu_pct:.1f}", f"{rss_mb:.1f}", threads])
        self.system_csv.writerow(
            [f"{wall:.3f}", "TOTAL", "",
             f"{100.0 * (d_total - d_idle) / d_total:.1f}",
             f"{_read_mem_used_mb():.1f}", ""])
        self.system_f.flush()

    def _summary(self):
        lines = []
        for node, payload in sorted(self.latest_perf.items()):
            stages = payload.get("stages", {})
            timed = [(k, v) for k, v in stages.items() if k.endswith("_ms")]
            timed.sort(key=lambda kv: kv[1].get("mean_ms", 0.0), reverse=True)
            top = ", ".join(
                f"{k} {v.get('mean_ms', 0.0):.1f}ms(n={v.get('n', 0)})"
                for k, v in timed[:3])
            counters = payload.get("counters", {})
            fps = counters.get("fps")
            fps_txt = f", fps {fps:.1f}/{counters.get('target_fps', 0):.0f}" \
                if fps is not None else ""
            lines.append(f"  {node}: {top}{fps_txt}")
        if lines:
            self.get_logger().info("perf summary:\n" + "\n".join(lines))

    def shutdown(self):
        for f in (self.stages_f, self.topics_f, self.system_f):
            f.close()
        self.get_logger().info(
            f"perf CSVs written: {self.run_prefix}_{{stages,topics,system}}.csv")


def main(args=None):
    rclpy.init(args=args)
    node = PerfMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
