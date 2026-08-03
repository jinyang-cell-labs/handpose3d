#!/usr/bin/env python3
"""Multi-camera teleop selector (body_cam_teleop).

Merges the per-camera ``robot_interfaces/TeleopMessage`` streams (one
hand_pose_node instance per camera, each publishing on ``<ns>/teleop``) into
the single ``/teleop_converted`` stream the arm controller consumes.

Selection is per hand and sticky: a camera "offers" a hand while it holds the
trigger button (hand_pose_node presses it only while that hand's pose is
fresh). The mux keeps the current source for a hand as long as it still
offers it, and only then falls over to another offering camera — switching
cameras mid-motion causes a small pose jump (each camera's estimate of the
operator_body-frame pose differs slightly), so it must not happen every frame.

All streams are best-effort KEEP_LAST(1), matching hand_pose_node's output
and the arm controller's sensor-data subscriber.

Diagnostics: source switches are always logged, and now carry the reason a
source stopped offering (no message at all / stale by N s / trigger not held).
``log_offers`` adds a throttled per-source status line so a hand that never
engages can be traced back to the camera that should have offered it, and
``log_gate_summary`` reports how many ticks each source held each hand.
"""
import time

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Pose
from robot_interfaces.msg import TeleopMessage

N_HAND_JOINTS = 25  # robot_interfaces/HandMessage contract


class TeleopMuxNode(Node):
    def __init__(self):
        super().__init__("teleop_mux_node")

        self.input_topics = list(
            self.declare_parameter("input_topics", ["/cam0/teleop"]).value)
        self.output_topic = self.declare_parameter(
            "output_topic", "/teleop_converted").value
        self.publish_hz = float(self.declare_parameter("publish_hz", 50.0).value)
        self.trigger_button_index = int(
            self.declare_parameter("trigger_button_index", 5).value)
        self.num_joy_buttons = int(self.declare_parameter("num_joy_buttons", 16).value)
        # A source whose last message is older than this is dead (node crashed
        # or camera stalled), regardless of what its buttons said.
        self.source_timeout_sec = float(
            self.declare_parameter("source_timeout_sec", 0.5).value)
        self.body_frame = self.declare_parameter("body_frame", "operator_body").value
        if not self.input_topics:
            raise ValueError("input_topics must not be empty")

        # --- diagnostics (opt-in, runtime-settable) --------------------------
        self.log_offers = bool(self.declare_parameter("log_offers", False).value)
        self.log_gate_summary = bool(
            self.declare_parameter("log_gate_summary", False).value)
        self.log_throttle_sec = float(
            self.declare_parameter("log_throttle_sec", 2.0).value)
        self.log_summary_period_sec = float(
            self.declare_parameter("log_summary_period_sec", 5.0).value)
        self._diag_last = {}
        self._gate_counts = {}
        self.add_on_set_parameters_callback(self._on_set_parameters)

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(TeleopMessage, self.output_topic, qos)
        # topic -> (msg, arrival time); single-threaded executor, no lock.
        self.last = {}
        self.subs = [
            self.create_subscription(
                TeleopMessage, topic,
                lambda msg, t=topic: self.last.__setitem__(t, (msg, self.get_clock().now())),
                qos)
            for topic in self.input_topics
        ]
        # Current source topic per hand ("left"/"right"), None = disengaged.
        self.source = {"left": None, "right": None}

        # Drains the counters regardless of log_gate_summary, so the flag can be
        # flipped at runtime without reporting a stale backlog.
        self.gate_timer = self.create_timer(
            max(self.log_summary_period_sec, 0.1), self._publish_gate_summary)
        self.timer = self.create_timer(1.0 / self.publish_hz, self.tick)
        self.get_logger().info(
            f"teleop_mux_node up: {self.input_topics} -> {self.output_topic} "
            f"@ {self.publish_hz:.0f} Hz")

    # ----------------------------------------------------------- diagnostics
    def _on_set_parameters(self, params):
        result = SetParametersResult(successful=True)
        for p in params:
            if p.name in ("log_offers", "log_gate_summary"):
                setattr(self, p.name, bool(p.value))
            elif p.name == "log_throttle_sec":
                if p.value < 0.0:
                    return SetParametersResult(
                        successful=False, reason="log_throttle_sec must be >= 0")
                self.log_throttle_sec = float(p.value)
        return result

    def _diag(self, key, msg, level="info"):
        """Throttled per-key line (the rclpy throttle keys on the call site)."""
        now = time.monotonic()
        if now - self._diag_last.get(key, -1e9) < self.log_throttle_sec:
            return
        self._diag_last[key] = now
        # rclpy caches logger state per call site and rejects a severity change
        # at the same site, so each severity needs its own line here.
        if level == "warn":
            self.get_logger().warn(msg)
        else:
            self.get_logger().info(msg)

    def _publish_gate_summary(self):
        c, self._gate_counts = self._gate_counts, {}
        if not self.log_gate_summary:
            return
        ticks = c.get("ticks", 0)
        held = ", ".join(
            f"{side}:{topic}={n}"
            for (side, topic), n in sorted(
                (k, v) for k, v in c.items() if isinstance(k, tuple)))
        self.get_logger().info(
            f"[mux funnel] {ticks} tick(s); source held: {held or 'none'}")

    def offer_reason(self, topic, side, now):
        """None if this source offers the hand, else why it does not."""
        entry = self.last.get(topic)
        if entry is None:
            return "no message received on this topic"
        msg, stamp = entry
        age = (now - stamp).nanoseconds * 1e-9
        if age > self.source_timeout_sec:
            return (f"stale: last message {age:.2f}s ago > source_timeout_sec "
                    f"{self.source_timeout_sec:.2f} (node crashed or camera stalled)")
        controller = msg.left_controller if side == "left" else msg.right_controller
        buttons = controller.joy.buttons
        if len(buttons) <= self.trigger_button_index:
            return (f"only {len(buttons)} joy button(s), need index "
                    f"{self.trigger_button_index}")
        if buttons[self.trigger_button_index] != 1:
            return (f"trigger button {self.trigger_button_index} not held: that "
                    "camera's hand_pose_node has no pose within its "
                    "pose_timeout_sec (enable its log_trigger for the reason)")
        return None

    def select(self, side, now):
        """Sticky per-hand source selection."""
        current = self.source[side]
        reasons = {t: self.offer_reason(t, side, now) for t in self.input_topics}
        if current is not None and reasons.get(current) is None:
            return current
        candidates = [t for t, reason in reasons.items() if reason is None]
        # Freshest candidate wins the failover.
        chosen = max(
            candidates, key=lambda t: self.last[t][1].nanoseconds, default=None)
        if chosen != current:
            if chosen is None:
                detail = "; ".join(f"{t}: {r}" for t, r in reasons.items())
                self.get_logger().warn(
                    f"{side} hand source: {current} -> None ({detail})")
            else:
                lost = reasons.get(current)
                self.get_logger().info(
                    f"{side} hand source: {current} -> {chosen}"
                    + (f" ({current}: {lost})" if lost else ""))
        self.source[side] = chosen
        if chosen is None and self.log_offers:
            for topic, reason in reasons.items():
                self._diag(
                    f"offer_{side}_{topic}",
                    f"[mux {side}] {topic} not offering: {reason}", "warn")
        return chosen

    def tick(self):
        now = self.get_clock().now()
        self._gate_counts["ticks"] = self._gate_counts.get("ticks", 0) + 1
        out = TeleopMessage()
        out.header.stamp = now.to_msg()
        out.header.frame_id = self.body_frame
        out.head_pose.orientation.w = 1.0  # unused downstream; keep quat valid

        nan = float("nan")
        nan_pose = Pose()
        nan_pose.position.x = nan_pose.position.y = nan_pose.position.z = nan
        nan_pose.orientation.x = nan_pose.orientation.y = nan
        nan_pose.orientation.z = nan_pose.orientation.w = nan

        for side in ("left", "right"):
            controller = out.left_controller if side == "left" else out.right_controller
            hand = out.left_hand if side == "left" else out.right_hand
            topic = self.select(side, now)
            if topic is not None:
                key = (side, topic)
                self._gate_counts[key] = self._gate_counts.get(key, 0) + 1
                msg = self.last[topic][0]
                src_controller = (
                    msg.left_controller if side == "left" else msg.right_controller)
                src_hand = msg.left_hand if side == "left" else msg.right_hand
                controller.pose = src_controller.pose
                controller.joy = src_controller.joy
                hand.joints = list(src_hand.joints) or [nan_pose] * N_HAND_JOINTS
            else:
                controller.joy.buttons = [0] * self.num_joy_buttons
                controller.pose.orientation.w = 1.0
                hand.joints = [nan_pose] * N_HAND_JOINTS

        self.pub.publish(out)


def main():
    rclpy.init()
    node = TeleopMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
