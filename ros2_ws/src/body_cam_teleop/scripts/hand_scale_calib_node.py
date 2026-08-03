#!/usr/bin/env python3
"""Estimate the optimal hand_size_scaling_factor from multi-camera agreement.

Monocular PnP places each hand somewhere ALONG the camera->hand ray (depth
comes entirely from the assumed hand size), so a wrong
hand_size_scaling_factor slides the cameras' estimates of the SAME physical
hand apart along their respective rays. Rescaling the factor by x moves
camera i's joint estimate J to C_i + x (J - C_i) (C_i = camera centre), so
the x that best collapses the cameras onto each other is scalar least
squares:

    min_x  sum_j || d + x e_j ||^2      d   = C_a - C_b
                                        e_j = (J_a_j - C_a) - (J_b_j - C_b)
    =>  x* = - sum_j (e_j . d) / sum_j (e_j . e_j)

accumulated here over all 21 joints, both hands, every matched frame and
every camera pair of a time window; recommended factor = current * x*. The
linear-ray model is approximate (the factor enters the PnP solve, not a
post-scale), so apply mode iterates damped window updates until x ~ 1.

Inputs (all published by the pipeline when reprojection is on):
  <ns>/body_cam_teleop/markers  hand_<label>_joints SPHERE_LIST, 21 points in
                            <ns>/operator_body, landmark order
  TF                        camera centres; the launch file's identity
                            bridges join the per-camera trees, so everything
                            resolves in <cameras[0]>/operator_body

Modes:
  apply=false (default)  log x*, the recommended factor and the before/after
                         rms cross-camera gap per window; edit the yaml
                         manually with the recommended value.
  apply=true             additionally set hand_size_scaling_factor on every
                         <ns>/hand_pose_node after each window (damped:
                         factor *= x**damping) until |x - 1| < tolerance for
                         convergence_windows consecutive windows. Parameters
                         die with the session: persist the converged value
                         into body_cam_teleop.yaml by hand.

Hold the hand still or move it slowly while calibrating: the per-camera
position filter lags motion, and lag shows up as noise in e_j. The residual
gap left after convergence is cross-camera intrinsics/mount error, which no
scale factor can remove.
"""
import math
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.time import Time

from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import MarkerArray

NUM_LANDMARKS = 21  # MediaPipe hand model


def quat_to_rotation(x, y, z, w):
    """Unit quaternion -> 3x3 rotation matrix."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class HandScaleCalibNode(Node):
    def __init__(self):
        super().__init__("hand_scale_calib_node")

        self.namespaces = [
            str(ns).strip("/") for ns in
            self.declare_parameter("camera_namespaces", ["cam0", "cam1"]).value]
        # TF frame of each camera (hand_pose_node camera_name), parallel to
        # camera_namespaces.
        self.camera_frames = [
            str(f) for f in
            self.declare_parameter("camera_frames", ["camera0", "camera1"]).value]
        body_frame = str(self.declare_parameter("body_frame", "operator_body").value)
        self.apply_mode = bool(self.declare_parameter("apply", False).value)
        # The factor currently in the yaml; x* is a multiplier on it.
        self.factor = float(self.declare_parameter("initial_factor", 1.3).value)
        self.window_sec = float(self.declare_parameter("window_sec", 8.0).value)
        # A window is only evaluated once it holds this many matched joint
        # pairs (21 per hand per matched frame pair).
        self.min_joint_pairs = int(self.declare_parameter("min_joint_pairs", 300).value)
        self.stamp_tol = float(self.declare_parameter("stamp_tol_sec", 0.02).value)
        self.damping = float(self.declare_parameter("damping", 0.5).value)
        self.tolerance = float(self.declare_parameter("tolerance", 0.01).value)
        self.convergence_windows = int(
            self.declare_parameter("convergence_windows", 2).value)
        # Ignore samples for this long after each apply (filter transient).
        self.settle_sec = float(self.declare_parameter("settle_sec", 1.0).value)
        # Optional CSV of the per-window estimates ("" = off).
        self.log_file = str(self.declare_parameter("log_file", "").value)

        if len(self.namespaces) < 2:
            raise ValueError("hand-scale calibration needs at least two cameras")
        if len(self.camera_frames) != len(self.namespaces):
            raise ValueError("camera_frames must parallel camera_namespaces")

        self.fixed_frame = f"{self.namespaces[0]}/{body_frame}"
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # (ns index, hand label) -> deque of (stamp_sec, (21,3) points in the
        # fixed frame); single-threaded executor, no locks.
        self.samples = {
            (i, label): deque(maxlen=15)
            for i in range(len(self.namespaces)) for label in ("Left", "Right")}
        self.subs = [
            self.create_subscription(
                MarkerArray, f"/{ns}/body_cam_teleop/markers",
                lambda msg, i=i: self._on_markers(i, msg), 5)
            for i, ns in enumerate(self.namespaces)]

        # Window accumulators of the least-squares sums (see module docstring).
        self.sum_ed = 0.0
        self.sum_ee = 0.0
        self.sum_dd = 0.0
        self.n_pairs = 0
        self.window_start = self._now_sec()
        self.resume_time = 0.0  # samples before this are dropped (post-apply)
        self.consecutive_ok = 0
        self.converged = False

        self.param_clients = {}
        if self.apply_mode:
            self.param_clients = {
                ns: AsyncParameterClient(self, f"/{ns}/hand_pose_node")
                for ns in self.namespaces}

        self.csv = None
        if self.log_file:
            self.csv = open(self.log_file, "w")
            self.csv.write("stamp_sec,x,factor_recommended,rms_before_mm,"
                           "rms_after_mm,n_joint_pairs,applied\n")

        self.timer = self.create_timer(1.0, self._tick)
        self.get_logger().info(
            f"hand_scale_calib_node up: {self.namespaces} in {self.fixed_frame}, "
            f"mode={'apply' if self.apply_mode else 'log-only'}, "
            f"current factor={self.factor:.4f}, window={self.window_sec:.0f}s. "
            "Show ONE hand to ALL cameras and hold it still / move slowly.")

    # ---- sampling -----------------------------------------------------------

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _camera_center(self, idx):
        """Camera centre in the fixed frame, or None while TF is unavailable."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.fixed_frame, self.camera_frames[idx], Time())
        except TransformException as e:
            self.get_logger().warning(
                f"no TF {self.fixed_frame} -> {self.camera_frames[idx]}: {e}",
                throttle_duration_sec=5.0)
            return None
        t = tf.transform.translation
        return np.array([t.x, t.y, t.z])

    def _to_fixed(self, frame, pts):
        """Transform (21,3) points from `frame` into the fixed frame."""
        if frame == self.fixed_frame:
            return pts
        try:
            tf = self.tf_buffer.lookup_transform(self.fixed_frame, frame, Time())
        except TransformException as e:
            self.get_logger().warning(
                f"no TF {self.fixed_frame} -> {frame}: {e}",
                throttle_duration_sec=5.0)
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        R = quat_to_rotation(q.x, q.y, q.z, q.w)
        return pts @ R.T + np.array([t.x, t.y, t.z])

    def _on_markers(self, idx, msg):
        for marker in msg.markers:
            if not marker.ns.endswith("_joints"):
                continue
            if len(marker.points) != NUM_LANDMARKS:
                continue  # hand not detected this frame (empty sphere list)
            label = marker.ns[len("hand_"):-len("_joints")]
            if (idx, label) not in self.samples:
                continue
            stamp = marker.header.stamp.sec + marker.header.stamp.nanosec * 1e-9
            if stamp < self.resume_time:
                continue
            pts = np.array([[p.x, p.y, p.z] for p in marker.points])
            pts = self._to_fixed(marker.header.frame_id, pts)
            if pts is None:
                continue
            self.samples[(idx, label)].append((stamp, pts))
            # Pair this frame against every LOWER camera index so each
            # unordered camera pair is accumulated once per frame.
            for j in range(idx):
                self._accumulate(idx, j, label, stamp, pts)

    def _accumulate(self, i, j, label, stamp_i, pts_i):
        match = min(
            self.samples[(j, label)],
            key=lambda s: abs(s[0] - stamp_i), default=None)
        if match is None or abs(match[0] - stamp_i) > self.stamp_tol:
            return
        c_i = self._camera_center(i)
        c_j = self._camera_center(j)
        if c_i is None or c_j is None:
            return
        d = c_i - c_j
        e = (pts_i - c_i) - (match[1] - c_j)
        self.sum_ed += float(np.sum(e @ d))
        self.sum_ee += float(np.sum(e * e))
        self.sum_dd += NUM_LANDMARKS * float(d @ d)
        self.n_pairs += NUM_LANDMARKS

    # ---- per-window estimate -------------------------------------------------

    def _tick(self):
        now = self._now_sec()
        if now - self.window_start < self.window_sec:
            return
        if self.n_pairs < self.min_joint_pairs:
            self.get_logger().info(
                f"collecting: {self.n_pairs}/{self.min_joint_pairs} matched joint "
                "pairs — show one hand to all cameras",
                throttle_duration_sec=5.0)
            return  # extend the window until there is enough data
        if self.sum_ee < 1e-9:
            self._reset_window(now)
            return

        x = -self.sum_ed / self.sum_ee
        n = self.n_pairs
        rms_before = math.sqrt(
            max(0.0, self.sum_dd + 2.0 * self.sum_ed + self.sum_ee) / n)
        rms_after = math.sqrt(
            max(0.0, self.sum_dd + 2.0 * x * self.sum_ed + x * x * self.sum_ee) / n)
        if x <= 0.0:
            self.get_logger().warning(
                f"window discarded: x={x:.3f} <= 0 (degenerate geometry or noise; "
                f"{n} joint pairs)")
            self._reset_window(now)
            return

        recommended = self.factor * x
        applied = False
        if self.apply_mode and not self.converged:
            applied = self._apply_update(x)
        self.get_logger().info(
            f"window: x={x:.4f} ({n} joint pairs); cross-camera gap rms "
            f"{rms_before * 1e3:.1f}mm -> {rms_after * 1e3:.1f}mm at x; "
            f"recommended hand_size_scaling_factor = {recommended:.4f}"
            + (f" (applied, damped -> {self.factor:.4f})" if applied else ""))
        if self.csv:
            self.csv.write(
                f"{now:.3f},{x:.6f},{recommended:.6f},{rms_before * 1e3:.2f},"
                f"{rms_after * 1e3:.2f},{n},{int(applied)}\n")
            self.csv.flush()
        self._reset_window(now)

    def _apply_update(self, x):
        if abs(x - 1.0) < self.tolerance:
            self.consecutive_ok += 1
            if self.consecutive_ok >= self.convergence_windows:
                self.converged = True
                self.get_logger().info(
                    f"CONVERGED: persist hand_size_scaling_factor: {self.factor:.4f} "
                    "into body_cam_teleop.yaml (runtime parameters die with the session)")
            return False
        self.consecutive_ok = 0
        self.factor *= x ** self.damping
        param = Parameter(
            "hand_size_scaling_factor", Parameter.Type.DOUBLE, self.factor)
        for ns, client in self.param_clients.items():
            future = client.set_parameters([param])
            future.add_done_callback(
                lambda f, ns=ns: self._on_set_result(ns, f))
        self.resume_time = self._now_sec() + self.settle_sec
        return True

    def _on_set_result(self, ns, future):
        try:
            results = future.result().results
            if not all(r.successful for r in results):
                raise RuntimeError(
                    "; ".join(r.reason for r in results if not r.successful))
        except Exception as e:  # noqa: BLE001 — report and keep calibrating
            self.get_logger().error(f"set_parameters on /{ns}/hand_pose_node: {e}")

    def _reset_window(self, now):
        self.sum_ed = self.sum_ee = self.sum_dd = 0.0
        self.n_pairs = 0
        self.window_start = now


def main():
    rclpy.init()
    node = HandScaleCalibNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.csv:
            node.csv.close()


if __name__ == "__main__":
    main()
