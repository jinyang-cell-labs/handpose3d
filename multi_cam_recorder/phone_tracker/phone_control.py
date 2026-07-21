#!/usr/bin/env python3
"""Control client for the Phone Pose Tracker app.

The app listens on UDP :9869 for JSON commands and replies to the sender.
This module wraps the command set (remote start/stop of the pose and IMU
streams) and the NTP-style clock-sync handshake that maps phone timestamps
(elapsedRealtimeNanos, CLOCK_BOOTTIME) onto the laptop wall clock.

Convention: offset_ns = phone_clock - laptop_clock, so

    t_laptop_unix_ns = t_phone_ns - offset_ns

CLI usage:

    python3 phone_control.py <phone_ip> sync [--pings 100]
    python3 phone_control.py <phone_ip> start-pose [--port 9870]
    python3 phone_control.py <phone_ip> stop-pose
    python3 phone_control.py <phone_ip> start-imu [--port 9871] [--rate 200]
    python3 phone_control.py <phone_ip> stop-imu
    python3 phone_control.py <phone_ip> calibrate | waypoint | clear-waypoints
    python3 phone_control.py <phone_ip> lock | unlock
"""
import argparse
import json
import socket
import statistics
import time
from dataclasses import dataclass

CONTROL_PORT = 9869


class ControlError(RuntimeError):
    pass


@dataclass
class ClockSync:
    """Result of one clock-sync handshake burst."""
    offset_ns: int   # phone_clock - laptop_clock (unix ns)
    rtt_ns: int      # best round-trip time seen; offset error is ~ +/- rtt/2
    pings: int       # replies actually received
    t_wall_ns: int   # laptop wall time when the sync finished

    def phone_to_laptop_ns(self, t_phone_ns: int) -> int:
        return t_phone_ns - self.offset_ns


class PhoneControl:
    def __init__(self, phone_ip: str, port: int = CONTROL_PORT, timeout: float = 0.5):
        self.addr = (phone_ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)

    def close(self):
        self.sock.close()

    # -- commands ----------------------------------------------------------

    def start_pose(self, port: int = 9870, host: str | None = None):
        self._rpc({"cmd": "start_pose", "port": port, **({"host": host} if host else {})})

    def stop_pose(self):
        self._rpc({"cmd": "stop_pose"})

    def start_imu(self, port: int = 9871, rate: int = 200, host: str | None = None):
        self._rpc({"cmd": "start_imu", "port": port, "rate": rate,
                   **({"host": host} if host else {})})

    def stop_imu(self):
        self._rpc({"cmd": "stop_imu"})

    def calibrate(self):
        """Set the teleop reference pose (same as the on-screen Calibrate button)."""
        self._rpc({"cmd": "calibrate"})

    def waypoint(self):
        self._rpc({"cmd": "waypoint"})

    def clear_waypoints(self):
        self._rpc({"cmd": "clear_waypoints"})

    def lock(self):
        """Disable the phone's on-screen controls so a stray touch can't
        trigger anything while it is moved as a controller. The control channel
        and both streams keep running regardless."""
        self._rpc({"cmd": "lock"})

    def unlock(self):
        """Re-enable the phone's on-screen controls."""
        self._rpc({"cmd": "unlock"})

    def _rpc(self, msg: dict, retries: int = 4) -> dict:
        """Send a command and wait for its ack; UDP, so retry on silence."""
        payload = json.dumps(msg).encode()
        for _ in range(retries):
            self.sock.sendto(payload, self.addr)
            try:
                while True:
                    data, _ = self.sock.recvfrom(2048)
                    reply = json.loads(data)
                    if reply.get("cmd") == "ack" and reply.get("for") == msg["cmd"]:
                        if not reply.get("ok"):
                            raise ControlError(
                                f"{msg['cmd']} rejected: {reply.get('err')}")
                        return reply
                    # stale packet from an earlier attempt; keep reading
            except socket.timeout:
                continue
        raise ControlError(
            f"no reply from {self.addr[0]}:{self.addr[1]} for {msg['cmd']} "
            f"(is the app open and on the same network?)")

    # -- clock sync ---------------------------------------------------------

    def sync_clock(self, pings: int = 100, spacing_s: float = 0.01) -> ClockSync:
        """NTP-style handshake: laptop t0 -> phone (t1 recv, t2 send) -> laptop t3.

        Per ping: rtt = (t3-t0)-(t2-t1), offset = ((t1-t0)+(t2-t3))/2.
        Network jitter only ever inflates the RTT, so the offset is taken as
        the median over the near-minimum-RTT pings.
        """
        samples = []  # (rtt, offset)
        for _ in range(pings):
            t0 = time.time_ns()
            self.sock.sendto(json.dumps({"cmd": "ping", "t0": t0}).encode(), self.addr)
            try:
                while True:
                    data, _ = self.sock.recvfrom(2048)
                    t3 = time.time_ns()
                    reply = json.loads(data)
                    if reply.get("cmd") == "pong" and reply.get("t0") == t0:
                        break
            except socket.timeout:
                continue
            t1, t2 = reply["t1"], reply["t2"]
            rtt = (t3 - t0) - (t2 - t1)
            offset = ((t1 - t0) + (t2 - t3)) // 2
            samples.append((rtt, offset))
            time.sleep(spacing_s)
        if not samples:
            raise ControlError(
                f"no pong replies from {self.addr[0]}:{self.addr[1]} "
                f"(is the app open and on the same network?)")
        samples.sort()
        rtt_min = samples[0][0]
        # Keep the pings whose RTT is within 20% (+0.2 ms slack) of the best.
        good = [off for rtt, off in samples if rtt <= rtt_min * 1.2 + 200_000]
        return ClockSync(
            offset_ns=int(statistics.median(good)),
            rtt_ns=rtt_min,
            pings=len(samples),
            t_wall_ns=time.time_ns(),
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phone_ip")
    ap.add_argument("command",
                    choices=["sync", "start-pose", "stop-pose", "start-imu", "stop-imu",
                             "calibrate", "waypoint", "clear-waypoints", "lock", "unlock"])
    ap.add_argument("--port", type=int, default=None,
                    help="stream target port (default 9870 pose / 9871 imu)")
    ap.add_argument("--rate", type=int, default=200, help="IMU rate in Hz")
    ap.add_argument("--pings", type=int, default=100, help="pings per sync burst")
    args = ap.parse_args()

    ctrl = PhoneControl(args.phone_ip)
    try:
        if args.command == "sync":
            s = ctrl.sync_clock(args.pings)
            print(f"offset (phone - laptop): {s.offset_ns} ns "
                  f"({s.offset_ns / 1e9:.6f} s)")
            print(f"best RTT: {s.rtt_ns / 1e6:.3f} ms "
                  f"(offset uncertainty ~ +/- {s.rtt_ns / 2e6:.3f} ms), "
                  f"{s.pings}/{args.pings} replies")
        elif args.command == "start-pose":
            ctrl.start_pose(args.port or 9870)
            print("pose stream started (target: this machine)")
        elif args.command == "stop-pose":
            ctrl.stop_pose()
            print("pose stream stopped")
        elif args.command == "start-imu":
            ctrl.start_imu(args.port or 9871, args.rate)
            print(f"imu stream started @ {args.rate} Hz (target: this machine)")
        elif args.command == "stop-imu":
            ctrl.stop_imu()
            print("imu stream stopped")
        elif args.command == "calibrate":
            ctrl.calibrate()
            print("calibrated: current phone pose is the new reference")
        elif args.command == "waypoint":
            ctrl.waypoint()
            print("waypoint set")
        elif args.command == "clear-waypoints":
            ctrl.clear_waypoints()
            print("waypoints cleared")
        elif args.command == "lock":
            ctrl.lock()
            print("screen locked (on-screen controls disabled)")
        elif args.command == "unlock":
            ctrl.unlock()
            print("screen unlocked")
    finally:
        ctrl.close()


if __name__ == "__main__":
    main()
