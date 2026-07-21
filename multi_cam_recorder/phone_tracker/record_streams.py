#!/usr/bin/env python3
"""Record the phone's pose/IMU streams with laptop-clock timestamps.

Remote-controls the app (no phone interaction needed): syncs clocks with an
NTP-style handshake, starts the requested streams pointed back at this
machine, records everything to JSONL, then stops the streams and re-syncs to
measure clock drift over the session.

    python3 record_streams.py <phone_ip>                 # pose + imu
    python3 record_streams.py <phone_ip> --pose          # pose only
    python3 record_streams.py <phone_ip> --imu --rate 400
    python3 record_streams.py <phone_ip> --duration 30 -o run1.jsonl

Stop with Ctrl-C (or --duration). Output lines:

    {"type":"sync","when":"start","offset_ns":...,"rtt_ns":...,...}
    {"type":"pose","t_laptop":...,"t_phone":...,"t_recv":...,"px":...,...}
    {"type":"imu","tag":"acc","t_laptop":...,"t_phone":...,"v":[...],"seq":N}
    {"type":"sync","when":"end","offset_ns":...,"drift_ppm":...,...}

t_laptop = t_phone - offset_ns: the sample's hardware timestamp mapped onto
the laptop wall clock (unix ns) — the ground-truth time base. t_recv is the
packet's arrival time; t_recv - t_laptop is the end-to-end pipeline latency.
"""
import argparse
import json
import select
import socket
import time

from phone_control import ControlError, PhoneControl


def sync_record(sync, when, extra=None):
    rec = {
        "type": "sync",
        "when": when,
        "offset_ns": sync.offset_ns,
        "rtt_ns": sync.rtt_ns,
        "pings": sync.pings,
        "t_wall_ns": sync.t_wall_ns,
        "note": "t_laptop = t_phone - offset_ns (unix ns, laptop clock)",
    }
    if extra:
        rec.update(extra)
    return rec


def pose_record(msg, t_recv_ns, offset_ns):
    """One pose packet -> one log record with calibrated timestamps."""
    rec = {"type": "pose",
           "t_laptop": msg["t"] - offset_ns,
           "t_phone": msg["t"],
           "t_recv": t_recv_ns}
    rec.update((k, v) for k, v in msg.items() if k != "t")
    return rec


def imu_records(msg, offset_ns):
    """One IMU batch packet -> one log record per sample."""
    return [{"type": "imu", "tag": s[0],
             "t_laptop": s[1] - offset_ns,
             "t_phone": s[1],
             "v": s[2:], "seq": msg["seq"]}
            for s in msg["samples"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phone_ip")
    ap.add_argument("-o", "--output", default=None,
                    help="output JSONL path (default: phone_rec_<timestamp>.jsonl)")
    ap.add_argument("--pose", action="store_true", help="record the ARCore pose stream")
    ap.add_argument("--imu", action="store_true", help="record the IMU stream")
    ap.add_argument("--rate", type=int, default=200, help="IMU rate in Hz")
    ap.add_argument("--pose-port", type=int, default=9870)
    ap.add_argument("--imu-port", type=int, default=9871)
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after N seconds (default: run until Ctrl-C)")
    ap.add_argument("--pings", type=int, default=100, help="pings per sync burst")
    args = ap.parse_args()

    # Default to both streams when neither flag is given.
    want_pose = args.pose or not args.imu
    want_imu = args.imu or not args.pose
    out_path = args.output or time.strftime("phone_rec_%Y%m%d_%H%M%S.jsonl")

    # Bind before starting the streams so no packets are lost.
    pose_sock = imu_sock = None
    socks = []
    if want_pose:
        pose_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        pose_sock.bind(("0.0.0.0", args.pose_port))
        socks.append(pose_sock)
    if want_imu:
        imu_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        imu_sock.bind(("0.0.0.0", args.imu_port))
        socks.append(imu_sock)

    ctrl = PhoneControl(args.phone_ip)
    print(f"Syncing clocks with {args.phone_ip} ({args.pings} pings)...")
    sync = ctrl.sync_clock(args.pings)
    offset = sync.offset_ns
    print(f"  offset (phone-laptop): {offset / 1e9:+.6f} s, "
          f"best RTT {sync.rtt_ns / 1e6:.3f} ms "
          f"(uncertainty ~ +/- {sync.rtt_ns / 2e6:.3f} ms)")

    if want_pose:
        ctrl.start_pose(args.pose_port)
    if want_imu:
        ctrl.start_imu(args.imu_port, args.rate)
    started = [n for n, w in (("pose", want_pose), ("imu", want_imu)) if w]
    print(f"Recording {'+'.join(started)} -> {out_path} "
          f"({'Ctrl-C to stop' if args.duration is None else f'{args.duration}s'})")

    n_pose = n_imu = 0
    latency_warned = False
    t_end = None if args.duration is None else time.monotonic() + args.duration
    t_status = time.monotonic() + 2.0

    with open(out_path, "w") as f:
        f.write(json.dumps(sync_record(
            sync, "start",
            {"phone_ip": args.phone_ip, "streams": started, "imu_rate_hz": args.rate},
        )) + "\n")
        try:
            while t_end is None or time.monotonic() < t_end:
                readable, _, _ = select.select(socks, [], [], 0.25)
                for s in readable:
                    data, _ = s.recvfrom(65535)
                    t_recv = time.time_ns()
                    msg = json.loads(data)
                    if s is pose_sock:
                        rec = pose_record(msg, t_recv, offset)
                        f.write(json.dumps(rec) + "\n")
                        n_pose += 1
                        # A huge apparent latency means the phone's ARCore
                        # frame timestamps are not in the CLOCK_BOOTTIME base
                        # the handshake measured (rare, device-specific).
                        lat_ms = (t_recv - rec["t_laptop"]) / 1e6
                        if not latency_warned and not -50 < lat_ms < 500:
                            latency_warned = True
                            print(f"WARNING: pose pipeline latency {lat_ms:.0f} ms — "
                                  "frame timestamps may use a different clock base; "
                                  "t_laptop for poses would then be invalid "
                                  "(IMU samples are unaffected)")
                    else:
                        for rec in imu_records(msg, offset):
                            f.write(json.dumps(rec) + "\n")
                            n_imu += 1
                if time.monotonic() >= t_status:
                    t_status += 2.0
                    print(f"  {n_pose} poses, {n_imu} imu samples", end="\r")
        except KeyboardInterrupt:
            print()
        finally:
            for name, want, stop in (("pose", want_pose, ctrl.stop_pose),
                                     ("imu", want_imu, ctrl.stop_imu)):
                if want:
                    try:
                        stop()
                    except ControlError as e:
                        print(f"warning: could not stop {name} stream: {e}")
            try:
                sync2 = ctrl.sync_clock(args.pings)
                elapsed_ns = sync2.t_wall_ns - sync.t_wall_ns
                drift_ppm = (sync2.offset_ns - offset) / elapsed_ns * 1e6
                f.write(json.dumps(sync_record(
                    sync2, "end", {"drift_ppm": round(drift_ppm, 3)})) + "\n")
                print(f"Clock drift over {elapsed_ns / 1e9:.1f} s: "
                      f"{(sync2.offset_ns - offset) / 1e6:+.3f} ms "
                      f"({drift_ppm:+.2f} ppm)")
            except ControlError as e:
                print(f"warning: end-of-run clock sync failed: {e}")
            ctrl.close()
            for s in socks:
                s.close()

    print(f"Done: {n_pose} poses, {n_imu} imu samples -> {out_path}")


if __name__ == "__main__":
    main()
