# Phone Pose Tracking

Streams the 6DoF pose of an Android phone (ARCore visual-inertial odometry)
over WiFi UDP to a laptop, where it can be printed or republished into ROS2.

```
┌─────────────┐  UDP JSON @ ~30Hz   ┌──────────────────────────┐
│ Android app │ ──────────────────► │ laptop                   │
│ (ARCore VIO)│                     │  udp_listener.py (test)  │
└─────────────┘                     │  phone_pose_bridge (ROS2)│
                                    └──────────────────────────┘
```

## Layout

- `android/` — the phone app (Kotlin, no Android Studio needed)
- `laptop/udp_listener.py` — standalone listener, prints poses
- `laptop/ros2_pose_bridge/` — ROS2 package publishing `PoseStamped` + TF

## Toolchain (already set up on this machine)

Everything lives user-local under `~/Android/`, no sudo required:

- JDK 17: `~/Android/jdk-17`
- Android SDK: `~/Android/Sdk` (platform-tools, android-34, build-tools 34.0.0)

## Build & flash the app

```bash
cd android
JAVA_HOME=~/Android/jdk-17 ./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Phone prerequisites: ARCore-supported device, "Google Play Services for AR"
installed, USB debugging enabled.

## Run

1. Laptop and phone on the same WiFi. Find the laptop IP: `ip addr` (or
   `hostname -I`). Make sure UDP port 9870 is not firewalled
   (`sudo ufw allow 9870/udp` if ufw is active).
2. On the laptop: `python3 laptop/udp_listener.py` — prints incoming poses.
3. Open **Phone Pose Tracker** on the phone, grant camera permission, enter
   the laptop IP and port, tap **Start streaming**. Move the phone; you
   should see position/quaternion updating at ~30 Hz.

## Packet format

One UDP datagram per camera frame, JSON:

```json
{"t": 123456789012, "px": 0.0, "py": 0.0, "pz": 0.0,
 "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0, "state": "TRACKING"}
```

- `t` — phone-side timestamp, nanoseconds (not synced to laptop clock)
- `px..pz` — position in meters, ARCore world frame (Y up, gravity-aligned,
  origin where the session started)
- `qx..qw` — orientation quaternion of the physical camera (X right, Y up,
  -Z forward)
- `state` — `TRACKING` or `PAUSED` (pose is stale/invalid while paused)
- `calib` — calibration counter, +1 per Calibrate press (0 = never)
- `wp` — waypoint counter, +1 per Waypoint press (0 = never)
- `wps` — array of `[x, y, z]` live positions of all waypoint anchors, in
  press order (drift-corrected by ARCore each frame)
- `apx..apz`, `aqx..aqw` — current pose of the ARCore **calibration anchor**
  (same conventions as the phone pose). Present only while an anchor exists
  and is tracking. ARCore continuously corrects the anchor to stay glued to
  the physical calibration spot, so the bridge uses it as a
  drift-compensated reference.

## ROS2 bridge (Docker — no ROS2 needed on the host)

```bash
cd docker
docker compose up -d pose-bridge          # container stays up, launches nothing
docker exec -it phone_pose_bridge bash    # shell: ROS + workspace pre-sourced
cbs phone_pose_bridge && sw               # build + source (aliases from bashrc)
ros2 launch phone_pose_bridge pose_bridge.launch.py
```

The image is ROS2 Jazzy with `laptop/ros2_pose_bridge` bind-mounted into
`/workspace/ros2_ws/src/`; UDP :9870 is published to the host. The launch
starts the bridge node **plus RViz2** on your desktop showing the pose as
moving axes. Host edits to the package take effect on the next build.
Aliases in every shell: `cb`, `cbs`, `cbc`, `ccw`, `sw`.

- Node parameters (topic name, UDP bind address/port, frames, conversion):
  `laptop/ros2_pose_bridge/config/pose_bridge.yaml`
- RViz layout: `laptop/ros2_pose_bridge/rviz/phone_pose.rviz` — if you change
  the topic in the YAML, update the Pose display topic here too.
- No RViz (headless): append `rviz:=false` to the launch command in
  `docker/docker-compose.yaml`, or run it via `docker compose run`.

For an interactive shell instead:

```bash
docker compose run --rm pose-bridge bash
```

Inspect the output from inside the container:

```bash
docker exec -it phone_pose_bridge bash
ros2 topic echo /phone/pose
```

Note: the container's DDS discovery is loopback-only, so other containers
(e.g. handpose3d) won't see `/phone/pose`. To consume the pose in the
handpose3d stack, symlink this package into that workspace instead:

## ROS2 bridge (existing host/container workspace)

Symlink or copy `laptop/ros2_pose_bridge` into your workspace `src/`, then:

```bash
ln -s ~/repo/phone_pose_tracking/laptop/ros2_pose_bridge \
      ~/repo/handpose3d/ros2_ws/src/phone_pose_bridge
cd ~/repo/handpose3d/ros2_ws
colcon build --packages-select phone_pose_bridge
source install/setup.bash
ros2 run phone_pose_bridge pose_bridge
```

Publishes `/phone/pose` (`geometry_msgs/PoseStamped`) and an `odom -> phone`
TF. By default poses are converted from ARCore's convention to REP-103
(X forward, Y left, Z up); set `convert_to_ros:=false` for raw ARCore frames.
The conversion math is in `phone_pose_bridge/frames.py` (run it directly for
its self-test).

## Calibration (teleoperation reference pose)

Notation: `a_T_b` = pose of frame `b` expressed in frame `a`.

Place the phone accurately at the physical reference pose and tap
**Calibrate** in the app. That packet's pose becomes `o_T_ref` (reference in
odom). The operator **body** frame is a fixed, configured offset from the
reference, `ref_T_body` (`body_in_ref` in the YAML: `[x, y, z, roll, pitch,
yaw]`, default `[0, 0.3, 0, pi, 0, 0]`). From then on the bridge publishes

    body_T_target = body_T_phone = inv(ref_T_body) * inv(o_T_ref) * o_T_phone

on `/phone/target_pose` (`PoseStamped`, frame `body`) — the teleop target
for the robot arm in the operator body frame. Details:

- TF tree for visualization: `odom -> phone` (live), `odom -> phone_ref`
  (latched, set on each calibration), `phone_ref -> body` (latched, from
  the YAML). All visible in RViz after the first calibration.
- Re-tap Calibrate any time to re-anchor (e.g. after VIO drift).
- The calibrate signal rides on every packet as a counter (`"calib": N`),
  so a dropped packet delays it by one frame instead of losing it.
- Laptop-side alternative: `ros2 service call /phone/calibrate std_srvs/srv/Trigger`
  anchors to the last received pose.
- The accuracy of the whole chain rests on placing the phone accurately at
  the physical reference pose when calibrating (e.g. a dock or jig).
- **Drift compensation:** the Calibrate press also creates an ARCore anchor
  at the reference. The phone streams the anchor's live pose (`apx..aqw`),
  and the bridge uses it as the reference every frame instead of a frozen
  snapshot — VIO drift that ARCore later corrects (loop closure) is then
  compensated automatically. Without anchor fields (old app), the bridge
  falls back to the frozen-snapshot reference.

## IMU streaming (independent pipeline)

The app can stream the phone's inertial sensors over a **separate UDP
pipeline** (default port 9871), fully independent of the ARCore pose stream:
own socket, own Start/Stop button, own rate selector (50/100/200/400 Hz;
rates above 200 Hz use the `HIGH_SAMPLING_RATE_SENSORS` permission; actual
rates depend on the phone's hardware).

Sensors: calibrated accelerometer / gyroscope / magnetometer + Android's
pre-fused `ROTATION_VECTOR`, `GAME_ROTATION_VECTOR`, `LINEAR_ACCELERATION`
and `GRAVITY`. Samples are batched into one datagram every 20 ms (batch size
therefore scales with the rate automatically), each sample keeping its own
hardware timestamp:

```json
{"type":"imu","seq":42,"samples":[["acc",123456789,0.01,-0.02,9.81],
 ["gyr",123456789,0.001,0.0,-0.002], ["rot",123456789,0.0,0.0,0.7,0.7]]}
```

Tags: `acc` `gyr` `mag` (calibrated trio), `lin` `grv` (fused vectors),
`rot` `grot` (fused quaternions, device-in-ENU). Values are in the Android
sensor frame; the bridge converts to the ROS body frame.

ROS topics (sensor-data QoS):

- `/phone/imu` — `sensor_msgs/Imu`, published at gyro rate: gyro + latest
  accel + orientation from `ROTATION_VECTOR`
- `/phone/mag` — `sensor_msgs/MagneticField` (tesla)
- `/phone/linear_acceleration`, `/phone/gravity` — `Vector3Stamped`
- `/phone/rotation_vector`, `/phone/game_rotation_vector` — `QuaternionStamped`

Timestamps: phone hardware timestamps mapped onto ROS time with a constant
offset estimated from the first sample — inter-sample spacing is exact;
absolute alignment is approximate (no clock sync). Note: the magnetometer
is unreliable near a robot arm (motors, steel); it is streamed but not used
in the composed Imu orientation... the `ROTATION_VECTOR` fusion does use it,
so prefer `/phone/game_rotation_vector` for orientation if mag interference
is visible.

## Remote control & synchronized recording

The app always listens on **UDP :9869** for JSON control commands and replies
to the sender, so the laptop can drive it without touching the phone. When a
start command omits `"host"`, the phone streams back to the **command
sender's IP** — no need to type the laptop IP into the app at all.

| Command | Reply |
|---|---|
| `{"cmd":"ping","t0":N}` | `{"cmd":"pong","t0":N,"t1":recv,"t2":send}` (phone ns) |
| `{"cmd":"start_pose","port":9870}` | `{"cmd":"ack","for":"start_pose","ok":true}` |
| `{"cmd":"stop_pose"}` | ack |
| `{"cmd":"start_imu","port":9871,"rate":200}` | ack |
| `{"cmd":"stop_imu"}` | ack |
| `{"cmd":"calibrate"}` | ack — sets the reference pose + ARCore anchor |
| `{"cmd":"waypoint"}` / `{"cmd":"clear_waypoints"}` | ack |

Starts are idempotent (re-start applies new parameters) and the phone UI
reflects remote commands (buttons/fields update, toasts on
calibrate/waypoint). Remote calibrate is the recommended way to set the
reference: with the phone seated in its jig, nothing touches the screen at
the trigger moment. The anchor is created on the next tracked ARCore frame
(≤ ~33 ms later), so keep the phone still on the reference when triggering.

**Clock sync (handshake):** `t1`/`t2` in the pong are the phone's
`elapsedRealtimeNanos` (CLOCK_BOOTTIME) — the same time base as the IMU
`SensorEvent` timestamps and (on ARCore-certified devices) the pose `t`
field. An NTP-style burst of pings gives the offset `phone − laptop`;
network jitter only inflates the RTT, so the offset is taken from the
near-minimum-RTT pings. On a quiet WiFi expect a best RTT of a few ms, i.e.
offset uncertainty of ~±1–2 ms; the recorder prints both.

**Laptop tools** (`laptop/`, stdlib only):

```bash
python3 laptop/phone_gui.py [phone_ip]                   # web control panel

python3 laptop/phone_control.py <phone_ip> sync          # print clock offset
python3 laptop/phone_control.py <phone_ip> start-imu --rate 400
python3 laptop/phone_control.py <phone_ip> calibrate     # also: waypoint, clear-waypoints
python3 laptop/phone_control.py <phone_ip> stop-imu

python3 laptop/record_streams.py <phone_ip>              # record pose + imu
python3 laptop/record_streams.py <phone_ip> --pose --duration 30 -o run1.jsonl
```

`phone_gui.py` serves a single-page control panel at
`http://localhost:9878` (opens your browser; `--no-browser` to skip) that
drives everything from the laptop: connect + clock sync, start/stop both
streams, Calibrate / Waypoint / Clear buttons, live pose/IMU readouts, and
one-click JSONL recording (same format as `record_streams.py`, including
the start/end sync records and drift). Last-used IP/ports are remembered in
`~/.phone_pose_gui.json`. Note: the panel binds the stream ports itself to
display and record the data, so stop the ROS2 bridge (or use different
ports) while it is running.

`record_streams.py` syncs clocks, remote-starts the requested streams
pointed at itself, records to JSONL, then stops the streams and re-syncs to
report clock drift over the session (also written to the log footer). Log
lines (`t_*` are int nanoseconds):

```json
{"type":"sync","when":"start","offset_ns":..., "rtt_ns":..., ...}
{"type":"pose","t_laptop":..., "t_phone":..., "t_recv":..., "px":..., ...}
{"type":"imu","tag":"acc","t_laptop":..., "t_phone":..., "v":[...], "seq":0}
{"type":"sync","when":"end","offset_ns":..., "drift_ppm":..., ...}
```

`t_laptop = t_phone − offset_ns` is the sample's hardware timestamp mapped
onto the **laptop wall clock** (unix ns, the ground-truth time base);
`t_recv` is the packet arrival time, so `t_recv − t_laptop` is the
end-to-end pipeline latency. If poses show absurd latency the recorder
warns once: that would mean the device's ARCore frame timestamps use a
different clock base than CLOCK_BOOTTIME (rare; IMU samples are unaffected).
For long sessions, the start/end sync records allow linear interpolation of
the offset offline instead of using the single start-time value.

## Waypoints (accuracy validation)

Mark a few physical points with known distances between them. Move the
phone to each and tap **Waypoint**. Each waypoint is its **own ARCore
anchor** (independent of the calibration anchor): the phone streams all of
their live positions (`wps` field, every packet), and the bridge mirrors
them, logging on creation:

    Waypoint 2 (phone anchor): x=+1.002 y=-0.013 z=+0.004, distance to previous point: 1.003 m

`/phone/waypoints` markers in RViz (numbered green spheres + line strip)
update **live** — when ARCore corrects its map you will see the whole
waypoint constellation jump in the odom frame, while the distances between
points stay physically meaningful. Compare them with the tape-measured
spacings to quantify tracking error. Also:

- **Long-press Waypoint** on the phone to clear all waypoints.
- `ros2 service call /phone/log_waypoints std_srvs/srv/Trigger` — print the
  current table of waypoints + consecutive distances + total path length
  (use after corrections settle).
- Max 20 waypoints (keeps the UDP packet under one MTU).
- `/phone/set_waypoint` and `/phone/clear_waypoints` services only apply to
  the legacy mode (old app without `wps` streaming).

### Getting the best tracking accuracy

- Bright, even lighting; avoid darkness and strong backlight.
- Texture-rich surroundings — posters, shelves, clutter. Blank white walls,
  glass and mirrors are VIO poison.
- Move smoothly; avoid covering the camera and avoid pure rotations.
- Let the phone map the workspace for ~10 s (slow sweep) before calibrating.
- Re-calibrate whenever precision matters after long sessions.

## Known limitations

- Screen must stay on with the app foregrounded (ARCore requirement).
- VIO drifts ~1% of distance traveled; world origin resets each session.
- UDP is lossy by design — each packet is a complete state, newest wins.
