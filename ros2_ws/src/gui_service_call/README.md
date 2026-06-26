# gui_service_call

A small Tkinter GUI that turns ROS 2 service calls into **buttons**. Click a
button → the service is called asynchronously → the response (`success` /
`message`) shows up next to the button and in a scrolling log. A green/grey dot
beside each button tracks whether its service server is currently up.

This is meant to be **the** service caller for the repo: instead of hand-typing
`ros2 service call ...`, every service-triggered feature gets a button here.

## Config-driven (add services without touching code)

Buttons come from a YAML file ([`config/services.yaml`](config/services.yaml)):

```yaml
buttons:
  - label: "Calibrate Extrinsics"
    service: "/calibration_extrinsic/calibrate"
    type: "std_srvs/srv/Trigger"   # optional; this is the default
    group: "Calibration"           # optional; groups buttons into a box
```

- `label` — button text + name used in the log (required)
- `service` — fully-qualified service name (required)
- `type` — service type as `pkg/srv/Type`, imported dynamically (default
  `std_srvs/srv/Trigger`)
- `group` — the labelled box the button sits in (default `Services`)

Requests are sent with **default field values**, which is exactly right for
`Trigger`-style services. Responses are rendered generically: anything with
`success`/`message` (Trigger, SetBool, most custom result services) is shown
nicely; otherwise the raw response is stringified.

### Currently wired up

| Group | Button | Service |
|---|---|---|
| Calibration | Calibrate Intrinsics | `/calibration_intrinsic/calibrate` |
| Calibration | Calibrate Extrinsics | `/calibration_extrinsic/calibrate` |
| Landmark Logging | Start Logging | `/mediapie_landmarks_node/start_log` |
| Landmark Logging | Stop Logging | `/mediapie_landmarks_node/stop_log` |

To add a future service, append an entry to `services.yaml` — no rebuild of the
Python is needed (re-run `colcon build` only to reinstall the updated config, or
point `config_file` at an external yaml).

## Build & run

```bash
# from /workspace/ros2_ws
colcon build --packages-select gui_service_call
source install/setup.bash

ros2 launch gui_service_call gui_service_call.launch.py
# or run directly with a custom config:
ros2 run gui_service_call service_caller_node --ros-args -p config_file:=/path/to/services.yaml
```

Needs an X display (`DISPLAY` set). The buttons work whether or not the target
servers are running — the availability dot is grey until a server appears, and
clicking an unavailable service just logs that it isn't up.
