#!/usr/bin/env python3

"""
A tiny Tkinter GUI that turns ROS 2 service calls into buttons.

This is meant to be *the* service caller for the repo: every service-triggered
feature (calibration, landmark logging, and whatever comes later) gets a button
here instead of a hand-typed ``ros2 service call``. Click a button, the call
fires asynchronously, and the response (`success` / `message`) is shown both
next to the button and in a scrolling log.

It is entirely **config-driven** — the buttons come from a YAML file
(``config_file`` parameter), so adding a new service is a config edit, not a
code change:

    buttons:
      - label: "Calibrate Extrinsics"
        service: "/calibration_extrinsic/calibrate"
        type: "std_srvs/srv/Trigger"     # default if omitted
        group: "Calibration"             # optional; groups buttons in a box

The service *type* is imported dynamically from its ``pkg/srv/Type`` string, and
the response is rendered generically: anything with ``success``/``message``
(Trigger, SetBool, most custom result services) is shown nicely; otherwise the
raw response is stringified. Requests are sent with default field values
(perfect for Trigger-style services).

Threading: rclpy spins in a background thread; Tkinter owns the main thread.
Cross-thread updates go through a queue the GUI drains via ``after()``.
"""

import importlib
import os
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import scrolledtext, ttk

import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

DEFAULT_TYPE = "std_srvs/srv/Trigger"


def load_srv_type(type_str):
    """Import a service type from a ``pkg/srv/Type`` (or ``pkg/Type``) string."""
    parts = [p for p in type_str.split("/") if p]
    if len(parts) == 3:
        pkg, _, name = parts
    elif len(parts) == 2:
        pkg, name = parts
    else:
        raise ValueError(
            f"service type '{type_str}' must be 'pkg/srv/Type' or 'pkg/Type'"
        )
    module = importlib.import_module(f"{pkg}.srv")
    return getattr(module, name)


def format_response(resp):
    """Render a service response generically -> (text, ok).

    Honors the common ``success: bool`` / ``message: string`` result pattern
    (Trigger, SetBool, ...); falls back to the stringified response otherwise.
    """
    has_success = hasattr(resp, "success")
    has_message = hasattr(resp, "message")
    if has_success and has_message:
        return (resp.message or ("success" if resp.success else "failed"),
                bool(resp.success))
    if has_success:
        return (str(resp.success), bool(resp.success))
    if has_message:
        return (str(resp.message), True)
    return (str(resp), True)


@dataclass
class ServiceButton:
    label: str
    service: str
    type_str: str = DEFAULT_TYPE
    group: str = "Services"
    srv_cls: type = None
    client: object = None
    # GUI widgets, wired up when the row is built.
    avail_dot: object = None
    result_label: object = None
    button: object = None


class ServiceCallerNode(Node):
    """Holds the service clients and an availability-polling timer."""

    def __init__(self, buttons, ui_queue):
        super().__init__("gui_service_call")
        self.buttons = buttons
        self.ui_queue = ui_queue
        for spec in self.buttons:
            spec.srv_cls = load_srv_type(spec.type_str)
            spec.client = self.create_client(spec.srv_cls, spec.service)
        # Poll service availability so buttons reflect whether the server is up.
        self.create_timer(1.5, self._poll_availability)
        self.get_logger().info(
            f"gui_service_call ready with {len(self.buttons)} service button(s)"
        )

    def _poll_availability(self):
        statuses = {
            i: spec.client.service_is_ready()
            for i, spec in enumerate(self.buttons)
        }
        self.ui_queue.put(("avail", statuses))

    def call(self, idx):
        """Fire an async service call; result is pushed to the UI queue."""
        spec = self.buttons[idx]
        req = spec.srv_cls.Request()
        future = spec.client.call_async(req)
        future.add_done_callback(
            lambda fut, i=idx: self.ui_queue.put(("result", i, fut))
        )


# Availability dot colors.
DOT_READY = "#2ecc71"     # green
DOT_DOWN = "#7f8c8d"      # grey
OK_COLOR = "#1e8449"      # dark green
FAIL_COLOR = "#c0392b"    # red
BUSY_COLOR = "#b9770e"    # amber


class ServiceCallerGUI:
    def __init__(self, buttons):
        self.buttons = buttons
        self.ui_queue = queue.Queue()

        # --- ROS in a background thread ------------------------------------
        rclpy.init()
        self.node = ServiceCallerNode(buttons, self.ui_queue)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self.executor.spin, daemon=True
        )
        self._spin_thread.start()

        # --- Tk window -----------------------------------------------------
        self.root = tk.Tk()
        self.root.title("ROS 2 Service Caller")
        self.root.minsize(460, 360)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self.root.after(100, self._drain_queue)

    # ------------------------------------------------------------------ build
    def _build_ui(self):
        header = ttk.Label(
            self.root, text="ROS 2 Service Caller",
            font=("TkDefaultFont", 13, "bold"),
        )
        header.pack(anchor="w", padx=10, pady=(10, 4))

        # Group buttons by their `group`, preserving first-seen order.
        groups = []
        by_group = {}
        for i, spec in enumerate(self.buttons):
            if spec.group not in by_group:
                by_group[spec.group] = []
                groups.append(spec.group)
            by_group[spec.group].append(i)

        for group in groups:
            frame = ttk.LabelFrame(self.root, text=group)
            frame.pack(fill="x", padx=10, pady=4)
            for row, idx in enumerate(by_group[group]):
                spec = self.buttons[idx]
                dot = tk.Label(frame, text="●", fg=DOT_DOWN, width=2)
                dot.grid(row=row, column=0, padx=(6, 0), pady=3)
                btn = ttk.Button(
                    frame, text=spec.label, width=22,
                    command=lambda i=idx: self._on_click(i),
                )
                btn.grid(row=row, column=1, padx=4, pady=3, sticky="w")
                result = ttk.Label(frame, text="", foreground="#555")
                result.grid(row=row, column=2, padx=6, pady=3, sticky="w")
                frame.columnconfigure(2, weight=1)
                spec.avail_dot = dot
                spec.button = btn
                spec.result_label = result

        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=(4, 4))
        self.log = scrolledtext.ScrolledText(
            log_frame, height=10, wrap="word", state="disabled"
        )
        self.log.pack(fill="both", expand=True, padx=4, pady=4)

        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(bottom, text="Clear log", command=self._clear_log).pack(
            side="right"
        )

    # ----------------------------------------------------------------- events
    def _on_click(self, idx):
        spec = self.buttons[idx]
        if not spec.client.service_is_ready():
            self._set_result(idx, "service not available", FAIL_COLOR)
            self._log(f"{spec.label}: service {spec.service} not available")
            return
        self._set_result(idx, "calling…", BUSY_COLOR)
        self._log(f"{spec.label}: calling {spec.service} …")
        self.node.call(idx)

    def _drain_queue(self):
        try:
            while True:
                kind, *rest = self.ui_queue.get_nowait()
                if kind == "result":
                    self._handle_result(*rest)
                elif kind == "avail":
                    self._handle_avail(rest[0])
        except queue.Empty:
            pass
        self.root.after(100, self._drain_queue)

    def _handle_result(self, idx, future):
        spec = self.buttons[idx]
        try:
            resp = future.result()
        except Exception as exc:  # noqa: BLE001
            self._set_result(idx, "call failed", FAIL_COLOR)
            self._log(f"{spec.label}: ERROR {exc}")
            return
        text, ok = format_response(resp)
        self._set_result(idx, ("✓ " if ok else "✗ ") + text,
                         OK_COLOR if ok else FAIL_COLOR)
        self._log(f"{spec.label}: {'OK' if ok else 'FAIL'} — {text}")

    def _handle_avail(self, statuses):
        for idx, ready in statuses.items():
            dot = self.buttons[idx].avail_dot
            if dot is not None:
                dot.config(fg=DOT_READY if ready else DOT_DOWN)

    # ---------------------------------------------------------------- helpers
    def _set_result(self, idx, text, color):
        lbl = self.buttons[idx].result_label
        if lbl is not None:
            lbl.config(text=text, foreground=color)

    def _log(self, message):
        stamp = time.strftime("%H:%M:%S")
        self.log.config(state="normal")
        self.log.insert("end", f"[{stamp}] {message}\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _clear_log(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    def _on_close(self):
        try:
            self.executor.shutdown()
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        finally:
            self.root.destroy()

    def run(self):
        self.root.mainloop()


def _load_buttons(config_file):
    if not config_file or not os.path.isfile(config_file):
        raise FileNotFoundError(
            f"config_file '{config_file}' not found; point it at a services.yaml"
        )
    with open(config_file, "r") as fh:
        data = yaml.safe_load(fh) or {}
    raw = data.get("buttons", [])
    if not raw:
        raise ValueError(f"{config_file}: no 'buttons' defined")
    buttons = []
    for entry in raw:
        if "label" not in entry or "service" not in entry:
            raise ValueError(f"button entry missing label/service: {entry}")
        buttons.append(ServiceButton(
            label=entry["label"],
            service=entry["service"],
            type_str=entry.get("type", DEFAULT_TYPE),
            group=entry.get("group", "Services"),
        ))
    return buttons


def main(args=None):
    # Read the config_file parameter via a throwaway node (the GUI owns the real
    # node). Done before building the GUI so a bad config fails fast.
    rclpy.init(args=args)
    param_node = rclpy.create_node("gui_service_call_params")
    config_file = param_node.declare_parameter("config_file", "").value
    param_node.destroy_node()
    rclpy.shutdown()

    buttons = _load_buttons(config_file)
    gui = ServiceCallerGUI(buttons)
    try:
        gui.run()
    except KeyboardInterrupt:
        gui._on_close()


if __name__ == "__main__":
    main()
