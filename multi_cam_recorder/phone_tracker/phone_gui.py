#!/usr/bin/env python3
"""Web control panel for the Phone Pose Tracker app (stdlib only).

    python3 phone_gui.py [phone_ip] [--http-port 9878] [--no-browser]

Serves one page at http://localhost:9878 that drives the whole setup from
the laptop: connect + clock sync, start/stop the pose and IMU streams
(pointed back at this machine), remote Calibrate / Waypoint / Clear, live
readouts, and JSONL recording with laptop-clock timestamps (same format as
record_streams.py).

The panel binds the stream ports itself to display and record the data, so
stop the ROS2 bridge (or point it at different ports) while it is running.
"""
import argparse
import collections
import http.server
import json
import socket
import threading
import time
import webbrowser
from pathlib import Path

from phone_control import ControlError, PhoneControl
from record_streams import imu_records, pose_record, sync_record

CONFIG_PATH = Path.home() / ".phone_pose_gui.json"
DEFAULT_CFG = {"ip": "192.168.0.110", "pose_port": 9870,
               "imu_port": 9871, "imu_rate": 200}
SYNC_PINGS = 50


class StreamRx:
    """Receives one UDP stream on a background thread."""

    def __init__(self, on_packet):
        self.on_packet = on_packet
        self.sock = None
        self.thread = None
        self.stop_ev = threading.Event()
        self.port = None
        self.count = 0
        self.last = None
        self.last_mono = 0.0

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, port):
        self.stop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        sock.bind(("0.0.0.0", port))
        self.sock, self.port, self.count = sock, port, 0
        self.stop_ev = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_ev.is_set():
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed
            t_recv = time.time_ns()
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            self.count += 1
            self.last = msg
            self.last_mono = time.monotonic()
            self.on_packet(msg, t_recv)

    def stop(self):
        self.stop_ev.set()
        if self.sock:
            self.sock.close()
            self.sock = None
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None


class Backend:
    def __init__(self):
        self.cfg = dict(DEFAULT_CFG)
        try:
            self.cfg.update(json.loads(CONFIG_PATH.read_text()))
        except (OSError, ValueError):
            pass
        self.log_lines = collections.deque(maxlen=200)
        self.log_lock = threading.Lock()
        self.ctrl = None
        self.ctrl_lock = threading.Lock()  # serializes control RPCs
        self.phone_ip = None
        self.sync = None
        self.pose_rx = StreamRx(self._on_pose)
        self.imu_rx = StreamRx(self._on_imu)
        self.imu_acc_count = 0
        self.imu_last = {}
        self.rec_lock = threading.Lock()
        self.rec_file = None
        self.rec_path = None
        self.rec_offset = 0
        self.rec_sync_start = None
        self.rec_poses = 0
        self.rec_imu = 0

    def log(self, msg):
        with self.log_lock:
            self.log_lines.append(f"{time.strftime('%H:%M:%S')}  {msg}")

    def _save_cfg(self):
        try:
            CONFIG_PATH.write_text(json.dumps(self.cfg))
        except OSError:
            pass

    def _require_ctrl(self):
        if self.ctrl is None:
            raise ControlError("connect to the phone first")
        return self.ctrl

    # -- actions (called from HTTP handler threads; raise on failure) -------

    def api(self, req):
        action = req.get("action")
        if action == "connect":
            self.connect(req["ip"].strip())
        elif action == "sync":
            self.do_sync()
        elif action == "start_pose":
            self.start_pose(int(req.get("port", 9870)))
        elif action == "stop_pose":
            self.stop_pose()
        elif action == "start_imu":
            self.start_imu(int(req.get("port", 9871)), int(req.get("rate", 200)))
        elif action == "stop_imu":
            self.stop_imu()
        elif action in ("calibrate", "waypoint", "clear_waypoints"):
            with self.ctrl_lock:
                getattr(self._require_ctrl(), action)()
            self.log(action.replace("_", " ") + " ok")
        elif action == "start_rec":
            self.start_rec(req.get("path") or None)
        elif action == "stop_rec":
            self.stop_rec()
        else:
            raise ControlError(f"unknown action {action!r}")

    def connect(self, ip):
        if not ip:
            raise ControlError("enter the phone IP")
        with self.ctrl_lock:
            if self.ctrl:
                self.ctrl.close()
            self.ctrl = PhoneControl(ip)
            self.phone_ip = ip
        self.cfg["ip"] = ip
        self._save_cfg()
        self.do_sync()

    def do_sync(self):
        self.log(f"syncing clocks with {self.phone_ip} ({SYNC_PINGS} pings)...")
        with self.ctrl_lock:
            self.sync = self._require_ctrl().sync_clock(SYNC_PINGS)
        self.log(f"offset {self.sync.offset_ns / 1e9:+.6f} s, "
                 f"best RTT {self.sync.rtt_ns / 1e6:.2f} ms "
                 f"(uncertainty ~ +/- {self.sync.rtt_ns / 2e6:.2f} ms)")

    def start_pose(self, port):
        ctrl = self._require_ctrl()
        self.pose_rx.start(port)  # bind before the phone starts sending
        try:
            with self.ctrl_lock:
                ctrl.start_pose(port)
        except Exception:
            self.pose_rx.stop()
            raise
        self.cfg["pose_port"] = port
        self._save_cfg()
        self.log(f"pose stream started -> :{port}")

    def stop_pose(self):
        try:
            with self.ctrl_lock:
                self._require_ctrl().stop_pose()
        finally:
            self.pose_rx.stop()
        self.log("pose stream stopped")

    def start_imu(self, port, rate):
        ctrl = self._require_ctrl()
        self.imu_acc_count = 0
        self.imu_last = {}
        self.imu_rx.start(port)
        try:
            with self.ctrl_lock:
                ctrl.start_imu(port, rate)
        except Exception:
            self.imu_rx.stop()
            raise
        self.cfg["imu_port"] = port
        self.cfg["imu_rate"] = rate
        self._save_cfg()
        self.log(f"imu stream started -> :{port} @ {rate} Hz")

    def stop_imu(self):
        try:
            with self.ctrl_lock:
                self._require_ctrl().stop_imu()
        finally:
            self.imu_rx.stop()
        self.log("imu stream stopped")

    # -- recording -----------------------------------------------------------

    def start_rec(self, path):
        if self.sync is None:
            raise ControlError("connect first (recording needs a clock sync)")
        path = path or time.strftime("phone_rec_%Y%m%d_%H%M%S.jsonl")
        streams = [n for n, rx in (("pose", self.pose_rx), ("imu", self.imu_rx))
                   if rx.running]
        with self.rec_lock:
            if self.rec_file:
                raise ControlError("already recording")
            f = open(path, "w")
            f.write(json.dumps(sync_record(
                self.sync, "start",
                {"phone_ip": self.phone_ip, "streams": streams})) + "\n")
            self.rec_file, self.rec_path = f, path
            self.rec_offset = self.sync.offset_ns
            self.rec_sync_start = self.sync
            self.rec_poses = self.rec_imu = 0
        self.log(f"recording -> {path}" +
                 ("" if streams else " (no streams running yet!)"))

    def stop_rec(self, end_sync=True):
        with self.rec_lock:
            f, self.rec_file = self.rec_file, None
        if not f:
            raise ControlError("not recording")
        if end_sync:
            try:
                with self.ctrl_lock:
                    sync2 = self._require_ctrl().sync_clock(SYNC_PINGS)
                elapsed_ns = sync2.t_wall_ns - self.rec_sync_start.t_wall_ns
                drift_ppm = (sync2.offset_ns - self.rec_offset) / elapsed_ns * 1e6
                f.write(json.dumps(sync_record(
                    sync2, "end", {"drift_ppm": round(drift_ppm, 3)})) + "\n")
                self.sync = sync2
                self.log(f"clock drift over {elapsed_ns / 1e9:.1f} s: "
                         f"{(sync2.offset_ns - self.rec_offset) / 1e6:+.3f} ms "
                         f"({drift_ppm:+.2f} ppm)")
            except ControlError as e:
                self.log(f"warning: end-of-recording sync failed: {e}")
        f.close()
        self.log(f"saved {self.rec_poses} poses + {self.rec_imu} imu samples "
                 f"-> {self.rec_path}")

    def _on_pose(self, msg, t_recv):
        with self.rec_lock:
            if self.rec_file:
                self.rec_file.write(
                    json.dumps(pose_record(msg, t_recv, self.rec_offset)) + "\n")
                self.rec_poses += 1

    def _on_imu(self, msg, t_recv):
        for s in msg.get("samples", ()):
            if s[0] == "acc":
                self.imu_acc_count += 1
            self.imu_last[s[0]] = s[2:]
        with self.rec_lock:
            if self.rec_file:
                for rec in imu_records(msg, self.rec_offset):
                    self.rec_file.write(json.dumps(rec) + "\n")
                self.rec_imu += len(msg["samples"])

    # -- state for the page ---------------------------------------------------

    def state(self):
        now = time.monotonic()
        pose_last = self.pose_rx.last
        with self.log_lock:
            log = list(self.log_lines)
        return {
            "phone_ip": self.phone_ip,
            "connected": self.ctrl is not None,
            "sync": None if self.sync is None else {
                "offset_ns": self.sync.offset_ns,
                "rtt_ns": self.sync.rtt_ns,
                "age_s": round((time.time_ns() - self.sync.t_wall_ns) / 1e9),
            },
            "pose": {
                "running": self.pose_rx.running,
                "count": self.pose_rx.count,
                "last": pose_last,
                "age_s": None if pose_last is None
                         else round(now - self.pose_rx.last_mono, 1),
            },
            "imu": {
                "running": self.imu_rx.running,
                "acc_count": self.imu_acc_count,
                "last": self.imu_last,
            },
            "rec": {
                "running": self.rec_file is not None,
                "path": self.rec_path,
                "poses": self.rec_poses,
                "imu": self.rec_imu,
            },
            "cfg": self.cfg,
            "log": log,
        }

    def shutdown(self):
        if self.rec_file:
            try:
                self.stop_rec()
            except ControlError:
                pass
        for stop in (self.stop_pose, self.stop_imu):
            try:
                stop()
            except Exception:
                pass
        if self.ctrl:
            self.ctrl.close()


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Phone Pose Tracker</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;background:#14171c;color:#dde3ea;
      max-width:760px;margin:20px auto;padding:0 14px}
 h1{font-size:18px;margin:0 0 14px}
 fieldset{border:1px solid #2c333d;border-radius:8px;margin:0 0 12px;
          padding:10px 14px}
 legend{color:#8b98a8;font-size:12px;text-transform:uppercase;
        letter-spacing:.08em;padding:0 6px}
 input,select{background:#1d232b;color:#dde3ea;border:1px solid #39424e;
              border-radius:5px;padding:5px 8px;font:inherit}
 input.port{width:70px} input.ip{width:140px} input.path{width:280px}
 button{background:#2b3a4e;color:#e8eef5;border:1px solid #45566b;
        border-radius:5px;padding:5px 14px;font:inherit;cursor:pointer}
 button:hover{background:#375070}
 button.on{background:#7a2e2e;border-color:#9c4444}
 button.rec{background:#2e5a3a;border-color:#437a52}
 button.rec.on{background:#7a2e2e}
 .live{font-family:ui-monospace,monospace;font-size:13px;color:#9fd0a8;
       white-space:pre;margin-top:6px}
 .dim{color:#8b98a8}
 .dot{display:inline-block;width:10px;height:10px;border-radius:50%;
      background:#555;margin-right:6px;vertical-align:baseline}
 .dot.ok{background:#4fc26a}
 #log{font-family:ui-monospace,monospace;font-size:12px;background:#101318;
      border:1px solid #2c333d;border-radius:6px;height:150px;
      overflow-y:auto;padding:8px;white-space:pre-wrap}
 .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
</style></head><body>
<h1><span class="dot" id="conn_dot"></span>Phone Pose Tracker — control panel</h1>

<fieldset><legend>Phone</legend>
 <div class="row">
  IP <input class="ip" id="ip">
  <button onclick="api('connect',{ip:val('ip')})">Connect + sync</button>
  <button onclick="api('sync')">Re-sync clock</button>
  <span class="dim" id="sync_info">not synced</span>
 </div>
</fieldset>

<fieldset><legend>Pose stream (ARCore)</legend>
 <div class="row">
  port <input class="port" id="pose_port">
  <button id="pose_btn" onclick="togglePose()">Start</button>
  <button onclick="api('calibrate')">Calibrate</button>
  <button onclick="api('waypoint')">Waypoint</button>
  <button onclick="api('clear_waypoints')">Clear waypoints</button>
 </div>
 <div class="live" id="pose_live">—</div>
</fieldset>

<fieldset><legend>IMU stream</legend>
 <div class="row">
  port <input class="port" id="imu_port">
  rate <select id="imu_rate"><option>50</option><option>100</option>
       <option>200</option><option>400</option></select> Hz
  <button id="imu_btn" onclick="toggleImu()">Start</button>
 </div>
 <div class="live" id="imu_live">—</div>
</fieldset>

<fieldset><legend>Recording (JSONL, laptop-clock timestamps)</legend>
 <div class="row">
  file <input class="path" id="rec_path" placeholder="auto: phone_rec_<time>.jsonl">
  <button class="rec" id="rec_btn" onclick="toggleRec()">Start recording</button>
  <span class="dim" id="rec_info"></span>
 </div>
</fieldset>

<fieldset><legend>Log</legend><div id="log"></div></fieldset>

<script>
const $=id=>document.getElementById(id), val=id=>$(id).value;
let S=null, inited=false, prevPose=0, prevAcc=0, prevT=0;

async function api(action, params={}){
  try{
    const r=await fetch('/api',{method:'POST',
      body:JSON.stringify(Object.assign({action},params))});
    await r.json();
  }catch(e){}
  refresh();
}
function togglePose(){ api(S&&S.pose.running?'stop_pose':'start_pose',
                           {port:+val('pose_port')}); }
function toggleImu(){ api(S&&S.imu.running?'stop_imu':'start_imu',
                          {port:+val('imu_port'),rate:+val('imu_rate')}); }
function toggleRec(){ api(S&&S.rec.running?'stop_rec':'start_rec',
                          {path:val('rec_path')}); }

function fmt(v,d=2){ return (v>=0?'+':'')+v.toFixed(d); }
function vec(a,d=2){ return a?a.map(x=>fmt(x,d)).join(' '):'-'; }

async function refresh(){
  let s; try{ s=await (await fetch('/state')).json(); }catch(e){ return; }
  const now=performance.now()/1000;
  if(!inited){ inited=true;
    $('ip').value=s.cfg.ip; $('pose_port').value=s.cfg.pose_port;
    $('imu_port').value=s.cfg.imu_port; $('imu_rate').value=s.cfg.imu_rate; }
  $('conn_dot').className='dot'+(s.connected?' ok':'');
  $('sync_info').textContent = s.sync
    ? `offset ${(s.sync.offset_ns/1e9).toFixed(6)} s · RTT ${(s.sync.rtt_ns/1e6).toFixed(2)} ms · ${s.sync.age_s}s ago`
    : 'not synced';
  const dt=Math.max(now-prevT,0.001);
  const poseHz=(s.pose.count-prevPose)/dt, accHz=(s.imu.acc_count-prevAcc)/dt;
  prevPose=s.pose.count; prevAcc=s.imu.acc_count; prevT=now;
  $('pose_btn').textContent=s.pose.running?'Stop':'Start';
  $('pose_btn').className=s.pose.running?'on':'';
  const p=s.pose.last;
  $('pose_live').textContent = !s.pose.running ? '—' : !p ? 'waiting for packets…' :
    `${p.state}  ${poseHz.toFixed(1)} Hz  p=(${fmt(p.px)} ${fmt(p.py)} ${fmt(p.pz)})\\n`+
    `calib #${p.calib}${p.apx!==undefined?' (anchor ok)':''}   waypoints: ${(p.wps||[]).length}`;
  $('imu_btn').textContent=s.imu.running?'Stop':'Start';
  $('imu_btn').className=s.imu.running?'on':'';
  const im=s.imu.last||{};
  $('imu_live').textContent = !s.imu.running ? '—' :
    `acc ~${accHz.toFixed(0)} Hz  acc=(${vec(im.acc)})  gyr=(${vec(im.gyr,3)})\\n`+
    `rot=(${vec(im.rot)})  grot=(${vec(im.grot)})`;
  $('rec_btn').textContent=s.rec.running?'Stop recording':'Start recording';
  $('rec_btn').className='rec'+(s.rec.running?' on':'');
  $('rec_info').textContent = s.rec.running
    ? `● ${s.rec.poses} poses, ${s.rec.imu} imu → ${s.rec.path}`
    : (s.rec.path?`last: ${s.rec.path}`:'');
  const log=$('log'), atEnd=log.scrollTop+log.clientHeight>=log.scrollHeight-8;
  log.textContent=s.log.join('\\n');
  if(atEnd) log.scrollTop=log.scrollHeight;
  S=s;
}
setInterval(refresh, 400); refresh();
</script></body></html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    backend = None

    def log_message(self, *args):  # silence per-request stderr spam
        pass

    def _send(self, code, body, ctype):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/state":
            self._send(200, json.dumps(self.backend.state()), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/api":
            self._send(404, "not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
            self.backend.api(req)
            self._send(200, '{"ok":true}', "application/json")
        except Exception as e:
            self.backend.log(f"ERROR: {e}")
            self._send(200, json.dumps({"ok": False, "err": str(e)}),
                       "application/json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phone_ip", nargs="?",
                    help="phone IP to prefill and auto-connect to")
    ap.add_argument("--http-port", type=int, default=9878)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    backend = Backend()
    Handler.backend = backend
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.http_port), Handler)
    url = f"http://127.0.0.1:{args.http_port}"
    print(f"Phone Pose control panel: {url}  (Ctrl-C to quit)")

    if args.phone_ip:
        backend.cfg["ip"] = args.phone_ip
        threading.Thread(
            target=lambda: _try(backend, args.phone_ip), daemon=True).start()
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("shutting down (stopping streams)...")
        backend.shutdown()


def _try(backend, ip):
    try:
        backend.connect(ip)
    except Exception as e:
        backend.log(f"auto-connect failed: {e}")


if __name__ == "__main__":
    main()
