import os
import cv2
import cv2.aruco as aruco
import numpy as np
from pymavlink import mavutil
import time
import math
import threading
import csv
import traceback
from datetime import datetime
from picamera2 import Picamera2

# ----------------- 1. 系統與相機參數設定 -----------------
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# 實體鏡頭方向已轉正，關閉軟體旋轉
ROTATE_MODE = None

MAVLINK_PORT = '/dev/ttyS0'
MAVLINK_BAUD = 57600

# ----------------- 實測標定之內參矩陣與畸變係數 (RMS: 0.88px) -----------------
CAMERA_MATRIX = np.array([
    [996.7714577681224,   0.0,               648.1193015712994],
    [  0.0,             996.7635118078291,   348.73488146794733],
    [  0.0,               0.0,                 1.0]
], dtype=np.float32)

DIST_COEFFS = np.array([
    [0.20590003614264327, -0.5582882506247375, -0.00026368475249116663, -7.103303081480804e-05, 0.4427011387711153]
], dtype=np.float32)

# ----------------- 2. 雙標籤 (Nested Tag) 參數定義 -----------------
# ID 0: 低空 AprilTag 小標籤 (10cm)
# ID 1: 高空 AprilTag 大標籤 (80cm)
TARGETS = {
    0: {"size": 0.1, "obj_points": None},
    1: {"size": 0.80, "obj_points": None}
}

for tid, info in TARGETS.items():
    h_size = info["size"] / 2.0
    info["obj_points"] = np.array([
        [-h_size,  h_size, 0],
        [ h_size,  h_size, 0],
        [ h_size, -h_size, 0],
        [-h_size, -h_size, 0]
    ], dtype=np.float32)

# ----------------- 2.5 深度 (Z) 估測參數 -----------------
# solvePnP IPPE_SQUARE returns two candidate poses (plane-flip ambiguity).
# When their reprojection errors are within this ratio the pose is ambiguous
# and the size-based depth is preferred instead.
PNP_AMBIGUITY_RATIO = 1.5
# Best-solution reprojection error above this (px) means a poor fit: distrust PnP depth.
PNP_MAX_REPROJ_PX = 2.0
# Temporal gate: a new Z must be reachable from the last accepted Z at this
# vertical speed, plus a relative noise allowance, as long as the last accepted
# Z is recent. Both estimators failing the gate drops the frame.
Z_MAX_RATE_MPS = 3.0
Z_GATE_REL = 0.15
Z_GATE_MAX_DT = 1.0
# Cap on acquire/loss frame snapshots saved per run (protects the SD card).
MAX_SAVED_FRAMES = 100

# ----------------- 3. Picamera2 官方原生影像讀取器 -----------------
class NonBlockingPiCameraReader:
    def __init__(self, width=1280, height=720, logger=None):
        self.width = width
        self.height = height
        self.logger = logger
        
        self.picam2 = Picamera2()
        sensor_config = {"output_size": (1640, 1232)}
        
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            sensor=sensor_config
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        # 預熱讓硬體自動曝光 (AEC) 與白平衡 (AWB) 穩定
        time.sleep(1.0)

        if self.logger:
            self.logger.camera_connect(30.0)

        self.latest_frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        rx_count = 0
        rx_start = time.monotonic()

        while not self.stopped:
            frame_rgb = self.picam2.capture_array()
            if frame_rgb is not None:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                if ROTATE_MODE is not None:
                    frame_bgr = cv2.rotate(frame_bgr, ROTATE_MODE)

                with self.lock:
                    self.latest_frame = frame_bgr
                self.new_frame_event.set()

                rx_count += 1
                now = time.monotonic()
                elapsed = now - rx_start
                if elapsed >= 5.0:
                    rx_msg = f"[RX] Picamera2 Native FPS: {rx_count / elapsed:.1f}"
                    if self.logger:
                        self.logger.event(rx_msg)
                    else:
                        print(rx_msg, flush=True)
                    rx_count = 0
                    rx_start = now
            else:
                time.sleep(0.005)

    def read_new(self, timeout=0.5):
        if not self.new_frame_event.wait(timeout):
            return False, None
        with self.lock:
            self.new_frame_event.clear()
            frame = self.latest_frame
            self.latest_frame = None
            if frame is None:
                return False, None
            return True, frame

    def stop(self):
        self.stopped = True
        try:
            self.picam2.stop()
            self.picam2.close()
        except Exception:
            pass

# ----------------- 4. 本地飛行日誌 (FlightLogger) -----------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

class FlightLogger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(LOG_DIR, f"landing_{stamp}")
        suffix = 0
        while True:
            run_dir = base if suffix == 0 else f"{base}_{suffix}"
            if not os.path.exists(run_dir):
                break
            suffix += 1
        os.makedirs(run_dir)
        self.run_dir = run_dir
        self.text_path = os.path.join(run_dir, "run.log")
        self.csv_path = os.path.join(run_dir, "targets.csv")
        self.summary_path = os.path.join(run_dir, "summary.txt")

        self._lock = threading.Lock()
        self._text = open(self.text_path, "w", buffering=1)
        self._csv_file = open(self.csv_path, "w", buffering=1, newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow([
            "wall_time_iso", "t_rel_s", "tag_id",
            "front_m", "right_m", "down_m",
            "distance_m", "angle_x_rad", "angle_y_rad",
            "detect_ms", "proc_ms",
            "raw_z_m", "stable_z_m", "z_source", "pnp_reproj_px", "pnp_ambiguity",
        ])

        self._start_mono = time.monotonic()
        self._start_wall = time.time()

        self._sends_total = 0
        self._per_tag = {}
        self._send_gap = {"n": 0, "sum": 0.0, "max": 0.0}
        self._last_send_mono = None

        self._tracks = 0
        self._tracked_time = 0.0
        self._track_durations = []
        self._acquire_events = []
        self._loss_events = []
        self._handoffs = []

        self._frames_total = 0
        self._detect_time_total = 0.0
        self._proc_time_total = 0.0
        self._proc_time_peak = 0.0

        self._outliers = 0
        self._outlier_z_min = None
        self._outlier_z_max = None
        self._pnp_failures = 0
        self._camera_connects = 0

        self._open_track = None
        self._last_checkpoint_mono = 0.0

        # Z-estimator diagnostics
        self._z_source_counts = {}
        self._z_diff_sum = 0.0
        self._z_diff_max = 0.0
        self._z_diff_n = 0
        self._pnp_ambiguous = 0
        self._gate_rejections = 0

        # acquire / loss frame snapshots
        self.frames_dir = os.path.join(run_dir, "frames")
        self._frames_saved = 0

        self.event(f"[LOG] Writing to {self.text_path}", console=False)

    def _rel(self):
        return time.monotonic() - self._start_mono

    def event(self, message, console=True):
        with self._lock:
            self._text.write(f"[{self._rel():9.2f}s] {message}\n")
        if console:
            print(message, flush=True)

    def camera_connect(self, fps):
        with self._lock:
            self._camera_connects += 1
        self.event(f"[SYSTEM] Camera FPS: {fps}")

    def rate_report(self, message, window_s, frames, sends, detect_sum, proc_sum, proc_max):
        self._frames_total += frames
        self._detect_time_total += detect_sum
        self._proc_time_total += proc_sum
        self._proc_time_peak = max(self._proc_time_peak, proc_max)
        self.event(message)
        self._sync_to_disk()
        if time.monotonic() - self._last_checkpoint_mono >= 10.0:
            self._last_checkpoint_mono = time.monotonic()
            self._write_checkpoint()

    def target_sent(self, tag_id, x, y, z, distance, angle_x, angle_y, detect_s, proc_s,
                    raw_z=None, stable_z=None, z_source="", reproj_px=None, ambiguity=None):
        now_mono = time.monotonic()
        self._sends_total += 1
        self._z_source_counts[z_source] = self._z_source_counts.get(z_source, 0) + 1
        if raw_z is not None and stable_z is not None and not math.isnan(stable_z):
            diff = abs(raw_z - stable_z)
            self._z_diff_sum += diff
            self._z_diff_max = max(self._z_diff_max, diff)
            self._z_diff_n += 1
        if ambiguity is not None and ambiguity < PNP_AMBIGUITY_RATIO:
            self._pnp_ambiguous += 1
        stats = self._per_tag.setdefault(tag_id, {"n": 0, "z_min": z, "z_max": z, "z_sum": 0.0})
        stats["n"] += 1
        stats["z_min"] = min(stats["z_min"], z)
        stats["z_max"] = max(stats["z_max"], z)
        stats["z_sum"] += z
        if self._open_track is not None:
            self._open_track[2] = z
        if self._last_send_mono is not None:
            gap = now_mono - self._last_send_mono
            if gap < 2.0:
                self._send_gap["n"] += 1
                self._send_gap["sum"] += gap
                self._send_gap["max"] = max(self._send_gap["max"], gap)
        self._last_send_mono = now_mono
        self._csv.writerow([
            datetime.now().isoformat(timespec="milliseconds"),
            f"{self._rel():.3f}", tag_id,
            f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
            f"{distance:.4f}", f"{angle_x:.5f}", f"{angle_y:.5f}",
            f"{detect_s * 1000:.2f}", f"{proc_s * 1000:.2f}",
            f"{raw_z:.4f}" if raw_z is not None else "",
            f"{stable_z:.4f}" if stable_z is not None else "",
            z_source,
            f"{reproj_px:.3f}" if reproj_px is not None else "",
            f"{ambiguity:.3f}" if ambiguity is not None else "",
        ])

    def track_acquired(self, tag_id, z, frame=None):
        self._tracks += 1
        t_rel = self._rel()
        self._acquire_events.append((t_rel, tag_id, z))
        self._open_track = [tag_id, time.monotonic(), z]
        self.event(f"[TRACK] Target acquired: ID {tag_id} at Z={z:.2f} m")
        self._sync_to_disk()
        self.save_frame(frame, f"acquired_{t_rel:.1f}s_ID{tag_id}_Z{z:.1f}m.jpg")

    def track_handoff(self, from_tag, to_tag, z):
        self._handoffs.append((self._rel(), from_tag, to_tag, z))
        if self._open_track is not None:
            self._open_track[0] = to_tag
            self._open_track[2] = z
        self.event(f"[TRACK] Tag handoff ID {from_tag} -> ID {to_tag} at Z={z:.2f} m")
        self._sync_to_disk()

    def track_lost(self, tag_id, last_z, duration, last_seen_frame=None, current_frame=None):
        self._tracked_time += duration
        self._track_durations.append(duration)
        t_rel = self._rel()
        self._loss_events.append((t_rel, tag_id, last_z, duration))
        self._open_track = None
        self.event(f"[TRACK] Target lost: ID {tag_id}, last Z={last_z:.2f} m, tracked {duration:.1f} s", console=False)
        self._sync_to_disk()
        stem = f"lost_{t_rel:.1f}s_ID{tag_id}_Z{last_z:.1f}m"
        self.save_frame(last_seen_frame, f"{stem}_last_seen.jpg")
        self.save_frame(current_frame, f"{stem}_current.jpg")

    def z_gate_rejected(self, raw_z, stable_z, last_z, dt):
        self._gate_rejections += 1
        self.event(
            f"[Z] Gate dropped frame: pnp={raw_z:.2f} m, size={stable_z:.2f} m, "
            f"last accepted={last_z:.2f} m, dt={dt * 1000:.0f} ms",
            console=False,
        )

    def save_frame(self, frame, name):
        """Write a JPEG snapshot to <run_dir>/frames/ in the background (never stalls the loop)."""
        if frame is None:
            return
        with self._lock:
            if self._frames_saved >= MAX_SAVED_FRAMES:
                return
            self._frames_saved += 1
        path = os.path.join(self.frames_dir, name)

        def _write():
            try:
                os.makedirs(self.frames_dir, exist_ok=True)
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    raise RuntimeError("imencode failed")
                with open(path, "wb") as f:
                    f.write(encoded.tobytes())
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                self.event(f"[FRAME] Failed to save {name}: {e}", console=False)

        threading.Thread(target=_write, daemon=True).start()
        self.event(f"[FRAME] Saved frames/{name}", console=False)

    def outlier_rejected(self, z):
        self._outliers += 1
        self._outlier_z_min = z if self._outlier_z_min is None else min(self._outlier_z_min, z)
        self._outlier_z_max = z if self._outlier_z_max is None else max(self._outlier_z_max, z)

    def pnp_failure(self):
        self._pnp_failures += 1

    def _sync_to_disk(self):
        with self._lock:
            try:
                self._text.flush()
                os.fsync(self._text.fileno())
                self._csv_file.flush()
                os.fsync(self._csv_file.fileno())
            except (OSError, ValueError):
                pass

    def _write_checkpoint(self):
        tracked = self._tracked_time
        if self._open_track is not None:
            tracked += time.monotonic() - self._open_track[1]
        header = [
            "LATEST SUMMARY SNAPSHOT (refreshed every ~10 s while running).",
            "If the battery was pulled, the .log file ends without a final",
            "summary block - this file is the last state that reached the disk.",
        ]
        body = "\n".join(header + self._summary_lines(tracked)) + "\n"
        tmp_path = self.summary_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.summary_path)
        except OSError:
            pass

    def summary(self):
        if self._open_track is not None:
            tag_id, start_mono, last_z = self._open_track
            self.track_lost(tag_id, last_z, time.monotonic() - start_mono)

        block = "\n".join(self._summary_lines(self._tracked_time))
        with self._lock:
            self._text.write(block + "\n")
        print(block, flush=True)
        self._sync_to_disk()
        self._write_checkpoint()

    def _summary_lines(self, tracked_time):
        run = self._rel()
        lines = ["", "=" * 62, "PRECISION LANDING RUN SUMMARY", "=" * 62]
        lines.append(f"Start:                 {datetime.fromtimestamp(self._start_wall).isoformat(timespec='seconds')}")
        lines.append(f"Duration:              {run:.1f} s")

        fps = self._frames_total / run if run > 0 else 0.0
        lines.append(f"Frames processed:      {self._frames_total} ({fps:.1f} fps avg)")
        if self._frames_total:
            lines.append(f"Detect time avg:       {self._detect_time_total / self._frames_total * 1000:.1f} ms")
            lines.append(f"Frame proc avg / max:  {self._proc_time_total / self._frames_total * 1000:.1f} / {self._proc_time_peak * 1000:.1f} ms")

        lines.append("")
        lines.append(f"LANDING_TARGET sent:   {self._sends_total}")
        if run > 0:
            lines.append(f"Avg rate (whole run):  {self._sends_total / run:.1f} Hz")
        if tracked_time > 0:
            lines.append(f"Avg rate (tracking):   {self._sends_total / tracked_time:.1f} Hz")
        if self._send_gap["n"]:
            lines.append(f"Send interval:         mean {self._send_gap['sum'] / self._send_gap['n'] * 1000:.1f} ms, max {self._send_gap['max'] * 1000:.1f} ms")
        for tag in sorted(self._per_tag):
            s = self._per_tag[tag]
            lines.append(f"  ID {tag}: {s['n']} msgs, Z {s['z_min']:.2f} - {s['z_max']:.2f} m (mean {s['z_sum'] / s['n']:.2f})")

        lines.append("")
        pct = tracked_time / run * 100 if run > 0 else 0.0
        lines.append(f"Tracks:                {self._tracks} ({tracked_time:.1f} s tracked, {pct:.1f}% of run)")

        def _capped(items, fmt):
            shown = [fmt(item) for item in items[:30]]
            if len(items) > 30:
                shown.append(f"  ... and {len(items) - 30} more")
            return shown

        if self._acquire_events:
            lines.append("Acquisitions:")
            lines.extend(_capped(self._acquire_events, lambda e: f"  t={e[0]:7.1f}s  ID {e[1]}  Z={e[2]:.2f} m"))
        if self._loss_events:
            lines.append("Losses:")
            lines.extend(_capped(self._loss_events, lambda e: f"  t={e[0]:7.1f}s  ID {e[1]}  last Z={e[2]:.2f} m  after {e[3]:.1f} s"))
        if self._handoffs:
            lines.append("Handoffs:")
            lines.extend(_capped(self._handoffs, lambda e: f"  t={e[0]:7.1f}s  ID {e[1]} -> ID {e[2]}  Z={e[3]:.2f} m"))

        lines.append("")
        lines.append("Z estimation:")
        if self._z_source_counts:
            lines.append("  Source of sent Z:     " + ", ".join(
                f"{src} {n}" for src, n in sorted(self._z_source_counts.items())))
        if self._z_diff_n:
            lines.append(f"  |pnp - size|:         mean {self._z_diff_sum / self._z_diff_n:.2f} m, "
                         f"max {self._z_diff_max:.2f} m")
        lines.append(f"  PnP ambiguous frames: {self._pnp_ambiguous}")
        lines.append(f"  Gate dropped frames:  {self._gate_rejections}")
        lines.append(f"Frames saved:          {self._frames_saved} (frames/)")

        lines.append("")
        outlier_txt = f"Outliers rejected:     {self._outliers}"
        if self._outliers:
            outlier_txt += f" (Z {self._outlier_z_min:.2f} - {self._outlier_z_max:.2f} m)"
        lines.append(outlier_txt)
        lines.append(f"solvePnP failures:     {self._pnp_failures}")
        reconnects = max(0, self._camera_connects - 1)
        lines.append(f"Camera connects:       {self._camera_connects} (reconnects: {reconnects})")
        lines.append("=" * 62)
        return lines

    def close(self):
        with self._lock:
            self._text.close()
            self._csv_file.close()

# ----------------- 5. MAVLink 發送 -----------------
def connect_mavlink(logger):
    logger.event(f"[MAVLink] Connecting to Pixhawk on {MAVLINK_PORT}...")
    master = mavutil.mavlink_connection(
        MAVLINK_PORT, baud=MAVLINK_BAUD, source_system=255, source_component=190
    )
    msg = master.wait_heartbeat(timeout=5)
    if msg is None:
        logger.event("[MAVLink] 【錯誤】無法取得 Pixhawk 心跳包！請檢查 UART 接線。")
        return None
    else:
        logger.event("[MAVLink] Heartbeat received successfully!")
        return master

def send_landing_target(master, x, y, z):
    distance = math.sqrt(x**2 + y**2 + z**2)
    angle_x = math.atan2(y, z)
    angle_y = math.atan2(x, z)
    time_boot_us = int(time.time() * 1e6)
    
    master.mav.landing_target_send(
        time_boot_us, 0,
        mavutil.mavlink.MAV_FRAME_BODY_FRD,
        angle_x, angle_y, distance,
        0.0, 0.0,
        x, y, z,
        [1.0, 0.0, 0.0, 0.0],
        2, 1
    )
    return distance, angle_x, angle_y

# ----------------- 5.5 深度 (Z) 估測 -----------------
def estimate_depth(img_points, obj_pts, tag_size, focal_length):
    """Two independent depth estimates for one detected tag.

    Returns (raw_z, stable_z, reproj_px, ambiguity):
      raw_z      depth from the solvePnP (IPPE_SQUARE) solution with the lowest
                 reprojection error, or None if solvePnP failed
      stable_z   pinhole depth from the mean of both image diagonals (using both
                 diagonals cancels most of the foreshortening from tilt)
      reproj_px  reprojection error of the chosen PnP solution
      ambiguity  2nd-best / best reprojection error; close to 1.0 means the two
                 IPPE solutions (plane flip) are indistinguishable
    """
    d1 = np.linalg.norm(img_points[0] - img_points[2])
    d2 = np.linalg.norm(img_points[1] - img_points[3])
    diag_px = (d1 + d2) / 2.0
    stable_z = (tag_size * math.sqrt(2) * focal_length) / diag_px if diag_px > 0 else float("nan")

    count, rvecs, tvecs, errors = cv2.solvePnPGeneric(
        obj_pts, img_points, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not count or len(tvecs) == 0:
        return None, stable_z, None, None

    errors = np.asarray(errors, dtype=float).flatten()
    best = int(np.argmin(errors))
    raw_z = float(tvecs[best][2][0])
    ambiguity = None
    if len(errors) > 1:
        second = float(np.sort(errors)[1])
        ambiguity = second / errors[best] if errors[best] > 1e-9 else 1.0
    return raw_z, stable_z, float(errors[best]), ambiguity


def choose_depth(raw_z, stable_z, reproj_px, ambiguity, last_z, dt):
    """Pick the depth to send. Returns (z, source); z is None when the frame should be dropped.

    1. PnP is trusted only when its best solution is unambiguous and fits well;
       otherwise the size-based estimate is preferred.
    2. Temporal gate: if the last accepted Z is recent (dt <= Z_GATE_MAX_DT), a
       candidate must be reachable at Z_MAX_RATE_MPS plus a relative noise
       allowance. If the preferred estimator fails the gate the other one is
       tried ("*_gated"); if both fail the frame is dropped ("gate").
    """
    pnp_trusted = (
        reproj_px is not None
        and reproj_px <= PNP_MAX_REPROJ_PX
        and (ambiguity is None or ambiguity >= PNP_AMBIGUITY_RATIO)
    )
    if pnp_trusted:
        candidates = [(raw_z, "pnp"), (stable_z, "size")]
    else:
        candidates = [(stable_z, "size"), (raw_z, "pnp")]

    if last_z is None or dt is None or dt > Z_GATE_MAX_DT:
        return candidates[0]

    allowed = Z_MAX_RATE_MPS * dt + Z_GATE_REL * last_z
    for i, (z, source) in enumerate(candidates):
        if z is not None and not math.isnan(z) and abs(z - last_z) <= allowed:
            return z, (source if i == 0 else source + "_gated")
    return None, "gate"

# ----------------- 6. 主程式迴圈 -----------------
def main():
    logger = FlightLogger()
    logger.event(
        f"[SYSTEM] OpenCV {cv2.__version__} | Pi Camera V2.1 (Calibrated) | "
        f"MAVLink: {MAVLINK_PORT} @ {MAVLINK_BAUD}",
        console=False,
    )

    # 1. 嚴格連線檢查：未連上 Pixhawk 則直接終止程式
    master = connect_mavlink(logger)
    if master is None:
        print("[SYSTEM] 【終止】Pixhawk 未連線，自動停止精準降落程式。")
        logger.close()
        return

    reader = None
    current_tag = None
    track_start_mono = 0.0
    track_last_z = 0.0
    last_seen_frame = None      # most recent frame with a valid detection (saved on loss)
    last_z_accepted = None      # temporal gate state
    last_z_mono = None

    fx = CAMERA_MATRIX[0, 0]
    fy = CAMERA_MATRIX[1, 1]
    cx = CAMERA_MATRIX[0, 2]
    cy = CAMERA_MATRIX[1, 2]
    focal_length = (fx + fy) / 2.0

    try:
        reader = NonBlockingPiCameraReader(FRAME_WIDTH, FRAME_HEIGHT, logger)

        dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

        try:
            parameters = aruco.DetectorParameters()
            parameters.maxErroneousBitsInBorderRate = 0.05
            parameters.polygonalApproxAccuracyRate = 0.02
            parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            detector = aruco.ArucoDetector(dictionary, parameters)
            is_new_api = True
        except AttributeError:
            parameters = aruco.DetectorParameters_create()
            parameters.maxErroneousBitsInBorderRate = 0.05
            parameters.polygonalApproxAccuracyRate = 0.02
            parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            is_new_api = False

        logger.event("[SYSTEM] ArduCopter Precision Landing Daemon Running (Picamera2 Native)...")

        MAX_VALID_Z = 35.0
        logger.event(f"[SYSTEM] Valid Z range: 0.1 - {MAX_VALID_Z} m", console=False)
        logger.event(
            f"[SYSTEM] Z estimator: PnP trusted if ambiguity >= {PNP_AMBIGUITY_RATIO} and "
            f"reproj <= {PNP_MAX_REPROJ_PX} px, else size-based (both diagonals); "
            f"gate {Z_MAX_RATE_MPS} m/s + {Z_GATE_REL:.0%} within {Z_GATE_MAX_DT} s",
            console=False,
        )

        send_count = 0
        frame_count = 0
        rate_window_start = time.time()

        detect_time_sum = 0.0
        proc_time_sum = 0.0
        proc_time_max = 0.0

        while True:
            now = time.time()
            elapsed = now - rate_window_start
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                rate_hz = send_count / elapsed
                if frame_count > 0:
                    avg_detect_ms = detect_time_sum / frame_count * 1000.0
                    avg_proc_ms = proc_time_sum / frame_count * 1000.0
                    max_proc_ms = proc_time_max * 1000.0
                    rate_msg = (f"[RATE] Camera FPS: {fps:.1f} | LANDING_TARGET send rate: {rate_hz:.1f} Hz | "
                                f"Proc avg: {avg_proc_ms:.1f}ms (detect {avg_detect_ms:.1f}ms) max: {max_proc_ms:.1f}ms")
                else:
                    rate_msg = f"[RATE] Camera FPS: {fps:.1f} | LANDING_TARGET send rate: {rate_hz:.1f} Hz"
                logger.rate_report(rate_msg, elapsed, frame_count, send_count,
                                    detect_time_sum, proc_time_sum, proc_time_max)
                frame_count = 0
                send_count = 0
                detect_time_sum = 0.0
                proc_time_sum = 0.0
                proc_time_max = 0.0
                rate_window_start = now

            ret, frame = reader.read_new(timeout=0.5)
            if not ret or frame is None:
                continue
            frame_count += 1

            proc_start = time.monotonic()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if is_new_api:
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)

            detect_s = time.monotonic() - proc_start
            detect_time_sum += detect_s

            target_to_use = None

            if ids is not None:
                ids_flat = ids.flatten()
                if 0 in ids_flat:
                    target_to_use = 0
                elif 1 in ids_flat:
                    target_to_use = 1

            if target_to_use is not None:
                index = np.where(ids.flatten() == target_to_use)[0][0]
                img_points = corners[index][0]
                obj_pts = TARGETS[target_to_use]["obj_points"]

                raw_z, stable_z, reproj_px, ambiguity = estimate_depth(
                    img_points, obj_pts, TARGETS[target_to_use]["size"], focal_length
                )

                if raw_z is None:
                    logger.pnp_failure()
                else:
                    now_mono = time.monotonic()
                    dt_gate = (now_mono - last_z_mono) if last_z_mono is not None else None
                    z_m, z_source = choose_depth(raw_z, stable_z, reproj_px, ambiguity,
                                                 last_z_accepted, dt_gate)

                    if z_m is None:
                        # Both estimators jumped implausibly since the last accepted Z.
                        logger.z_gate_rejected(raw_z, stable_z, last_z_accepted, dt_gate)
                        proc_time = time.monotonic() - proc_start
                        proc_time_sum += proc_time
                        proc_time_max = max(proc_time_max, proc_time)
                        continue

                    if z_m > MAX_VALID_Z or z_m < 0.1:
                        logger.outlier_rejected(z_m)
                        proc_time = time.monotonic() - proc_start
                        proc_time_sum += proc_time
                        proc_time_max = max(proc_time_max, proc_time)
                        continue

                    u = np.mean(img_points[:, 0])
                    v = np.mean(img_points[:, 1])

                    angle_x = math.atan((u - cx) / fx)
                    angle_y = math.atan((cy - v) / fy)

                    y_m = float(z_m * math.tan(angle_x))
                    x_m = float(z_m * math.tan(angle_y))

                    if current_tag is None:
                        logger.track_acquired(target_to_use, z_m, frame=frame)
                        track_start_mono = time.monotonic()
                    elif current_tag != target_to_use:
                        logger.track_handoff(current_tag, target_to_use, z_m)
                    current_tag = target_to_use
                    track_last_z = z_m
                    last_seen_frame = frame
                    last_z_accepted = z_m
                    last_z_mono = now_mono

                    distance, angle_x_rad, angle_y_rad = send_landing_target(master, x=x_m, y=y_m, z=z_m)
                    send_count += 1
                    logger.target_sent(target_to_use, x_m, y_m, z_m,
                                       distance, angle_x_rad, angle_y_rad,
                                       detect_s, time.monotonic() - proc_start,
                                       raw_z=raw_z, stable_z=stable_z, z_source=z_source,
                                       reproj_px=reproj_px, ambiguity=ambiguity)

                    tag_type = "ID:0(Small)" if target_to_use == 0 else "ID:1(Large)"
                    print(f"[{tag_type}] Front: {x_m:.2f}m | Right: {y_m:.2f}m | "
                          f"Down(Z): {z_m:.2f}m [{z_source}]")
            else:
                if current_tag is not None:
                    logger.track_lost(current_tag, track_last_z, time.monotonic() - track_start_mono,
                                      last_seen_frame=last_seen_frame, current_frame=frame)
                    current_tag = None
                    last_seen_frame = None
                    last_z_accepted = None
                    last_z_mono = None
                    print("==========================================")
                    print(" [WARNING] Target Lost! Stop sending MAVLink.")
                    print("==========================================")

            proc_time = time.monotonic() - proc_start
            proc_time_sum += proc_time
            proc_time_max = max(proc_time_max, proc_time)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Terminating Precision Landing Daemon...")
        logger.event("[SYSTEM] Stopped by user (Ctrl+C)", console=False)
    except Exception:
        logger.event("[ERROR] Daemon crashed:\n" + traceback.format_exc(), console=False)
        raise
    finally:
        if reader is not None:
            reader.stop()
        logger.summary()
        logger.close()

if __name__ == '__main__':
    main()
