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

# ----------------- 2. 多標籤 (Nested Tags) 參數定義 -----------------
# Each tag: physical edge length + where its centre sits relative to the LANDING
# POINT (pad centre). Tags may be placed off-centre; the drone is always steered
# to the landing point, not to the tag.
#
# Pad frame used for "offset": look at the pad from above with the tags printed
# upright, all in the SAME orientation.
#   x = metres to the RIGHT of the landing point
#   y = metres toward the TOP of the tags (up on the print)
# A tag whose centre is 30 cm right and 20 cm up of the landing point has
# offset (0.30, 0.20). Measure "size" as the outer edge of the black square on
# the actual print (printers scale).
TARGETS = {
    0: {"size": 0.10, "offset": (0.0, 0.0)},     # 低空小標籤 at the landing point
    1: {"size": 0.80, "offset": (0.0, 0.0)},     # 高空大標籤
    2: {"size": 0.10, "offset": (0.30, 0.20)}, # example: off-centre small tag
    # 3: {"size": 0.10, "offset": (-0.30, -0.20)},
}

# Which tag to use when several are visible: "smallest" (previous behaviour) or
# "largest" (best range/bearing precision; smaller tags take over automatically
# once the larger one leaves the frame).
TAG_PRIORITY = "smallest"

for tid, info in TARGETS.items():
    h_size = info["size"] / 2.0
    info.setdefault("offset", (0.0, 0.0))
    # Tag-frame corners: top-left, top-right, bottom-right, bottom-left,
    # x right, y up (order required by SOLVEPNP_IPPE_SQUARE).
    info["plane_pts"] = np.array([
        [-h_size,  h_size],
        [ h_size,  h_size],
        [ h_size, -h_size],
        [-h_size, -h_size]
    ], dtype=np.float32)
    info["obj_points"] = np.hstack([info["plane_pts"], np.zeros((4, 1), dtype=np.float32)])

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

    def target_sent(self, tag_id, x, y, z, distance, angle_x, angle_y, detect_s, proc_s):
        now_mono = time.monotonic()
        self._sends_total += 1
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
        ])

    def track_acquired(self, tag_id, z):
        self._tracks += 1
        self._acquire_events.append((self._rel(), tag_id, z))
        self._open_track = [tag_id, time.monotonic(), z]
        self.event(f"[TRACK] Target acquired: ID {tag_id} at Z={z:.2f} m")
        self._sync_to_disk()

    def track_handoff(self, from_tag, to_tag, z):
        self._handoffs.append((self._rel(), from_tag, to_tag, z))
        if self._open_track is not None:
            self._open_track[0] = to_tag
            self._open_track[2] = z
        self.event(f"[TRACK] Tag handoff ID {from_tag} -> ID {to_tag} at Z={z:.2f} m")
        self._sync_to_disk()

    def track_lost(self, tag_id, last_z, duration):
        self._tracked_time += duration
        self._track_durations.append(duration)
        self._loss_events.append((self._rel(), tag_id, last_z, duration))
        self._open_track = None
        self.event(f"[TRACK] Target lost: ID {tag_id}, last Z={last_z:.2f} m, tracked {duration:.1f} s", console=False)
        self._sync_to_disk()

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

# ----------------- 5.5 標籤 → 降落點 幾何 -----------------
def _homography_from_corners(plane_pts, img_pts):
    """3x3 homography mapping tag-plane coordinates (m) to pixels from 4 correspondences.

    Solves the 8 linear DLT equations with h33 = 1:
        u = (h11 x + h12 y + h13) / (h31 x + h32 y + 1)
        v = (h21 x + h22 y + h23) / (h31 x + h32 y + 1)
    """
    A = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    for i, ((x, y), (u, v)) in enumerate(zip(plane_pts, img_pts)):
        A[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y]
        b[2 * i] = u
        A[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y]
        b[2 * i + 1] = v
    h = np.linalg.solve(A, b)
    return np.array([[h[0], h[1], h[2]],
                     [h[3], h[4], h[5]],
                     [h[6], h[7], 1.0]])


def _apply_homography(H, point):
    x, y = point
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return ((H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w,
            (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w)


def pad_center_pixel(img_points, target):
    """Pixel where the LANDING POINT projects, computed from one tag's corners.

    The four corners define the exact plane-to-image mapping, so any point of
    the pad plane can be located in the image without recovering the (ambiguous)
    3D pose. The landing point lies at -offset from the tag centre in the tag
    frame; yaw and perspective are handled by the homography itself.
    """
    ox, oy = target["offset"]
    try:
        H = _homography_from_corners(target["plane_pts"], img_points)
        return _apply_homography(H, (-ox, -oy))
    except np.linalg.LinAlgError:
        # Degenerate corners: fall back to the tag centroid (offset ignored).
        return float(np.mean(img_points[:, 0])), float(np.mean(img_points[:, 1]))


def select_target(ids):
    """Return (tag id to use or None, list of all visible configured tag ids)."""
    if ids is None:
        return None, []
    visible = set(int(i) for i in ids.flatten())
    candidates = [tid for tid in TARGETS if tid in visible]
    if not candidates:
        return None, []
    size_of = lambda tid: TARGETS[tid]["size"]
    if TAG_PRIORITY == "largest":
        return max(candidates, key=size_of), candidates
    return min(candidates, key=size_of), candidates

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
    last_pad_check_mono = 0.0   # rate limit for the multi-tag consistency report

    fx = CAMERA_MATRIX[0, 0]
    fy = CAMERA_MATRIX[1, 1]
    cx = CAMERA_MATRIX[0, 2]
    cy = CAMERA_MATRIX[1, 2]
    focal_length = (fx + fy) / 2.0

    logger.event(
        f"[SYSTEM] Tags (size @ offset from landing point, priority={TAG_PRIORITY}): " + ", ".join(
            f"ID {tid} {info['size'] * 100:.0f}cm @({info['offset'][0]:+.2f},{info['offset'][1]:+.2f})m"
            for tid, info in sorted(TARGETS.items())
        ),
        console=False,
    )

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

            target_to_use, visible_targets = select_target(ids)

            if target_to_use is not None:
                index = np.where(ids.flatten() == target_to_use)[0][0]
                img_points = corners[index][0]
                obj_pts = TARGETS[target_to_use]["obj_points"]

                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_points, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    raw_z = float(tvec[2][0])

                    diag_px = np.linalg.norm(img_points[0] - img_points[2])
                    tag_real_size = TARGETS[target_to_use]["size"]
                    stable_z = (tag_real_size * math.sqrt(2) * focal_length) / diag_px

                    z_m = stable_z if abs(raw_z - stable_z) > 0.4 else raw_z

                    if z_m > MAX_VALID_Z or z_m < 0.1:
                        logger.outlier_rejected(z_m)
                        proc_time = time.monotonic() - proc_start
                        proc_time_sum += proc_time
                        proc_time_max = max(proc_time_max, proc_time)
                        continue

                    # Locate the LANDING POINT (not the tag centre) in the image via
                    # this tag's plane homography: handles off-centre tags, yaw and
                    # perspective exactly.
                    u, v = pad_center_pixel(img_points, TARGETS[target_to_use])

                    angle_x = math.atan((u - cx) / fx)
                    angle_y = math.atan((cy - v) / fy)

                    y_m = float(z_m * math.tan(angle_x))
                    x_m = float(z_m * math.tan(angle_y))

                    # Bench check for offsets: every visible tag must point at the same
                    # landing point. Report the spread at most once per second.
                    now_check = time.monotonic()
                    if len(visible_targets) > 1 and now_check - last_pad_check_mono >= 1.0:
                        last_pad_check_mono = now_check
                        ids_flat = ids.flatten()
                        estimates = np.array([
                            pad_center_pixel(corners[np.where(ids_flat == tid)[0][0]][0], TARGETS[tid])
                            for tid in visible_targets
                        ])
                        spread_px = float(np.max(np.linalg.norm(
                            estimates[:, None, :] - estimates[None, :, :], axis=2)))
                        logger.event(
                            f"[PAD] Tags {sorted(visible_targets)} visible: landing-point estimates "
                            f"spread {spread_px:.1f} px (~{spread_px * z_m / focal_length * 100:.1f} cm "
                            f"at Z={z_m:.1f} m)"
                        )

                    if current_tag is None:
                        logger.track_acquired(target_to_use, z_m)
                        track_start_mono = time.monotonic()
                    elif current_tag != target_to_use:
                        logger.track_handoff(current_tag, target_to_use, z_m)
                    current_tag = target_to_use
                    track_last_z = z_m

                    distance, angle_x_rad, angle_y_rad = send_landing_target(master, x=x_m, y=y_m, z=z_m)
                    send_count += 1
                    logger.target_sent(target_to_use, x_m, y_m, z_m,
                                       distance, angle_x_rad, angle_y_rad,
                                       detect_s, time.monotonic() - proc_start)

                    tag_type = f"ID:{target_to_use}({TARGETS[target_to_use]['size'] * 100:.0f}cm)"
                    print(f"[{tag_type}] Front: {x_m:.2f}m | Right: {y_m:.2f}m | Down(Z): {z_m:.2f}m")
                else:
                    logger.pnp_failure()
            else:
                if current_tag is not None:
                    logger.track_lost(current_tag, track_last_z, time.monotonic() - track_start_mono)
                    current_tag = None
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
