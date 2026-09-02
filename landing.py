import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"

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

# ----------------- 2. 系統與相機參數設定 -----------------
RTSP_URL = "rtsp://192.168.144.25:8554/main.264"
MAVLINK_PORT = '/dev/ttyS0'
MAVLINK_BAUD = 57600

USE_FISHEYE = False

# 相機內參 (從 JSON 匯入)
CAMERA_MATRIX = np.array([
    [884.43516979,   0.00000000, 636.61557038],
    [  0.00000000, 885.04866594, 390.53879835],
    [  0.00000000,   0.00000000,   1.00000000]
], dtype=np.float32)

DIST_COEFFS = np.array([
    [-0.16986182, 0.24550093, 0.00031341, 0.00140645, -0.04924192]
], dtype=np.float32)

# ----------------- 3. 雙標籤 參數定義 -----------------
# ID 0: 低空小標籤 (16cm)
# ID 1: 30 米高空大標籤 (例如 1.5m，戶外強烈建議做大)
TARGETS = {
    0: {"size": 0.16, "obj_points": None},
    1: {"size": 1.50, "obj_points": None}
}

for tid, info in TARGETS.items():
    h_size = info["size"] / 2.0
    info["obj_points"] = np.array([
        [-h_size,  h_size, 0],
        [ h_size,  h_size, 0],
        [ h_size, -h_size, 0],
        [-h_size, -h_size, 0]
    ], dtype=np.float32)

# ----------------- 4. RTSP 讀取器 -----------------
class NonBlockingRTSPReader:
    def __init__(self, url, logger=None):
        self.url = url
        self.logger = logger
        self.cap = None
        self.latest_frame = None
        self.last_update_time = 0
        self.stopped = False
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _connect(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.logger:
            self.logger.camera_connect(fps)
        else:
            print(f"[SYSTEM] Camera FPS: {fps}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _update_loop(self):
        self._connect()

        # 實際接收/解碼幀率統計（與主迴圈處理速度無關）
        rx_count = 0
        rx_start = time.monotonic()

        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(0.5)
                self._connect()
                continue
            
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
                    self.last_update_time = time.time()
                # 喚醒等待新幀的主迴圈 (取代輪詢 + sleep)
                self.new_frame_event.set()

                rx_count += 1
                now = time.monotonic()
                elapsed = now - rx_start
                if elapsed >= 5.0:
                    rx_msg = f"[RX] Receive/decode FPS: {rx_count / elapsed:.1f}"
                    if self.logger:
                        self.logger.event(rx_msg)
                    else:
                        print(rx_msg, flush=True)
                    rx_count = 0
                    rx_start = now
            else:
                if time.time() - self.last_update_time > 1.0:
                    time.sleep(0.2)
                    self._connect()
                time.sleep(0.005)

    def read_new(self, timeout=0.5):
        """Block until a frame we have not consumed yet is available.

        Returns (True, frame) as soon as the reader thread decodes one, or
        (False, None) after `timeout` seconds without a new frame. The frame
        is handed over by ownership (no copy): the reader always stores newly
        allocated buffers, so the consumer can use it freely.
        """
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
        if self.cap and self.cap.isOpened():
            self.cap.release()

# ----------------- 4.5 本地飛行日誌 -----------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


class FlightLogger:
    """Per-run flight log on local disk. Never overwrites a previous run.

    Creates two files under logs/:
        landing_YYYYmmdd_HHMMSS[_N].log          events, rate reports, summary
        landing_YYYYmmdd_HHMMSS[_N]_targets.csv  one row per LANDING_TARGET sent
    """

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(LOG_DIR, f"landing_{stamp}")
        suffix = 0
        while True:
            candidate = base if suffix == 0 else f"{base}_{suffix}"
            if not os.path.exists(candidate + ".log"):
                break
            suffix += 1
        self.text_path = candidate + ".log"
        self.csv_path = candidate + "_targets.csv"
        self.summary_path = candidate + "_summary.txt"

        self._lock = threading.Lock()
        # line-buffered so data survives a power cut mid-flight
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

        # ---- aggregate statistics for the end-of-run summary ----
        self._sends_total = 0
        self._per_tag = {}                    # tag id -> n / z_min / z_max / z_sum
        self._send_gap = {"n": 0, "sum": 0.0, "max": 0.0}
        self._last_send_mono = None

        self._tracks = 0
        self._tracked_time = 0.0
        self._track_durations = []
        self._acquire_events = []             # (t_rel, tag, z)
        self._loss_events = []                # (t_rel, tag, last z, duration)
        self._handoffs = []                   # (t_rel, from tag, to tag, z)

        self._frames_total = 0
        self._detect_time_total = 0.0
        self._proc_time_total = 0.0
        self._proc_time_peak = 0.0

        self._outliers = 0
        self._outlier_z_min = None
        self._outlier_z_max = None
        self._pnp_failures = 0
        self._camera_connects = 0

        self._open_track = None          # [tag_id, start_mono, last_z] while tracking
        self._last_checkpoint_mono = 0.0

        self.event(f"[LOG] Writing to {self.text_path}", console=False)

    def _rel(self):
        return time.monotonic() - self._start_mono

    def event(self, message, console=True):
        """Timestamped line in the log file; unchanged message on the console."""
        with self._lock:
            self._text.write(f"[{self._rel():9.2f}s] {message}\n")
        if console:
            print(message, flush=True)

    def camera_connect(self, fps):
        with self._lock:
            self._camera_connects += 1
        self.event(f"[SYSTEM] Camera FPS: {fps}")

    def rate_report(self, message, window_s, frames, sends,
                    detect_sum, proc_sum, proc_max):
        self._frames_total += frames
        self._detect_time_total += detect_sum
        self._proc_time_total += proc_sum
        self._proc_time_peak = max(self._proc_time_peak, proc_max)
        self.event(message)
        # The battery is usually just pulled: force data onto the SD card every
        # rate window (~2 s) and refresh the summary snapshot every ~10 s.
        self._sync_to_disk()
        if time.monotonic() - self._last_checkpoint_mono >= 10.0:
            self._last_checkpoint_mono = time.monotonic()
            self._write_checkpoint()

    def target_sent(self, tag_id, x, y, z, distance, angle_x, angle_y,
                    detect_s, proc_s):
        now_mono = time.monotonic()
        self._sends_total += 1
        stats = self._per_tag.setdefault(
            tag_id, {"n": 0, "z_min": z, "z_max": z, "z_sum": 0.0})
        stats["n"] += 1
        stats["z_min"] = min(stats["z_min"], z)
        stats["z_max"] = max(stats["z_max"], z)
        stats["z_sum"] += z
        if self._open_track is not None:
            self._open_track[2] = z
        if self._last_send_mono is not None:
            gap = now_mono - self._last_send_mono
            if gap < 2.0:  # ignore pauses between separate tracks
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
        self.event(
            f"[TRACK] Target lost: ID {tag_id}, last Z={last_z:.2f} m, "
            f"tracked {duration:.1f} s",
            console=False,
        )
        self._sync_to_disk()

    def outlier_rejected(self, z):
        self._outliers += 1
        self._outlier_z_min = z if self._outlier_z_min is None else min(self._outlier_z_min, z)
        self._outlier_z_max = z if self._outlier_z_max is None else max(self._outlier_z_max, z)

    def pnp_failure(self):
        self._pnp_failures += 1

    def _sync_to_disk(self):
        """fsync both log files so data survives an abrupt battery disconnect."""
        with self._lock:
            try:
                self._text.flush()
                os.fsync(self._text.fileno())
                self._csv_file.flush()
                os.fsync(self._csv_file.fileno())
            except (OSError, ValueError):
                pass  # files already closed

    def _write_checkpoint(self):
        """Atomically refresh the on-disk summary snapshot (survives power cut)."""
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
        # A track still active at shutdown counts toward tracked time.
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
        lines.append(f"Start:                 "
                     f"{datetime.fromtimestamp(self._start_wall).isoformat(timespec='seconds')}")
        lines.append(f"Duration:              {run:.1f} s")

        fps = self._frames_total / run if run > 0 else 0.0
        lines.append(f"Frames processed:      {self._frames_total} ({fps:.1f} fps avg)")
        if self._frames_total:
            lines.append(f"Detect time avg:       "
                         f"{self._detect_time_total / self._frames_total * 1000:.1f} ms")
            lines.append(f"Frame proc avg / max:  "
                         f"{self._proc_time_total / self._frames_total * 1000:.1f} / "
                         f"{self._proc_time_peak * 1000:.1f} ms")

        lines.append("")
        lines.append(f"LANDING_TARGET sent:   {self._sends_total}")
        if run > 0:
            lines.append(f"Avg rate (whole run):  {self._sends_total / run:.1f} Hz")
        if tracked_time > 0:
            lines.append(f"Avg rate (tracking):   "
                         f"{self._sends_total / tracked_time:.1f} Hz")
        if self._send_gap["n"]:
            lines.append(f"Send interval:         "
                         f"mean {self._send_gap['sum'] / self._send_gap['n'] * 1000:.1f} ms, "
                         f"max {self._send_gap['max'] * 1000:.1f} ms")
        for tag in sorted(self._per_tag):
            s = self._per_tag[tag]
            lines.append(f"  ID {tag}: {s['n']} msgs, "
                         f"Z {s['z_min']:.2f} - {s['z_max']:.2f} m "
                         f"(mean {s['z_sum'] / s['n']:.2f})")

        lines.append("")
        pct = tracked_time / run * 100 if run > 0 else 0.0
        lines.append(f"Tracks:                {self._tracks} "
                     f"({tracked_time:.1f} s tracked, {pct:.1f}% of run)")
        if self._track_durations:
            mean_track = sum(self._track_durations) / len(self._track_durations)
            lines.append(f"Track duration:        mean {mean_track:.1f} s, "
                         f"longest {max(self._track_durations):.1f} s")

        def _capped(items, fmt):
            shown = [fmt(item) for item in items[:30]]
            if len(items) > 30:
                shown.append(f"  ... and {len(items) - 30} more")
            return shown

        if self._acquire_events:
            lines.append("Acquisitions:")
            lines.extend(_capped(
                self._acquire_events,
                lambda e: f"  t={e[0]:7.1f}s  ID {e[1]}  Z={e[2]:.2f} m"))
        if self._loss_events:
            lines.append("Losses:")
            lines.extend(_capped(
                self._loss_events,
                lambda e: f"  t={e[0]:7.1f}s  ID {e[1]}  last Z={e[2]:.2f} m  "
                          f"after {e[3]:.1f} s"))
        if self._handoffs:
            lines.append("Handoffs:")
            lines.extend(_capped(
                self._handoffs,
                lambda e: f"  t={e[0]:7.1f}s  ID {e[1]} -> ID {e[2]}  Z={e[3]:.2f} m"))

        lines.append("")
        outlier_txt = f"Outliers rejected:     {self._outliers}"
        if self._outliers:
            outlier_txt += (f" (Z {self._outlier_z_min:.2f} - "
                            f"{self._outlier_z_max:.2f} m)")
        lines.append(outlier_txt)
        lines.append(f"solvePnP failures:     {self._pnp_failures}")
        reconnects = max(0, self._camera_connects - 1)
        lines.append(f"Camera connects:       {self._camera_connects} "
                     f"(reconnects: {reconnects})")
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
        logger.event("[MAVLink] 【警告】無法取得 Pixhawk 心跳包！")
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

# ----------------- 6. 主程式迴圈 -----------------
def main():
    logger = FlightLogger()
    logger.event(
        f"[SYSTEM] OpenCV {cv2.__version__} | RTSP: {RTSP_URL} | "
        f"MAVLink: {MAVLINK_PORT} @ {MAVLINK_BAUD}",
        console=False,
    )
    logger.event(
        "[SYSTEM] Tags: " + ", ".join(
            f"ID {tid} = {info['size']:.2f} m" for tid, info in sorted(TARGETS.items())
        ),
        console=False,
    )

    reader = None
    # 追蹤狀態：目前鎖定的標籤 ID (None = 未追蹤)
    current_tag = None
    track_start_mono = 0.0
    track_last_z = 0.0

    try:
        master = connect_mavlink(logger)
        reader = NonBlockingRTSPReader(RTSP_URL, logger)

        # 【防鬼影策略 1】：改用抗噪能力極強的 AprilTag (36h11) 字典
        # 備註：若實體打印紙為 5x5_1000，請改用 aruco.DICT_5X5_1000
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

        try:
            parameters = aruco.DetectorParameters()
            # 【防鬼影策略 2】：嚴格限制邊框錯誤率與輪廓近似度
            parameters.maxErroneousBitsInBorderRate = 0.05 # 極度嚴格，杜絕誤判
            parameters.polygonalApproxAccuracyRate = 0.02

            detector = aruco.ArucoDetector(dictionary, parameters)
            is_new_api = True
        except AttributeError:
            parameters = aruco.DetectorParameters_create()
            parameters.maxErroneousBitsInBorderRate = 0.05
            parameters.polygonalApproxAccuracyRate = 0.02
            is_new_api = False

        logger.event("[SYSTEM] ArduCopter Precision Landing Daemon Running (30m Altitude Ready)...")

        MAX_VALID_Z = 35.0  # 支援最高 35 米的高空識別
        logger.event(f"[SYSTEM] Valid Z range: 0.1 - {MAX_VALID_Z} m", console=False)

        # MAVLink 發送頻率與影像幀率統計
        send_count = 0
        frame_count = 0
        rate_window_start = time.time()

        # 每幀處理時間統計 (秒)
        detect_time_sum = 0.0
        proc_time_sum = 0.0
        proc_time_max = 0.0

        while True:
            # 每 2 秒回報一次影像幀率與 MAVLink 發送頻率
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

            # 阻塞等待下一張「新」影像幀 (事件驅動，無輪詢延遲)
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

                if USE_FISHEYE:
                    undist_img_pts = cv2.fisheye.undistortPoints(
                        img_points.reshape(-1, 1, 2), CAMERA_MATRIX, DIST_COEFFS
                    )
                    success, rvec, tvec = cv2.solvePnP(
                        obj_pts, undist_img_pts, np.eye(3, dtype=np.float32), None, flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                else:
                    success, rvec, tvec = cv2.solvePnP(
                        obj_pts, img_points, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )

                if success:
                    x_m = -float(tvec[1][0])
                    y_m = float(tvec[0][0])
                    z_m = float(tvec[2][0])

                    # 濾除 35 米以上的異常值
                    if z_m > MAX_VALID_Z or z_m < 0.1:
                        logger.outlier_rejected(z_m)
                        proc_time = time.monotonic() - proc_start
                        proc_time_sum += proc_time
                        proc_time_max = max(proc_time_max, proc_time)
                        continue

                    # 追蹤狀態事件：捕獲 / 標籤交接
                    if current_tag is None:
                        logger.track_acquired(target_to_use, z_m)
                        track_start_mono = time.monotonic()
                    elif current_tag != target_to_use:
                        logger.track_handoff(current_tag, target_to_use, z_m)
                    current_tag = target_to_use
                    track_last_z = z_m

                    # 發送 MAVLink landing_target
                    distance, angle_x, angle_y = send_landing_target(master, x=x_m, y=y_m, z=z_m)
                    send_count += 1
                    logger.target_sent(target_to_use, x_m, y_m, z_m,
                                       distance, angle_x, angle_y,
                                       detect_s, time.monotonic() - proc_start)

                    tag_type = "ID:0(Small)" if target_to_use == 0 else "ID:1(Large)"
                    print(f"[{tag_type}] Front: {x_m:.2f}m | Right: {y_m:.2f}m | Down(Z): {z_m:.2f}m")
                else:
                    logger.pnp_failure()
            else:
                if current_tag is not None:
                    logger.track_lost(current_tag, track_last_z,
                                      time.monotonic() - track_start_mono)
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
