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
    def __init__(self, url):
        self.url = url
        self.cap = None
        self.latest_frame = None
        self.last_update_time = 0
        self.stopped = False
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _connect(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        print(f"[SYSTEM] Camera FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _update_loop(self):
        self._connect()
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
            else:
                if time.time() - self.last_update_time > 1.0:
                    time.sleep(0.2)
                    self._connect()
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if self.latest_frame is None or time.time() - self.last_update_time > 0.5:
                return False, None
            return True, self.latest_frame.copy()

    def stop(self):
        self.stopped = True
        if self.cap and self.cap.isOpened():
            self.cap.release()

# ----------------- 5. MAVLink 發送 -----------------
def connect_mavlink():
    print(f"[MAVLink] Connecting to Pixhawk on {MAVLINK_PORT}...")
    master = mavutil.mavlink_connection(
        MAVLINK_PORT, baud=MAVLINK_BAUD, source_system=255, source_component=190
    )
    msg = master.wait_heartbeat(timeout=5)
    if msg is None:
        print("[MAVLink] 【警告】無法取得 Pixhawk 心跳包！")
    else:
        print("[MAVLink] Heartbeat received successfully!")
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

# ----------------- 6. 主程式迴圈 -----------------
def main():
    master = connect_mavlink()
    reader = NonBlockingRTSPReader(RTSP_URL)

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

    print(f"[SYSTEM] ArduCopter Precision Landing Daemon Running (30m Altitude Ready)...")

    target_was_found = False
    MAX_VALID_Z = 35.0  # 支援最高 35 米的高空識別

    # MAVLink 發送頻率與影像幀率統計
    send_count = 0
    frame_count = 0
    last_frame_timestamp = 0.0
    rate_window_start = time.time()

    try:
        while True:
            # 每 2 秒回報一次影像幀率與 MAVLink 發送頻率
            now = time.time()
            elapsed = now - rate_window_start
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                rate_hz = send_count / elapsed
                print(f"[RATE] Camera FPS: {fps:.1f} | LANDING_TARGET send rate: {rate_hz:.1f} Hz")
                frame_count = 0
                send_count = 0
                rate_window_start = now

            ret, frame = reader.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # 只在收到「新」影像幀時計數（read() 可能重複回傳同一幀）
            if reader.last_update_time != last_frame_timestamp:
                frame_count += 1
                last_frame_timestamp = reader.last_update_time

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if is_new_api:
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)

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
                        continue

                    
                    # Comment out u_raw and v_raw... redundant calculation from tvec
                    # 單點去畸變中心投射
                    # u_raw = np.mean(img_points[:, 0])
                    # v_raw = np.mean(img_points[:, 1])
                    # raw_pt = np.array([[[u_raw, v_raw]]], dtype=np.float32)
                    
                    # if USE_FISHEYE:
                    #     undist_pt = cv2.fisheye.undistortPoints(raw_pt, CAMERA_MATRIX, DIST_COEFFS)
                    # else:
                    #     undist_pt = cv2.undistortPoints(raw_pt, CAMERA_MATRIX, DIST_COEFFS)
                        
                    # norm_x = undist_pt[0][0][0]
                    # norm_y = undist_pt[0][0][1]

                    # y_m = float(norm_x * z_m)
                    # x_m = float(-norm_y * z_m)

                    # 發送 MAVLink landing_target
                    send_landing_target(master, x=x_m, y=y_m, z=z_m)
                    send_count += 1
                    
                    tag_type = "ID:0(Small)" if target_to_use == 0 else "ID:1(Large)"
                    print(f"[{tag_type}] Front: {x_m:.2f}m | Right: {y_m:.2f}m | Down(Z): {z_m:.2f}m")
                    
                    target_was_found = True
            else:
                if target_was_found:
                    print("==========================================")
                    print(" [WARNING] Target Lost! Stop sending MAVLink.")
                    print("==========================================")
                    target_was_found = False

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Terminating Precision Landing Daemon...")
    finally:
        reader.stop()

if __name__ == '__main__':
    main()
