# Allswell Precision Landing System

**Vision-Based Precision Landing Daemon for ArduPilot / ArduCopter**

## Overview

The Allswell Precision Landing System is an onboard computer-vision daemon that enables centimeter-level autonomous landings for ArduCopter-based aircraft. It ingests a live RTSP video feed from a downward-facing camera, detects fiducial landing markers (AprilTag 36h11), estimates the full 3D position of the landing target relative to the aircraft, and streams `LANDING_TARGET` MAVLink messages to the flight controller in real time.

The system implements a **dual-marker handoff strategy** for reliable acquisition from high altitude down to touchdown:

| Marker ID | Physical Size | Operational Role                                                                    |
| --------- | ------------- | ----------------------------------------------------------------------------------- |
| `ID 1`    | 1.50 m        | High-altitude acquisition (up to ~30–35 m AGL)                                      |
| `ID 0`    | 0.16 m        | Terminal guidance at low altitude, when the large marker overflows the camera frame |

When both markers are visible, the small marker (`ID 0`) is prioritized, ensuring a seamless transition as the aircraft descends.

## Key Features

- **Full 6-DoF pose estimation** — uses `solvePnP` with the `IPPE_SQUARE` solver against calibrated camera intrinsics for accurate X/Y/Z target position, not just angular offsets.
- **Dual-tag altitude handoff** — large tag for long-range acquisition, small tag for terminal precision.
- **Robust detection tuning** — AprilTag 36h11 dictionary with strict border-error (`maxErroneousBitsInBorderRate = 0.05`) and polygonal-approximation (`polygonalApproxAccuracyRate = 0.02`) thresholds to suppress false positives ("ghost tags") in cluttered outdoor scenes.
- **Non-blocking RTSP capture** — a dedicated reader thread maintains only the latest frame (no buffering lag) and automatically reconnects on stream dropout, using TCP transport with `nobuffer` FFmpeg flags for minimal latency.
- **Outlier rejection** — pose solutions outside the valid range (0.1 m – 35 m) are discarded before transmission.
- **Fisheye lens support** — optional fisheye undistortion path (`USE_FISHEYE`) for wide-angle camera modules.
- **Telemetry diagnostics** — per-detection position logging, target-lost warnings, and a rolling `LANDING_TARGET` transmit-rate report (Hz) every 2 seconds.
- **Legacy OpenCV compatibility** — automatically falls back to the pre-4.7 ArUco API when the modern `ArucoDetector` class is unavailable.

## System Architecture

```
┌──────────────────┐   RTSP (TCP)   ┌─────────────────────────────┐
│  Gimbal Camera   │ ─────────────► │  NonBlockingRTSPReader      │
│ 192.168.144.25   │                │  (background thread,        │
└──────────────────┘                │   latest-frame-only)        │
                                    └──────────────┬──────────────┘
                                                   │ frame
                                                   ▼
                                    ┌─────────────────────────────┐
                                    │  AprilTag 36h11 Detection   │
                                    │  + solvePnP (IPPE_SQUARE)   │
                                    │  → target pose (x, y, z)    │
                                    └──────────────┬──────────────┘
                                                   │ pose
                                                   ▼
┌──────────────────┐  /dev/ttyS0    ┌─────────────────────────────┐
│  Flight Ctrl     │ ◄───────────── │  MAVLink LANDING_TARGET     │
│  (Pixhawk /      │  57600 baud    │  MAV_FRAME_BODY_FRD         │
│   ArduCopter)    │                │  position_valid = 1         │
└──────────────────┘                └─────────────────────────────┘
```

### Coordinate Convention

Camera-frame pose (`tvec`) is remapped to the MAVLink `MAV_FRAME_BODY_FRD` body frame before transmission:

| Body-Frame Axis | Source     | Meaning                            |
| --------------- | ---------- | ---------------------------------- |
| `x`             | `-tvec[1]` | Forward (+)                        |
| `y`             | `+tvec[0]` | Right (+)                          |
| `z`             | `+tvec[2]` | Down (+), distance to target plane |

Angular offsets (`angle_x`, `angle_y`) and slant distance are derived from the positional solution and included in the same message.

## Requirements

### Hardware

- Companion computer with a UART link to the flight controller (default: `/dev/ttyS0` @ 57600 baud)
- ArduCopter-compatible flight controller (Pixhawk or equivalent)
- Downward-facing camera streaming RTSP H.264 (default: `rtsp://192.168.144.25:8554/main.264`)
- Printed landing markers, **AprilTag 36h11** family:
  - ID 0 at 16 cm
  - ID 1 at 150 cm (matte finish strongly recommended for outdoor use)

### Software

- Python 3.8+
- OpenCV (`opencv-contrib-python`) with FFmpeg support — both modern (≥ 4.7) and legacy ArUco APIs are supported
- NumPy
- pymavlink

```bash
pip install opencv-contrib-python numpy pymavlink
```

## Configuration

All operational parameters are defined at the top of `landing.py`:

| Parameter       | Default                               | Description                                                 |
| --------------- | ------------------------------------- | ----------------------------------------------------------- |
| `RTSP_URL`      | `rtsp://192.168.144.25:8554/main.264` | Camera stream endpoint                                      |
| `MAVLINK_PORT`  | `/dev/ttyS0`                          | Serial device connected to the flight controller            |
| `MAVLINK_BAUD`  | `57600`                               | Serial baud rate                                            |
| `USE_FISHEYE`   | `False`                               | Enable the fisheye undistortion model                       |
| `CAMERA_MATRIX` | _(calibrated)_                        | 3×3 camera intrinsic matrix                                 |
| `DIST_COEFFS`   | _(calibrated)_                        | Lens distortion coefficients                                |
| `TARGETS`       | ID 0 → 0.16 m, ID 1 → 1.50 m          | Marker IDs and physical edge lengths                        |
| `MAX_VALID_Z`   | `35.0`                                | Maximum accepted target distance (m); outliers are rejected |

> **Important:** `CAMERA_MATRIX` and `DIST_COEFFS` must match the deployed camera. Re-run camera calibration and update these values whenever the camera module, lens, or stream resolution changes. Marker sizes in `TARGETS` must match the printed markers exactly — a size mismatch scales all distance estimates.

### Flight Controller Setup (ArduCopter)

Suggested parameters for MAVLink-based precision landing:

```
PLND_ENABLED   = 1      # Enable precision landing
PLND_TYPE      = 1      # Companion computer (MAVLink)
PLND_EST_TYPE  = 0      # Raw sensor (or 1 for Kalman filter)
LAND_SPEED     = 30     # Gentle final descent (cm/s), adjust per airframe
```

Verify the serial port used by the companion link is configured for MAVLink (e.g. `SERIALx_PROTOCOL = 2`).

## Usage

Start the daemon on the companion computer:

```bash
python3 landing.py
```

Startup sequence:

1. Connects to the flight controller and waits (up to 5 s) for a heartbeat.
2. Spawns the RTSP reader thread and begins frame acquisition.
3. Enters the detection loop (~30 Hz ceiling); transmits `LANDING_TARGET` whenever a valid marker pose is obtained.

Stop with `Ctrl+C`; the video stream is released cleanly on exit.

### Console Output

| Message                                                     | Meaning                                                                      |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------- | ------------------------------------ |
| `[MAVLink] Heartbeat received successfully!`                | Flight controller link established                                           |
| `[ID:0(Small)] Front: …                                     | Right: …                                                                     | Down(Z): …` | Valid detection; position sent to FC |
| `[RATE] LANDING_TARGET send rate: 14.5 Hz (29 msgs / 2.0s)` | Rolling transmit-rate report (target ≥ 10 Hz for reliable precision landing) |
| `[WARNING] Target Lost! Stop sending MAVLink.`              | Marker no longer detected; transmission paused                               |

## Troubleshooting

| Symptom                                            | Likely Cause / Remedy                                                                                                     |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| No heartbeat warning at startup                    | Check UART wiring, baud rate, and the flight controller's `SERIALx_PROTOCOL` setting                                      |
| No frames / repeated reconnects                    | Verify camera IP and RTSP path; confirm the companion computer can reach `192.168.144.25`                                 |
| Send rate well below 10 Hz                         | Detection is CPU-bound or the stream frame rate is low; reduce stream resolution or check companion computer load         |
| Distances are consistently wrong by a scale factor | Printed marker size does not match `TARGETS` configuration                                                                |
| No detection at altitude                           | Large marker too small, glossy, or low-contrast; confirm the print is AprilTag **36h11** (not a 4×4/5×5 ArUco dictionary) |
| Spurious detections                                | Already mitigated by strict detector thresholds; if persisting, further reduce `maxErroneousBitsInBorderRate`             |

## Safety Notice

This system provides guidance _inputs_ to the flight controller; final landing behavior is governed by ArduCopter's precision-landing logic and tuning. Always validate detection performance and transmit rates on the ground, then in guided flight at safe altitude, before relying on the system for autonomous landings. Maintain manual override capability at all times during flight testing.

---

**Allswell Technology Services** — Precision Landing Division
