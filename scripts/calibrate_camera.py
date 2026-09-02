#!/usr/bin/env python3
"""Camera intrinsics calibration for the Android precision landing app.

Computes the camera matrix and distortion coefficients from chessboard
pictures (or a video) taken with the phone, and writes a JSON file that can be
pasted directly into the app's Settings > Camera intrinsics screen.

IMPORTANT: calibrate at the same aspect ratio the app analyzes at (16:9 by
default). The easiest way is to record a 1920x1080 video with the MAIN (wide)
camera while slowly moving around a printed chessboard, then run:

    python3 scripts/calibrate_camera.py --video calib.mp4 --frame-step 15

Or with individual photos (must all share one resolution):

    python3 scripts/calibrate_camera.py "calib_photos/*.jpg"

Print a chessboard pattern (e.g. 10x7 squares = 9x6 inner corners), tape it to
something rigid, and capture 15-30 views: near/far, all four screen corners,
and tilted up to ~45 degrees. Requires: pip install opencv-python numpy

The resulting JSON looks like:
    {"image_width": 1920, "image_height": 1080,
     "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]],
     "dist_coeffs": [k1,k2,p1,p2,k3], "rms": 0.31}
"""

import argparse
import glob
import json
import sys

import cv2
import numpy as np


def iter_images_from_files(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"No images match {pattern!r}")
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"[WARN] Could not read {path}, skipping")
            continue
        yield path, image


def iter_images_from_video(path, frame_step):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        sys.exit(f"Could not open video {path!r}")
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % frame_step == 0:
            yield f"{path}#frame{index}", frame
        index += 1
    capture.release()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("images", nargs="?",
                        help='glob of calibration photos, e.g. "calib/*.jpg"')
    parser.add_argument("--video", help="calibration video file (alternative to photos)")
    parser.add_argument("--frame-step", type=int, default=15,
                        help="use every Nth video frame (default: 15)")
    parser.add_argument("--cols", type=int, default=9,
                        help="chessboard INNER corners per row (default: 9)")
    parser.add_argument("--rows", type=int, default=6,
                        help="chessboard INNER corners per column (default: 6)")
    parser.add_argument("--square-size", type=float, default=0.025,
                        help="square edge length in meters (default: 0.025; "
                             "does not affect intrinsics, only reprojection scale)")
    parser.add_argument("-o", "--output", default="camera_intrinsics.json",
                        help="output JSON path (default: camera_intrinsics.json)")
    args = parser.parse_args()

    if not args.images and not args.video:
        parser.error("provide a photo glob or --video")

    pattern_size = (args.cols, args.rows)
    object_grid = np.zeros((args.cols * args.rows, 3), np.float32)
    object_grid[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    object_grid *= args.square_size

    object_points = []
    image_points = []
    image_size = None
    used = 0
    total = 0

    source = (iter_images_from_video(args.video, args.frame_step)
              if args.video else iter_images_from_files(args.images))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for name, image in source:
        total += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            print(f"[WARN] {name} is {size}, expected {image_size}, skipping")
            continue

        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            print(f"[----] no chessboard: {name}")
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_grid)
        image_points.append(corners)
        used += 1
        print(f"[ OK ] {name}")

    if used < 8:
        sys.exit(f"Only {used}/{total} views had a detectable chessboard; "
                 f"need at least 8 (ideally 15+). Re-shoot and try again.")

    print(f"\nCalibrating from {used} views at {image_size[0]}x{image_size[1]} ...")
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    dist = dist_coeffs.flatten().tolist()
    dist = (dist + [0.0] * 5)[:5]  # k1 k2 p1 p2 k3

    result = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "camera_matrix": [[float(v) for v in row] for row in camera_matrix],
        "dist_coeffs": [float(v) for v in dist],
        "rms": float(rms),
        "views_used": used,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    fx = camera_matrix[0][0]
    fy = camera_matrix[1][1]
    print(f"RMS reprojection error: {rms:.3f} px "
          f"({'good' if rms < 0.5 else 'acceptable' if rms < 1.0 else 'poor - re-shoot'})")
    print(f"fx={fx:.1f}  fy={fy:.1f}  cx={camera_matrix[0][2]:.1f}  cy={camera_matrix[1][2]:.1f}")
    print(f"\nWrote {args.output}")
    print("Paste its contents into the app: Settings > Camera intrinsics > Import")


if __name__ == "__main__":
    main()
