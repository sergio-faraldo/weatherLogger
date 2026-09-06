#!/usr/bin/env python
"""Display a live camera feed with the saved camera calibration applied."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the camera matrix and distortion coefficients from a JSON file."""
    with path.open(encoding="utf-8") as file:
        calibration = json.load(file)

    try:
        camera_matrix = np.asarray(calibration["mtx"], dtype=np.float64)
        distortion = np.asarray(calibration["dist"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid calibration file: {path}") from error

    if camera_matrix.shape != (3, 3) or distortion.size == 0:
        raise ValueError(f"Invalid camera matrix or distortion data in {path}")

    return camera_matrix, distortion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera device index (default: 0)")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(__file__).with_name("calibration.json"),
        help="path to the calibration JSON file",
    )
    args = parser.parse_args()

    camera_matrix, distortion = load_calibration(args.calibration)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    map_x = map_y = None
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera")

            if map_x is None or map_x.shape[:2] != frame.shape[:2]:
                height, width = frame.shape[:2]
                new_matrix, _ = cv2.getOptimalNewCameraMatrix(
                    camera_matrix, distortion, (width, height), 1, (width, height)
                )
                map_x, map_y = cv2.initUndistortRectifyMap(
                    camera_matrix, distortion, None, new_matrix, (width, height), cv2.CV_32FC1
                )

            corrected = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            cv2.imshow("Calibrated camera", corrected)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()