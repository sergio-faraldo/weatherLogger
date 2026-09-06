#!/usr/bin/env python
"""Capture calibration images from a live camera feed at a fixed interval."""

import argparse
import time
from pathlib import Path

import cv2


WINDOW_NAME = "Calibration capture"
CAPTURE_INTERVAL_SECONDS = 3.0
BORDER_COLOR = (255, 0, 0)  # Blue in OpenCV's BGR format.
BORDER_THICKNESS = 12
BORDER_DISPLAY_SECONDS = 0.35


def next_image_number(output_directory: Path) -> int:
    """Return the next sequential number for a calibration image."""
    numbers = []
    for image_path in output_directory.glob("*.jpg"):
        try:
            numbers.append(int(image_path.stem))
        except ValueError:
            continue
    return max(numbers, default=0) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera device index (default: 0)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("calibration-images"),
        help="directory for captured JPEG images",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=CAPTURE_INTERVAL_SECONDS,
        help="seconds between captures (default: 3)",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    args.output.mkdir(parents=True, exist_ok=True)
    image_number = next_image_number(args.output)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    next_capture = time.monotonic() + args.interval
    border_until = 0.0
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera")

            now = time.monotonic()
            if now >= next_capture:
                image_path = args.output / f"{image_number:04d}.jpg"
                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError(f"Could not save calibration image: {image_path}")
                print(f"Captured {image_path}")
                image_number += 1
                border_until = now + BORDER_DISPLAY_SECONDS
                next_capture = now + args.interval

            preview = frame.copy()
            if now < border_until:
                cv2.rectangle(
                    preview,
                    (0, 0),
                    (preview.shape[1] - 1, preview.shape[0] - 1),
                    BORDER_COLOR,
                    BORDER_THICKNESS,
                )

            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
