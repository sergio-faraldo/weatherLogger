"""Track an LCD/LED display from a live calibrated camera feed."""

import argparse
import csv
import gzip
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, Iterable

import cv2
import numpy as np

from display_analysis import (
    annotate_top_sectors,
    detect_display_grid,
    enhance_image,
    extract_top_sectors,
    load_calibration,
    make_sector_mosaic,
    prepare_template,
    process_image,
    undistort_image,
)
from display_masks import load_masks
from display_readings import format_readings, recognize_sectors


MEASUREMENT_COLUMNS = (
    ("S1", "temperature_c", "outside_temperature_c"),
    ("S1", "humidity_percent", "outside_humidity_percent"),
    ("S2", "wind_speed_kmh", "wind_speed_kmh"),
    ("S2", "wind_direction_degrees", "wind_direction_degrees"),
    ("S3", "temperature_c", "inside_temperature_c"),
    ("S3", "humidity_percent", "inside_humidity_percent"),
    ("S4", "hourly_rainfall_mm", "hourly_rainfall_mm"),
    ("S6", "pressure_kpa", "pressure_kpa"),
)


def read_average_frame(camera: cv2.VideoCapture, frame_count: int) -> np.ndarray:
    """Read and average a batch of frames from a stationary camera."""
    accumulated: np.ndarray | None = None
    for _ in range(frame_count):
        success, frame = camera.read()
        if not success:
            raise RuntimeError("Could not read a frame from the camera")
        if accumulated is None:
            accumulated = frame.astype(np.float32)
        else:
            accumulated += frame

    return np.rint(accumulated / frame_count).astype(np.uint8)


def most_common_measurements(readings: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Choose the most common value independently for every weather variable."""
    readings = list(readings)
    return {
        output_name: Counter(
            reading.get(sector, {}).get("measurement", {}).get(variable)
            for reading in readings
        ).most_common(1)[0][0]
        for sector, variable, output_name in MEASUREMENT_COLUMNS
    }


def rotate_readings_file(path: Path, timestamp: datetime | None = None) -> None:
    """Gzip a readings file when it was created on the previous calendar day."""
    if not path.exists():
        return

    now = timestamp or datetime.now()
    created = datetime.fromtimestamp(path.stat().st_ctime)
    if created.date() != now.date() and (now.date() - created.date()).days == 1:
        archive_directory = path.parent / "older-data"
        archive_directory.mkdir(parents=True, exist_ok=True)
        archive_path = archive_directory / f"readings-{created:%Y-%m-%d}.gz"
        with path.open("rb") as source, gzip.open(archive_path, "wb") as archive:
            archive.write(source.read())
        path.unlink()


def append_csv_reading(
    path: Path,
    measurements: dict[str, Any],
    timestamp: datetime | None = None,
) -> None:
    """Append one timestamped measurement row, creating the CSV header if needed."""
    fieldnames = ["timestamp", *(column[2] for column in MEASUREMENT_COLUMNS)]
    timestamp = timestamp or datetime.now()
    rotate_readings_file(path, timestamp)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp.isoformat(timespec="seconds"),
                **measurements,
            }
        )


def find_first_usb_mount() -> Path | None:
    """Return the first mounted Linux USB storage path, if one is available."""
    if sys.platform != "linux":
        return None

    try:
        result = subprocess.run(
            ["lsblk", "--json", "--paths", "--output", "NAME,TRAN,MOUNTPOINT,TYPE"],
            capture_output=True,
            check=False,
            text=True,
        )
        devices = json.loads(result.stdout).get("blockdevices", [])
    except (OSError, json.JSONDecodeError):
        return None

    def mounted_usb_path(device: dict[str, Any], transport: str | None = None) -> Path | None:
        transport = device.get("tran") or transport
        mountpoint = device.get("mountpoint")
        if transport == "usb" and mountpoint:
            path = Path(mountpoint)
            if path.is_dir():
                return path
        for child in device.get("children", []):
            path = mounted_usb_path(child, transport)
            if path is not None:
                return path
        return None

    for device in devices:
        path = mounted_usb_path(device)
        if path is not None:
            return path
    return None


def collect_recording(
    readings: dict[str, Any],
    collected_readings: list[dict[str, Any]],
    mode: str,
    output: Path,
    usb_output: Path | None = None,
) -> bool:
    """Store one reading and write a completed 11-reading batch.

    Return whether the caller should stop after the batch.
    """
    collected_readings.append(readings)
    if len(collected_readings) < 11:
        return False

    measurements = most_common_measurements(collected_readings)
    timestamp = datetime.now()
    append_csv_reading(output, measurements, timestamp)
    if usb_output is not None:
        try:
            append_csv_reading(usb_output, measurements, timestamp)
        except OSError as error:
            print(f"Could not write USB copy to {usb_output}: {error}", flush=True)
    if mode == "record-loop":
        collected_readings.clear()
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera device index (default: 0)")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(__file__).with_name("calibration.json"),
        help="path to the calibration JSON file",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).with_name("template.jpg"),
        help="display template image (default: template.jpg)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.7,
        help="SIFT match ratio threshold (default: 0.7)",
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=8,
        help="minimum RANSAC inliers required for a detection (default: 8)",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        default=Path(__file__).with_name("masks.json"),
        help="path to the normalized mask configuration (default: masks.json)",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.25,
        help="CLAHE contrast strength (default: 1.25; use 0 to disable)",
    )
    parser.add_argument(
        "--sharpen",
        type=float,
        default=1.25,
        help="unsharp-mask strength (default: 1.25; use 0 to disable)",
    )
    parser.add_argument(
        "--denoise",
        type=float,
        default=3.0,
        help="bilateral denoise strength before contrast and sharpening (default: 3.0; use 0 to disable)",
    )
    parser.add_argument(
        "--gate",
        type=int,
        default=100,
        help="binary threshold after enhancement (default: 100; 0 disables; otherwise 0-255)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=50,
        help="number of camera frames to average for each reading (default: 10)",
    )
    parser.add_argument(
        "--mode",
        choices=("live", "record", "record-loop"),
        default="record",
        help="display live readings, record 11 readings once, or continuously record batches of 11 readings (default: live)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("readings.csv"),
        help="CSV output path for record mode (default: readings.csv)",
    )
    parser.add_argument(
        "--usb-copy",
        action="store_true",
        help="also write readings.csv to the first mounted Linux USB storage device",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        default="true",
        help="disable OpenCV preview windows and keyboard handling",
    )
    args = parser.parse_args()
    if not 0 < args.ratio < 1:
        parser.error("--ratio must be between 0 and 1")
    if args.min_inliers < 4:
        parser.error("--min-inliers must be at least 4")
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    if args.contrast < 0 or args.sharpen < 0 or args.denoise < 0 or not 0 <= args.gate <= 255:
        parser.error("--contrast, --sharpen, and --denoise must not be negative; --gate must be 0-255")

    usb_output = None
    if args.usb_copy:
        usb_mount = find_first_usb_mount()
        if usb_mount is None:
            print("USB copy requested, but no mounted Linux USB storage device was found", flush=True)
        else:
            usb_output = usb_mount / "readings.csv"

    camera_matrix, distortion = load_calibration(args.calibration)
    masks = load_masks(args.masks)
    template = cv2.imread(str(args.template), cv2.IMREAD_COLOR)
    if template is None:
        raise RuntimeError(f"Could not read template image: {args.template}")
    template = enhance_image(template, args.contrast, args.sharpen, args.denoise, args.gate)
    grid = detect_display_grid(template)
    detector = cv2.SIFT_create()
    template_keypoints, template_descriptors = prepare_template(template, detector)
    if len(template_keypoints) < args.min_inliers:
        raise RuntimeError(f"Template does not contain enough features: {args.template}")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    last_printed: tuple[str, ...] | None = None
    collected_readings: list[dict[str, Any]] = []
    try:
        while True:
            corrected = enhance_image(
                undistort_image(read_average_frame(camera, args.frames), camera_matrix, distortion),
                args.contrast,
                args.sharpen,
                args.denoise,
                args.gate,
            )
            result = process_image(
                corrected,
                template,
                template_keypoints,
                template_descriptors,
                detector,
                minimum_inliers=args.min_inliers,
                ratio=args.ratio,
            )

            preview = corrected.copy()
            if not result.found:
                status = "Display not found"
            else:
                cv2.polylines(preview, [np.int32(result.corners)], True, (0, 255, 0), 2)
                status = f"Display tracked ({result.inlier_count} inliers)"
                rectified_preview = annotate_top_sectors(result.rectified, grid)
                sectors = extract_top_sectors(result.rectified, grid)
                readings = recognize_sectors(sectors, masks)
                if args.mode in ("record", "record-loop"):
                    batch_complete = len(collected_readings) == 10
                    should_stop = collect_recording(
                        readings,
                        collected_readings,
                        args.mode,
                        args.output,
                        usb_output,
                    )
                    if batch_complete:
                        print(f"Recorded reading to {args.output}", flush=True)
                    if should_stop:
                        break
                reading_lines = tuple(format_readings(readings))
                if reading_lines != last_printed:
                    print("\n".join(reading_lines), flush=True)
                    last_printed = reading_lines
                if not args.no_ui:
                    cv2.imshow(
                        "Top two rows - sectors S1 to S6",
                        make_sector_mosaic(sectors, scale=2.0),
                    )

            if not args.no_ui:
                cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Camera - Q/Esc: quit", preview)
                cv2.imshow(
                    "Rectified display (template coordinates)",
                    rectified_preview if result.found else result.rectified,
                )
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            sleep(30) # wait 30 seconds until next reading
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
