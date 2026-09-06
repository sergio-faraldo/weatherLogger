"""Detect and show the perspective-corrected display in one photo."""

import argparse
from pathlib import Path

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
from display_masks import annotate_masks, load_masks
from display_readings import format_readings, recognize_sectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=Path("./photos/0002.jpg"),
        help="photo to analyze (default: ./photos/0002.jpg)",
    )
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
        help="unsharp-mask strength (default: 1.0; use 0 to disable)",
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
        help="binary threshold after enhancement (0 disables; otherwise 0-255)",
    )
    args = parser.parse_args()
    if not 0 < args.ratio < 1:
        parser.error("--ratio must be between 0 and 1")
    if args.min_inliers < 4:
        parser.error("--min-inliers must be at least 4")
    if args.contrast < 0 or args.sharpen < 0 or args.denoise < 0 or not 0 <= args.gate <= 255:
        parser.error("--contrast, --sharpen, and --denoise must not be negative; --gate must be 0-255")

    camera_matrix, distortion = load_calibration(args.calibration)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {args.image}")
    template = cv2.imread(str(args.template), cv2.IMREAD_COLOR)
    if template is None:
        raise RuntimeError(f"Could not read template image: {args.template}")
    template = enhance_image(template, args.contrast, args.sharpen, args.denoise, args.gate)
    grid = detect_display_grid(template)
    masks = load_masks(args.masks)

    detector = cv2.SIFT_create()
    template_keypoints, template_descriptors = prepare_template(template, detector)
    if len(template_keypoints) < args.min_inliers:
        raise RuntimeError(f"Template does not contain enough features: {args.template}")

    corrected = enhance_image(
        undistort_image(image, camera_matrix, distortion),
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
    rectified_preview = result.rectified
    readings = {}
    if not result.found:
        status = "Display not found"
    else:
        cv2.polylines(preview, [np.int32(result.corners)], True, (0, 255, 0), 2)
        status = f"Display found ({result.inlier_count} inliers)"
        rectified_preview = annotate_top_sectors(result.rectified, grid)
        sectors = extract_top_sectors(result.rectified, grid)
        readings = recognize_sectors(sectors, masks)
        print("\n".join(format_readings(readings)))
        mask_window = "Top two rows - sectors S1 to S6 with masks"

        def show_masks() -> None:
            annotated_sectors = annotate_masks(sectors, masks)
            cv2.imshow(mask_window, make_sector_mosaic(annotated_sectors, scale=6.0))

        show_masks()
    cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if result.found:
        for index, line in enumerate(format_readings(readings), start=1):
            cv2.putText(preview, line, (10, 30 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    cv2.imshow("Analyzed photo - R: reload masks, Q/Esc: quit", preview)
    cv2.imshow("Rectified display with grid", rectified_preview if result.found else result.rectified)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r") and result.found:
            masks = load_masks(args.masks)
            show_masks()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
