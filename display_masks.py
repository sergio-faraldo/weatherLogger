"""Load and visualize normalized sampling regions for display sectors."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from display_analysis import (
    detect_display_grid,
    enhance_image,
    extract_top_sectors,
    load_calibration,
    make_sector_mosaic,
    prepare_template,
    process_image,
    undistort_image,
)


@dataclass(frozen=True)
class MaskBox:
    """A normalized mask region and its corresponding pixel bounds."""

    x: float
    y: float
    width: float
    height: float
    shape: str = "rectangle"
    role: str = "region"
    inner_circle: MaskBox | None = None
    ring_segments: int = 16
    minimum_brightness: int = 120

    def pixels(self, image: Any) -> tuple[int, int, int, int]:
        image_height, image_width = image.shape[:2]
        return (
            round(self.x * image_width),
            round(self.y * image_height),
            round((self.x + self.width) * image_width),
            round((self.y + self.height) * image_height),
        )


def _box(value: dict[str, Any], location: str, role: str = "region") -> MaskBox:
    try:
        box = MaskBox(
            x=float(value["x"]),
            y=float(value["y"]),
            width=float(value["width"]),
            height=float(value["height"]),
            role=role,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid mask box at {location}") from error
    if (
        not 0 <= box.x <= 1
        or not 0 <= box.y <= 1
        or not 0 < box.width <= 1
        or not 0 < box.height <= 1
    ):
        raise ValueError(f"Mask box is outside normalized coordinates at {location}")
    if box.x + box.width > 1 or box.y + box.height > 1:
        raise ValueError(f"Mask box extends outside its sector at {location}")
    return box


def load_masks(path: Path) -> dict[str, list[MaskBox]]:
    """Load all mask boxes, flattening each sector's box lists."""
    with path.open(encoding="utf-8") as file:
        configuration = json.load(file)
    try:
        sectors = configuration["sectors"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid mask configuration: {path}") from error

    result: dict[str, list[MaskBox]] = {}
    for sector_name, specification in sectors.items():
        if not isinstance(specification, dict):
            raise ValueError(f"Invalid specification for {sector_name}")
        boxes: list[MaskBox] = []
        for key, value in specification.items():
            values = value if isinstance(value, list) else [value]
            for index, item in enumerate(values):
                if not isinstance(item, dict):
                    raise ValueError(f"Invalid mask box at {sector_name}.{key}[{index}]")
                box = _box(item, f"{sector_name}.{key}[{index}]", role=key)
                if key == "compass_circle":
                    inner = item.get("inner_circle")
                    inner_box = (
                        _box(inner, f"{sector_name}.{key}[{index}].inner_circle", role="inner_circle")
                        if inner is not None else None
                    )
                    segments = int(item.get("ring_segments", 16))
                    if segments < 1:
                        raise ValueError(f"ring_segments must be positive at {sector_name}.{key}[{index}]")
                    box = MaskBox(
                        box.x, box.y, box.width, box.height,
                        shape="circle", role=key, inner_circle=inner_box,
                        ring_segments=segments,
                        minimum_brightness=int(item.get("minimum_brightness", 120)),
                    )
                boxes.append(box)
        result[sector_name] = boxes
    return result


def annotate_masks(sectors: list[Any], masks: dict[str, list[MaskBox]]) -> list[Any]:
    """Return sector images annotated with the configured pixel rectangles."""
    annotated = []
    for index, sector in enumerate(sectors, start=1):
        image = sector.copy()
        sector_masks = masks.get(f"S{index}", [])
        for mask in sector_masks:
            left, top, right, bottom = mask.pixels(image)
            if mask.shape == "circle":
                center = ((left + right) // 2, (top + bottom) // 2)
                axes = ((right - left) // 2, (bottom - top) // 2)
                cv2.ellipse(image, center, axes, 0, 0, 360, (0, 255, 255), 1)
                if mask.inner_circle is not None:
                    inner_left, inner_top, inner_right, inner_bottom = mask.inner_circle.pixels(image)
                    inner_center = ((inner_left + inner_right) // 2, (inner_top + inner_bottom) // 2)
                    inner_axes = ((inner_right - inner_left) // 2, (inner_bottom - inner_top) // 2)
                    cv2.ellipse(image, inner_center, inner_axes, 0, 0, 360, (255, 255, 0), 1)
                    for sector in range(mask.ring_segments):
                        angle = 2 * np.pi * (sector + 0.5) / mask.ring_segments - np.pi / 2
                        outer_point = (
                            round(center[0] + axes[0] * np.cos(angle)),
                            round(center[1] + axes[1] * np.sin(angle)),
                        )
                        inner_point = (
                            round(inner_center[0] + inner_axes[0] * np.cos(angle)),
                            round(inner_center[1] + inner_axes[1] * np.sin(angle)),
                        )
                        cv2.line(image, inner_point, outer_point, (0, 255, 0), 1)
            elif mask.role in {"top_digits", "bottom_digits", "digits"}:
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 255), 1)
                _annotate_digit_zones(image, mask)
            else:
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 255), 1)
        annotated.append(image)
    return annotated


def digit_zone_pixels(
    image: Any,
    box: MaskBox,
    horizontal_fraction: float = 0.14,
    vertical_fraction: float = 0.28,
) -> dict[str, tuple[int, int, int, int]]:
    """Return seven segment zones grown inward from a tightly fitted digit box."""
    left, top, right, bottom = box.pixels(image)
    width, height = max(2, right - left), max(2, bottom - top)
    horizontal = max(1, round(height * horizontal_fraction))
    vertical = max(1, round(width * vertical_fraction))
    middle = (top + bottom) // 2
    half_middle = max(1, vertical // 2)
    return {
        "a": (left + vertical, top, right - vertical, top + horizontal),
        "b": (right - horizontal, top + vertical, right, middle - half_middle),
        "c": (right - horizontal, middle + half_middle, right, bottom - vertical),
        "d": (left + vertical, bottom - horizontal, right - vertical, bottom),
        "e": (left, middle + half_middle, left + horizontal, bottom - vertical),
        "f": (left, top + vertical, left + horizontal, middle - half_middle),
        "g": (left + vertical, middle - half_middle, right - vertical, middle + half_middle),
    }


def _annotate_digit_zones(image: Any, box: MaskBox) -> None:
    colors = {"a": (255, 0, 0), "b": (255, 80, 0), "c": (255, 160, 0), "d": (0, 180, 255),
              "e": (0, 255, 180), "f": (0, 255, 80), "g": (0, 255, 255)}
    for segment, (left, top, right, bottom) in digit_zone_pixels(image, box).items():
        cv2.rectangle(image, (left, top), (right, bottom), colors[segment], 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, default=Path("./photos/0002.jpg"))
    parser.add_argument("--masks", type=Path, default=Path(__file__).with_name("masks.json"))
    parser.add_argument(
        "--calibration", type=Path, default=Path(__file__).with_name("calibration.json")
    )
    parser.add_argument(
        "--template", type=Path, default=Path(__file__).with_name("template.jpg")
    )
    parser.add_argument("--contrast", type=float, default=1.25)
    parser.add_argument("--sharpen", type=float, default=1.0)
    parser.add_argument("--denoise", type=float, default=3.0)
    parser.add_argument("--gate", type=int, default=0, help="binary threshold after enhancement (0 disables; otherwise 0-255)")
    args = parser.parse_args()
    if args.contrast < 0 or args.sharpen < 0 or args.denoise < 0 or not 0 <= args.gate <= 255:
        parser.error("--contrast, --sharpen, and --denoise must not be negative; --gate must be 0-255")

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    template = cv2.imread(str(args.template), cv2.IMREAD_COLOR)
    if image is None or template is None:
        raise RuntimeError("Could not read the input image or template")
    template = enhance_image(template, args.contrast, args.sharpen, args.denoise, args.gate)
    camera_matrix, distortion = load_calibration(args.calibration)
    corrected = enhance_image(
        undistort_image(image, camera_matrix, distortion),
        args.contrast,
        args.sharpen,
        args.denoise,
        args.gate,
    )
    detector = cv2.SIFT_create()
    keypoints, descriptors = prepare_template(template, detector)
    result = process_image(corrected, template, keypoints, descriptors, detector)
    if not result.found:
        raise RuntimeError("Display was not found in the image")

    grid = detect_display_grid(template)
    sectors = extract_top_sectors(result.rectified, grid)
    annotated = annotate_masks(sectors, load_masks(args.masks))
    cv2.imshow("Configured masks - press Q or Esc", make_sector_mosaic(annotated, scale=4.0))
    while cv2.waitKey(0) & 0xFF not in (ord("q"), 27):
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()