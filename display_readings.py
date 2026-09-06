"""Decode dark-background seven-segment digits and compass markers."""

from dataclasses import dataclass
from math import atan2, degrees
from typing import Any

import cv2
import numpy as np

from display_masks import MaskBox, digit_zone_pixels


SEGMENTS = {
    "0": frozenset("abcdef"),
    "1": frozenset("bc"),
    "2": frozenset("abdeg"),
    "3": frozenset("abcdg"),
    "4": frozenset("bcfg"),
    "5": frozenset("acdfg"),
    "6": frozenset("acdefg"),
    "7": frozenset("abc"),
    "8": frozenset("abcdefg"),
    "9": frozenset("abcdfg"),
}

@dataclass(frozen=True)
class DigitReading:
    """One decoded digit and the sampled segment state."""

    character: str
    confidence: float
    segments: frozenset[str]


@dataclass(frozen=True)
class CompassReading:
    """Compass marker direction, measured clockwise from twelve o'clock."""

    angle_degrees: float | None
    confidence: float
    sector_index: int | None = None


def _digits_to_number(digits: list[DigitReading], decimal_places: int = 0) -> float | None:
    """Convert ordered digit readings into a number, or None if any digit is unknown."""
    text = "".join(digit.character for digit in digits)
    if not text or "?" in text:
        return None
    value = int(text)
    return value / (10**decimal_places)


def _has_sign(image: np.ndarray, box: MaskBox) -> bool:
    """Return whether the configured sign region contains a lit minus sign."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    left, top, right, bottom = box.pixels(gray)
    return bool(gray[top:bottom, left:right].mean() > 127.5)


def decode_digit(image: np.ndarray, box: MaskBox) -> DigitReading:
    """Decode one digit by thresholding each segment zone at half brightness."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scores = {segment: float(gray[top:bottom, left:right].mean())
              for segment, (left, top, right, bottom) in digit_zone_pixels(gray, box).items()}
    active = frozenset(segment for segment, value in scores.items() if value > 127.5)
    if not active:
        return DigitReading("0", 1.0, active)
    character, expected = min(
        SEGMENTS.items(), key=lambda item: len(active ^ item[1])
    )
    distance = len(active ^ expected)
    confidence = max(0.0, 1.0 - distance / 7.0)
    if distance > 2:
        character = "?"
    return DigitReading(character, confidence, active)


def _mask_pixels(image: np.ndarray, box: MaskBox) -> tuple[int, int, int, int]:
    return box.pixels(image)


def detect_compass_marker(
    image: np.ndarray,
    circle: MaskBox,
    exclusions: list[MaskBox] | None = None,
) -> CompassReading:
    """Estimate a compass marker from the whitest ring sector on average.

    The returned angle is clockwise from twelve o'clock. Digit boxes are excluded
    because S2 contains its numeric wind direction inside the compass ellipse.
    Each ring sector is compared using its mean grayscale value, so a sector is
    selected for being brighter overall rather than for containing more bright
    pixels.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    left, top, right, bottom = _mask_pixels(gray, circle)
    roi = gray[top:bottom, left:right]
    if roi.size == 0:
        return CompassReading(None, 0.0)
    height, width = roi.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    nx = (xx - width / 2) / max(1.0, width / 2)
    ny = (yy - height / 2) / max(1.0, height / 2)
    radius = np.sqrt(nx * nx + ny * ny)
    candidate = radius <= 1.0
    if circle.inner_circle is not None:
        inner_left, inner_top, inner_right, inner_bottom = circle.inner_circle.pixels(gray)
        inner_nx = (xx + left - (inner_left + inner_right) / 2) / max(1.0, (inner_right - inner_left) / 2)
        inner_ny = (yy + top - (inner_top + inner_bottom) / 2) / max(1.0, (inner_bottom - inner_top) / 2)
        candidate &= inner_nx * inner_nx + inner_ny * inner_ny >= 1.0
    for exclusion in exclusions or []:
        ex_left, ex_top, ex_right, ex_bottom = _mask_pixels(gray, exclusion)
        candidate[max(0, ex_top - top):max(0, ex_bottom - top),
                  max(0, ex_left - left):max(0, ex_right - left)] = False
    values = roi[candidate]
    if values.size < 10:
        return CompassReading(None, 0.0)

    angles = np.arctan2(nx, -ny)
    sector_angles = (angles + 2 * np.pi) % (2 * np.pi)
    sector_width = 2 * np.pi / circle.ring_segments
    centered_sector_angles = (sector_angles + sector_width / 2) % (2 * np.pi)

    sector_indices = (centered_sector_angles / sector_width).astype(int)
    sector_means = np.full(circle.ring_segments, -np.inf, dtype=np.float32)
    for index in range(circle.ring_segments):
        sector_pixels = candidate & (sector_indices == index)
        if np.any(sector_pixels):
            sector_means[index] = float(roi[sector_pixels].mean())

    # Sector zero is centered on north, rather than starting at north.
    sector_index = int(np.argmax(sector_means))
    if float(values.max()) < circle.minimum_brightness:
        return CompassReading(None, 0.0)

    marker = candidate & (sector_indices == sector_index)
    marker_values = roi[marker].astype(np.float32)
    marker_angles = angles[marker]
    vector = np.sum(marker_values * np.exp(1j * marker_angles))
    angle = degrees(atan2(vector.imag, vector.real)) % 360.0
    finite_means = np.sort(sector_means[np.isfinite(sector_means)])
    second_mean = float(finite_means[-2]) if len(finite_means) > 1 else 0.0
    confidence = float(
        max(0.0, sector_means[sector_index] - second_mean)
        / max(1.0, float(sector_means[sector_index]))
    )
    return CompassReading(angle, confidence, sector_index)


def recognize_sectors(
    sectors: list[np.ndarray], masks: dict[str, list[MaskBox]]
) -> dict[str, Any]:
    """Decode configured display fields into raw digits and weather measurements."""
    readings: dict[str, Any] = {}
    for index, image in enumerate(sectors, start=1):
        name = f"S{index}"
        sector_masks = masks.get(name, [])
        digit_masks = [mask for mask in sector_masks if mask.role in {"top_digits", "bottom_digits", "digits"}]
        digits = [decode_digit(image, mask) for mask in digit_masks]
        result: dict[str, Any] = {"digits": digits, "text": "".join(d.character for d in digits)}
        grouped_digits: dict[str, list[DigitReading]] = {}
        for mask, digit in zip(digit_masks, digits):
            grouped_digits.setdefault(mask.role, []).append(digit)

        if name in {"S1", "S3"}:
            top = grouped_digits.get("top_digits", [])
            bottom = grouped_digits.get("bottom_digits", [])
            sign_mask = next((mask for mask in sector_masks if mask.role == "sign"), None)
            temperature = _digits_to_number(top, decimal_places=1)
            if temperature is not None and sign_mask is not None and _has_sign(image, sign_mask):
                temperature = -temperature
            result["measurement"] = {
                "temperature_c": temperature,
                "humidity_percent": _digits_to_number(bottom),
            }
        elif name == "S2":
            result["measurement"] = {"wind_speed_kmh": _digits_to_number(digits, decimal_places=1)}
        elif name == "S4":
            result["measurement"] = {"hourly_rainfall_mm": _digits_to_number(digits, decimal_places=1)}
        elif name == "S6":
            result["measurement"] = {"pressure_kpa": _digits_to_number(digits)}

        circles = [mask for mask in sector_masks if mask.shape == "circle"]
        if circles:
            compass = detect_compass_marker(image, circles[0], digit_masks)
            result["compass"] = compass
            if name == "S2":
                result.setdefault("measurement", {})["wind_direction_degrees"] = (
                    None if compass.angle_degrees is None else int(round(compass.angle_degrees))
                )
        readings[name] = result
    return readings


def format_readings(readings: dict[str, Any]) -> list[str]:
    """Create concise weather measurement lines for the terminal or OpenCV overlay."""
    lines = []
    for name, result in readings.items():
        measurement = result.get("measurement")
        if name == "S1" and measurement is not None:
            lines.append(
                f"S1 outside: {_format_value(measurement['temperature_c'], ' °C')}, "
                f"humidity {_format_value(measurement['humidity_percent'], '%')}"
            )
        elif name == "S2" and measurement is not None:
            lines.append(
                f"S2 wind: {_format_value(measurement['wind_speed_kmh'], ' km/h')}, "
                f"direction {_format_value(measurement['wind_direction_degrees'], '°')}"
            )
        elif name == "S3" and measurement is not None:
            lines.append(
                f"S3 inside: {_format_value(measurement['temperature_c'], ' °C')}, "
                f"humidity {_format_value(measurement['humidity_percent'], '%')}"
            )
        elif name == "S4" and measurement is not None:
            lines.append(f"S4 hourly rainfall: {_format_value(measurement['hourly_rainfall_mm'], ' mm')}")
        elif name == "S6" and measurement is not None:
            lines.append(f"S6 pressure: {_format_value(measurement['pressure_kpa'], ' kPa')}")
        else:
            lines.append(f"{name}: {result['text'] or '-'}")
    return lines


def _format_value(value: float | None, suffix: str) -> str:
    """Format a numeric measurement while keeping unavailable values readable."""
    return "?" if value is None else f"{value:g}{suffix}"