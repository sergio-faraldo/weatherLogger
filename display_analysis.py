"""Detect and perspective-correct an LCD/LED display in an image."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DisplayResult:
    """Results from processing one image for the tracked display."""

    rectified: np.ndarray
    homography: np.ndarray | None
    corners: np.ndarray | None
    inlier_count: int

    @property
    def found(self) -> bool:
        """Whether the display was detected in the image."""
        return self.homography is not None


@dataclass
class DisplayGrid:
    """Grid boundaries in rectified display coordinates."""

    x_boundaries: tuple[int, ...]
    y_boundaries: tuple[int, ...]


def _line_centers(scores: np.ndarray, minimum_score: float) -> list[int]:
    """Return the centers of contiguous strong line-score runs."""
    strong = scores >= minimum_score
    changes = np.diff(np.concatenate(([False], strong, [False])).astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [int((start + end - 1) // 2) for start, end in zip(starts, ends)]


def _strongest_separated_lines(
    scores: np.ndarray, candidates: list[int], count: int, minimum_separation: int
) -> list[int]:
    """Keep the strongest candidates while avoiding duplicate line detections."""
    selected: list[int] = []
    for candidate in sorted(candidates, key=lambda index: scores[index], reverse=True):
        if all(abs(candidate - other) >= minimum_separation for other in selected):
            selected.append(candidate)
            if len(selected) == count:
                break
    return sorted(selected)


def _strongest_line_in_band(scores: np.ndarray, start: int, end: int) -> int:
    """Return the strongest line score within a bounded y-coordinate range."""
    if start >= end:
        raise ValueError("Line-detection band is empty")
    return start + int(np.argmax(scores[start:end]))


def detect_display_grid(
    image: np.ndarray,
    white_threshold: int = 160,
    maximum_saturation: int = 90,
) -> DisplayGrid:
    """Find the display's white grid lines in a rectified image.

    The returned boundaries include the outer display edges. The two internal
    vertical lines and the three horizontal lines are identified from their
    long, bright, low-saturation runs.
    """
    if image.ndim == 2:
        gray = image
        saturation = np.zeros_like(gray)
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        saturation = hsv[:, :, 1]

    white = (gray >= white_threshold) & (saturation <= maximum_saturation)
    height, width = gray.shape[:2]
    grid_height = max(1, int(height * 0.72))
    vertical_scores = white[:grid_height, :].sum(axis=0)
    horizontal_height = min(height, max(grid_height, int(height * 0.85)))
    horizontal_scores = white[:horizontal_height, :].sum(axis=1)

    vertical_candidates = _line_centers(vertical_scores, grid_height * 0.08)
    vertical_lines = _strongest_separated_lines(
        vertical_scores, vertical_candidates, count=2, minimum_separation=width // 5
    )
    if len(vertical_lines) != 2:
        raise ValueError("Could not identify the display grid: expected two vertical white lines")

    # The top guide is only present over S2, while the lower divider must be
    # chosen from the full display width so the landscape graphic in S5 does
    # not get mistaken for the boundary.
    middle_start = max(0, int(height * 0.30))
    middle_end = min(grid_height, int(height * 0.65))
    lower_start = max(0, int(height * 0.65))
    lower_end = min(horizontal_height, int(height * 0.85))
    top_start = 0
    top_end = min(grid_height, max(1, int(height * 0.20)))
    s2_start = vertical_lines[0] + 1
    s2_end = vertical_lines[1]
    top_scores = white[:top_end, s2_start:s2_end].sum(axis=1)
    horizontal_lines = [
        _strongest_line_in_band(top_scores, top_start, top_end),
        _strongest_line_in_band(horizontal_scores, middle_start, middle_end),
        _strongest_line_in_band(horizontal_scores, lower_start, lower_end),
    ]

    return DisplayGrid(
        x_boundaries=(0, *vertical_lines, width - 1),
        y_boundaries=tuple(horizontal_lines),
    )


def extract_top_sectors(
    rectified: np.ndarray,
    grid: DisplayGrid,
    line_margin: int = 2,
) -> list[np.ndarray]:
    """Crop the three columns in each of the display's top two rows."""
    if len(grid.x_boundaries) != 4 or len(grid.y_boundaries) < 3:
        raise ValueError("A display grid must contain four x boundaries and three y boundaries")

    sectors = []
    for row in range(2):
        top = grid.y_boundaries[row] + line_margin
        bottom = grid.y_boundaries[row + 1] - line_margin
        for column in range(3):
            left = grid.x_boundaries[column] + line_margin
            right = grid.x_boundaries[column + 1] - line_margin
            if top >= bottom or left >= right:
                raise ValueError("Display grid lines are too close together")
            sectors.append(rectified[top:bottom, left:right])
    return sectors


def make_sector_mosaic(
    sectors: list[np.ndarray], columns: int = 3, scale: float = 1.0
) -> np.ndarray:
    """Combine sectors without changing their aspect ratios and optionally scale them."""
    if not sectors or len(sectors) % columns:
        raise ValueError("The number of sectors must be a non-zero multiple of columns")
    if scale <= 0:
        raise ValueError("The mosaic scale must be positive")

    rows = []
    for start in range(0, len(sectors), columns):
        row_sectors = sectors[start : start + columns]
        row_height = max(sector.shape[0] for sector in row_sectors)
        row_width = sum(sector.shape[1] for sector in row_sectors)
        row = np.zeros((row_height, row_width, *sectors[0].shape[2:]), dtype=sectors[0].dtype)
        offset = 0
        for sector in row_sectors:
            height, width = sector.shape[:2]
            row[:height, offset : offset + width] = sector
            offset += width
        rows.append(row)

    mosaic_width = max(row.shape[1] for row in rows)
    mosaic_height = sum(row.shape[0] for row in rows)
    mosaic = np.zeros((mosaic_height, mosaic_width, *sectors[0].shape[2:]), dtype=sectors[0].dtype)
    offset = 0
    for row in rows:
        mosaic[offset : offset + row.shape[0], : row.shape[1]] = row
        offset += row.shape[0]

    if scale == 1:
        return mosaic
    size = (round(mosaic.shape[1] * scale), round(mosaic.shape[0] * scale))
    return cv2.resize(mosaic, size, interpolation=cv2.INTER_NEAREST)


def annotate_top_sectors(
    rectified: np.ndarray,
    grid: DisplayGrid,
    line_margin: int = 2,
) -> np.ndarray:
    """Draw the grid and sector labels on a rectified display image."""
    annotated = rectified.copy()
    for x in grid.x_boundaries:
        cv2.line(annotated, (x, grid.y_boundaries[0]), (x, grid.y_boundaries[-1]), (0, 0, 255), 1)
    for y in grid.y_boundaries:
        cv2.line(annotated, (grid.x_boundaries[0], y), (grid.x_boundaries[-1], y), (0, 0, 255), 1)

    for index, (row, column) in enumerate((
        (row, column) for row in range(2) for column in range(3)
    ), start=1):
        x = grid.x_boundaries[column] + line_margin + 4
        y = grid.y_boundaries[row] + line_margin + 22
        cv2.putText(annotated, f"S{index}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return annotated


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load camera calibration values in the same format as calibration.py."""
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


def undistort_image(
    image: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray
) -> np.ndarray:
    """Apply the saved camera calibration to one image."""
    height, width = image.shape[:2]
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion, (width, height), 1, (width, height)
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        new_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


def enhance_image(
    image: np.ndarray,
    contrast: float = 1.25,
    sharpen: float = 1.0,
    denoise: float = 3.0,
    gate: int = 0,
) -> np.ndarray:
    """Denoise, enhance, and optionally gate an image before analysis.

    Denoising uses an edge-preserving bilateral filter. Contrast is applied
    with CLAHE to the luminance channel, while sharpening uses an unsharp
    mask. If ``gate`` is non-zero, pixels at or above that grayscale
    threshold become white and all other pixels become black. All strengths
    may be set to zero independently.
    """
    if contrast < 0 or sharpen < 0 or denoise < 0 or not 0 <= gate <= 255:
        raise ValueError("contrast, sharpen, and denoise must not be negative; gate must be 0..255")
    if image.ndim == 2:
        lab = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        grayscale = True
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        grayscale = False

    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if denoise:
        enhanced = cv2.bilateralFilter(
            enhanced,
            d=0,
            sigmaColor=float(denoise) * 10.0,
            sigmaSpace=float(denoise),
        )
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    if contrast:
        luminance, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(contrast), tileGridSize=(8, 8))
        lab = cv2.merge((clahe.apply(luminance), a_channel, b_channel))

    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if sharpen:
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
        enhanced = cv2.addWeighted(enhanced, 1.0 + sharpen, blurred, -sharpen, 0)
    if gate:
        enhanced = gate_image(enhanced, gate)
    if grayscale:
        return cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    return enhanced


def gate_image(image: np.ndarray, threshold: int) -> np.ndarray:
    """Clip an image to full black or full white using a grayscale threshold."""
    if not 0 <= threshold <= 255:
        raise ValueError("gate threshold must be between 0 and 255")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, gated = cv2.threshold(gray, max(0, threshold - 1), 255, cv2.THRESH_BINARY)
    if image.ndim == 2:
        return gated
    return cv2.cvtColor(gated, cv2.COLOR_GRAY2BGR)


def prepare_template(
    template: np.ndarray, detector: cv2.SIFT
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Calculate the feature data needed to search for a display."""
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_keypoints, template_descriptors = detector.detectAndCompute(template_gray, None)
    if template_descriptors is None:
        raise ValueError("Template does not contain any detectable features")
    return template_keypoints, template_descriptors


def find_display_homography(
    template_keypoints: list[cv2.KeyPoint],
    template_descriptors: np.ndarray,
    frame: np.ndarray,
    detector: cv2.SIFT,
    minimum_inliers: int,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find the template-to-frame transform and its detected corner polygon."""
    frame_keypoints, frame_descriptors = detector.detectAndCompute(frame, None)
    if frame_descriptors is None or len(frame_keypoints) < 4:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(template_descriptors, frame_descriptors, k=2)
    good_matches = [first for first, second in pairs if first.distance < ratio * second.distance]
    if len(good_matches) < minimum_inliers:
        return None

    template_points = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches])
    frame_points = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches])
    homography, inlier_mask = cv2.findHomography(
        template_points, frame_points, cv2.RANSAC, 5.0
    )
    if homography is None or inlier_mask is None:
        return None
    if int(inlier_mask.sum()) < minimum_inliers:
        return None

    return homography, inlier_mask


def display_corners(template: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Project the template's four corners into the camera frame."""
    height, width = template.shape[:2]
    corners = np.float32([[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]])
    return cv2.perspectiveTransform(corners, homography)[0]


def rectify_display(frame: np.ndarray, homography: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Warp the detected display into the template's fixed pixel coordinate system."""
    height, width = template.shape[:2]
    inverse = np.linalg.inv(homography)
    return cv2.warpPerspective(frame, inverse, (width, height))


def process_image(
    image: np.ndarray,
    template: np.ndarray,
    template_keypoints: list[cv2.KeyPoint],
    template_descriptors: np.ndarray,
    detector: cv2.SIFT,
    minimum_inliers: int = 8,
    ratio: float = 0.7,
) -> DisplayResult:
    """Detect and rectify the display in one image."""
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detection = find_display_homography(
        template_keypoints,
        template_descriptors,
        gray,
        detector,
        minimum_inliers,
        ratio,
    )
    if detection is None:
        return DisplayResult(np.zeros_like(template), None, None, 0)

    homography, inlier_mask = detection
    return DisplayResult(
        rectified=rectify_display(image, homography, template),
        homography=homography,
        corners=display_corners(template, homography),
        inlier_count=int(inlier_mask.sum()),
    )
