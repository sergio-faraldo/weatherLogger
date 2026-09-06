"""Small synthetic checks for the seven-segment and compass recognizers."""

import unittest
from datetime import datetime
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from display_masks import MaskBox, digit_zone_pixels
from display_analysis import gate_image
from display_readings import decode_digit, detect_compass_marker, format_readings, recognize_sectors
from main import (
    append_csv_reading,
    collect_recording,
    find_first_usb_mount,
    most_common_measurements,
    read_average_frame,
)


class DisplayReadingsTest(unittest.TestCase):
    def test_most_common_measurement_is_selected_per_variable(self) -> None:
        readings = [
            {
                "S1": {"measurement": {"temperature_c": 10, "humidity_percent": 40}},
                "S2": {"measurement": {"wind_speed_kmh": 5, "wind_direction_degrees": 90}},
            },
            {
                "S1": {"measurement": {"temperature_c": 10, "humidity_percent": 41}},
                "S2": {"measurement": {"wind_speed_kmh": 5, "wind_direction_degrees": 90}},
            },
            {
                "S1": {"measurement": {"temperature_c": 11, "humidity_percent": 41}},
                "S2": {"measurement": {"wind_speed_kmh": 5, "wind_direction_degrees": 90}},
            },
            {"S2": {"measurement": {"wind_speed_kmh": 5, "wind_direction_degrees": 90}}},
        ]

        measurements = most_common_measurements(readings)

        self.assertEqual(measurements["outside_temperature_c"], 10)
        self.assertEqual(measurements["outside_humidity_percent"], 41)
        self.assertEqual(measurements["wind_speed_kmh"], 5)
        self.assertIsNone(measurements["inside_temperature_c"])

    def test_appends_timestamped_csv_reading(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "readings.csv"
            append_csv_reading(path, {"outside_temperature_c": 12.3}, datetime(2026, 9, 6, 7, 4))

            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines()[0],
                "timestamp,outside_temperature_c,outside_humidity_percent,wind_speed_kmh,"
                "wind_direction_degrees,inside_temperature_c,inside_humidity_percent,"
                "hourly_rainfall_mm,pressure_kpa",
            )
            self.assertIn("2026-09-06T07:04:00,12.3", path.read_text(encoding="utf-8"))

    def test_rotates_previous_day_readings_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "readings.csv"
            path.write_text("old,data\n", encoding="utf-8")
            created = datetime(2026, 9, 5, 23, 59).timestamp()

            with patch(
                "main.Path.stat",
                return_value=SimpleNamespace(st_ctime=created, st_size=path.stat().st_size),
            ):
                append_csv_reading(path, {"outside_temperature_c": 12.3}, datetime(2026, 9, 6, 7, 4))

            archive = Path(directory) / "older-data" / "readings-2026-09-05.gz"
            self.assertTrue(archive.exists())
            with gzip.open(archive, "rt", encoding="utf-8") as file:
                self.assertEqual(file.read(), "old,data\n")
            self.assertIn("2026-09-06T07:04:00,12.3", path.read_text(encoding="utf-8"))

    def test_reads_and_averages_requested_frames(self) -> None:
        frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (10, 20, 30)]

        class Camera:
            def __init__(self) -> None:
                self.index = 0

            def read(self) -> tuple[bool, np.ndarray]:
                frame = frames[self.index]
                self.index += 1
                return True, frame

        averaged = read_average_frame(Camera(), 3)

        np.testing.assert_array_equal(averaged, np.full((2, 2, 3), 20, dtype=np.uint8))

    def test_record_loop_writes_each_completed_batch(self) -> None:
        reading = {"S1": {"measurement": {"temperature_c": 12.3}}}

        with TemporaryDirectory() as directory:
            path = Path(directory) / "readings.csv"
            collected: list[dict[str, object]] = []

            for _ in range(22):
                should_stop = collect_recording(reading, collected, "record-loop", path)
                self.assertFalse(should_stop)

            self.assertEqual(len(collected), 0)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 3)

    def test_recording_writes_to_usb_copy(self) -> None:
        reading = {"S1": {"measurement": {"temperature_c": 12.3}}}

        with TemporaryDirectory() as directory:
            output = Path(directory) / "readings.csv"
            usb_output = Path(directory) / "usb" / "readings.csv"
            usb_output.parent.mkdir()
            collected: list[dict[str, object]] = []

            for _ in range(11):
                should_stop = collect_recording(reading, collected, "record", output, usb_output)

            self.assertTrue(should_stop)

            self.assertEqual(output.read_text(encoding="utf-8"), usb_output.read_text(encoding="utf-8"))

    def test_finds_first_mounted_usb_partition(self) -> None:
        lsblk_output = {
            "blockdevices": [
                {"name": "/dev/sda", "tran": "sata", "mountpoint": None, "type": "disk"},
                {
                    "name": "/dev/sdb",
                    "tran": "usb",
                    "mountpoint": None,
                    "type": "disk",
                    "children": [
                        {
                            "name": "/dev/sdb1",
                            "tran": None,
                            "mountpoint": "/media/user/USB",
                            "type": "part",
                        }
                    ],
                },
            ]
        }

        with patch("main.sys.platform", "linux"), patch(
            "main.subprocess.run",
            return_value=SimpleNamespace(stdout=json.dumps(lsblk_output)),
        ), patch("main.Path.is_dir", return_value=True):
            self.assertEqual(find_first_usb_mount(), Path("/media/user/USB"))

    def test_usb_mount_is_unavailable_on_non_linux(self) -> None:
        with patch("main.sys.platform", "win32"):
            self.assertIsNone(find_first_usb_mount())

    def test_gate_image_clips_to_black_and_white(self) -> None:
        image = np.array([[0, 99, 100, 255]], dtype=np.uint8)
        gated = gate_image(image, 100)
        np.testing.assert_array_equal(gated, [[0, 0, 255, 255]])
    def test_decodes_eight(self) -> None:
        image = np.zeros((100, 60), dtype=np.uint8)
        box = MaskBox(0, 0, 1, 1)
        for left, top, right, bottom in digit_zone_pixels(image, box).values():
            image[top:bottom, left:right] = 230
        reading = decode_digit(image, box)
        self.assertEqual(reading.character, "8")

    def test_segment_is_on_only_above_half_gray(self) -> None:
        image = np.zeros((100, 60), dtype=np.uint8)
        box = MaskBox(0, 0, 1, 1)
        zones = digit_zone_pixels(image, box)
        for left, top, right, bottom in zones.values():
            image[top:bottom, left:right] = 128

        reading = decode_digit(image, box)

        self.assertEqual(reading.character, "8")
        self.assertEqual(reading.segments, frozenset(zones))

    def test_no_segments_is_interpreted_as_zero(self) -> None:
        image = np.zeros((100, 60), dtype=np.uint8)
        reading = decode_digit(image, MaskBox(0, 0, 1, 1))

        self.assertEqual(reading.character, "0")
        self.assertEqual(reading.confidence, 1.0)
        self.assertEqual(reading.segments, frozenset())

    def test_formats_measurements_from_sector_digits(self) -> None:
        readings = {
            "S1": {"text": "12345", "measurement": {"temperature_c": -12.3, "humidity_percent": 45}},
            "S2": {"text": "123", "measurement": {"wind_speed_kmh": 12.3, "wind_direction_degrees": 270.0}},
            "S3": {"text": "23456", "measurement": {"temperature_c": 23.4, "humidity_percent": 56}},
            "S4": {"text": "1234", "measurement": {"hourly_rainfall_mm": 12.3}},
            "S6": {"text": "1013", "measurement": {"pressure_kpa": 1013}},
        }

        self.assertEqual(
            format_readings(readings),
            [
                "S1 outside: -12.3 °C, humidity 45%",
                "S2 wind: 12.3 km/h, direction 270°",
                "S3 inside: 23.4 °C, humidity 56%",
                "S4 hourly rainfall: 12.3 mm",
                "S6 pressure: 1013 kPa",
            ],
        )

    def test_finds_north_marker(self) -> None:
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(image, (50, 50), (50, 15), 240, 4)
        reading = detect_compass_marker(image, MaskBox(0, 0, 1, 1))
        self.assertIsNotNone(reading.angle_degrees)
        self.assertEqual(reading.sector_index, 0)
        self.assertTrue(reading.angle_degrees < 10 or reading.angle_degrees > 350)

    def test_north_adjacent_marker_stays_in_centered_north_sector(self) -> None:
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(image, (50, 50), (56, 16), 240, 4)
        reading = detect_compass_marker(image, MaskBox(0, 0, 1, 1))
        self.assertEqual(reading.sector_index, 0)

    def test_compass_uses_whitest_average_ring_sector(self) -> None:
        image = np.zeros((100, 100), dtype=np.uint8)
        yy, xx = np.indices(image.shape, dtype=np.float32)
        angles = (np.arctan2(xx - 50, -(yy - 50)) + 2 * np.pi) % (2 * np.pi)
        sector_width = 2 * np.pi / 16
        sector = (angles + sector_width / 2) // sector_width
        ring = ((xx - 50) ** 2 + (yy - 50) ** 2 >= 25 ** 2) & ((xx - 50) ** 2 + (yy - 50) ** 2 <= 45 ** 2)
        image[ring & (sector == 5)] = 220
        image[ring & (sector == 6)] = 140

        reading = detect_compass_marker(image, MaskBox(0, 0, 1, 1))

        self.assertEqual(reading.sector_index, 5)


if __name__ == "__main__":
    unittest.main()