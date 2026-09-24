"""End-to-end reproduction and coordinate-cluster check for the full dataset."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from generate import create_dataset  # noqa: E402


class FullGenerationTest(unittest.TestCase):
    def test_full_dataset_reproduces_and_declared_clusters_are_real(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "second"
            config_path = PACKAGE / "config.json"
            template_path = PACKAGE / "coordinate_template.json"
            report_a = create_dataset(config_path, template_path, first)
            report_b = create_dataset(config_path, template_path, second)
            self.assertEqual(report_a, report_b)

            files_a = {path.relative_to(first).as_posix(): path.read_bytes()
                       for path in first.rglob("*") if path.is_file()}
            files_b = {path.relative_to(second).as_posix(): path.read_bytes()
                       for path in second.rglob("*") if path.is_file()}
            self.assertEqual(files_a, files_b)
            self.assertEqual(len(files_a), 38)
            self.assertEqual(len(list((first / "logs").rglob("*.txt"))), 35)
            self.assertEqual(report_a["overall"]["dut_count"], 3920)
            self.assertEqual(report_a["overall"]["executed_tests"],
                             {"100": 3920, "210": 3822, "606": 3724, "total": 11466})

            manifest = json.loads(files_a["manifest.json"])
            self.assertEqual(len(manifest["generated_failure_clusters"]), 6)
            template = json.loads(template_path.read_text(encoding="utf-8"))
            positions = {item["device"]: (item["x"], item["y"])
                         for item in template["coordinates"]}
            metadata = {item["dut_id"]: item for item in (
                json.loads(row) for row in files_a["device_metadata.jsonl"].splitlines()
            )}
            for cluster in manifest["generated_failure_clusters"]:
                devices = cluster["devices"]
                self.assertEqual(len(devices), 3)
                self.assertEqual(len(set(devices)), 3)
                coords = [positions[device] for device in devices]
                self.assertTrue(all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                                    for a, b in zip(coords, coords[1:])), cluster)
                for device in devices:
                    dut_id = f"Product_A-{cluster['lot']}-{cluster['wafer']}-{device}"
                    row = metadata[dut_id]
                    self.assertEqual(row["outcome"], cluster["outcome"])
                    self.assertEqual(row["tests"][-1]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
