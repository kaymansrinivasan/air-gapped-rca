"""Focused independent read-back tests for Product_A log evidence."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate import ValidationError, validate_dataset  # noqa: E402


CONFIG = {
    "product": "Product_A", "lots": ["L01"], "wafers_per_lot": 1,
    "devices_per_wafer": 4, "seed": 5952026,
    "start_time": "2026-01-01T08:00:00+08:00",
    "timezone": "Asia/Kuala_Lumpur", "tester": "ate-01",
    "program": "product_a_ws", "sequencer": "product_a_ws_seq",
    "station": 1, "site": 0,
    "outcome_quotas": {
        "PASS": 1, "CONTINUITY_FAIL": 1, "IDD_STATIC_FAIL": 1, "SCAN_FAIL": 1,
    },
    "outcome_bins": {"PASS": 1, "CONTINUITY_FAIL": 4, "IDD_STATIC_FAIL": 3, "SCAN_FAIL": 2},
    "idd_static": {"lower_mA": "35.00", "upper_mA": "55.00", "strict_limits": True},
    "pin_map": {"AMSDSM": 40, "TSTEN": 37, "TSTIN": 75},
    "scan_vector_min": 1, "scan_vector_max": 8191,
}


def _pin_test(number: int, name: str, pins: list[dict] | None = None,
              vector: int | None = None) -> dict:
    pins = pins or []
    result = {
        "number": number, "name": name, "status": "FAIL" if pins else "PASS",
        "failing_pins": pins,
        "halt_vector": vector if pins else None,
        "halt_cycle": vector if pins else None,
    }
    if number == 606:
        result["pattern"] = "scan"
    return result


class DatasetFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        log_dir = self.root / "logs" / "L01"
        log_dir.mkdir(parents=True)
        (self.root / "config.json").write_text(json.dumps(CONFIG) + "\n", encoding="utf-8")
        coordinates = [{"device": f"D{number:03d}", "x": number, "y": 1}
                       for number in range(1, 5)]
        (self.root / "coordinate_template.json").write_text(
            json.dumps({"coordinates": coordinates, "source_file": "source/a595.txt",
                        "source_sha256": "1" * 64}) + "\n", encoding="utf-8"
        )
        common = "Lot: L01                  Tester: ate-01          Program: product_a_ws\n"
        sequencer = "Sequencer:  product_a_ws_seq\n\n"
        pass_100 = "   100 Open/Short-         Failing Pins:  0\n\n"
        pass_210 = "   210 IDD_Static         curr         35.00 ma <    43.46 ma     <  55.00 ma\n\n"
        pass_606 = "   606 SCAN Test           Failing Pins:  0\n       Pattern: scan\n"
        blocks = [
            ("001", "08:00:10", pass_100 + pass_210 + pass_606, 1),
            ("002", "08:00:13",
             "   100 Open/Short-         Halt Vector:  0   Halt Cycle:  0   Failing Pins:  2 (F)\n"
             "          Failed Pins:\n               AMSDSM : 40    TSTEN : 37\n", 4),
            ("003", "08:00:16", pass_100 +
             "   210 IDD_Static         curr         35.00 ma <    58.18 ma (F) <  55.00 ma\n", 3),
            ("004", "08:00:19", pass_100 + pass_210 +
             "   606 SCAN Test           Halt Vector:  100   Halt Cycle:  100   Failing Pins:  1 (F)\n"
             "       Pattern: scan\n          Failed Pins:\n               TSTIN : 75\n", 2),
        ]
        text = "Wafer: W01   Start Time: Thu Jan  1 08:00:00 2026\n\n"
        for number, clock, result_rows, bin_number in blocks:
            text += (common + f"Device: {number}        Station: 1   Site: 0    "
                    f"Date: Thu Jan  1 {clock} 2026\n\n" + sequencer + result_rows +
                    f"Bin:  {bin_number}     Wafer Coordinates: ( {int(number)} , 1)\n\n")
        text += "End of Wafer: W01  Part Count: 4   Finish Time: Thu Jan  1 08:00:22 2026\n"
        self.log_path = log_dir / "W01.txt"
        self.log_path.write_text(text, encoding="utf-8", newline="\n")
        pass_100_meta = _pin_test(100, "Open/Short-")
        pass_210_meta = {
            "number": 210, "name": "IDD_Static", "status": "PASS",
            "current_mA": "43.46", "lower_mA": "35.00", "upper_mA": "55.00",
        }
        self.metadata = [
            {"dut_id": "Product_A-L01-W01-D001", "product": "Product_A", "lot": "L01",
             "wafer": "W01", "device": "D001", "coordinate": [1, 1],
             "device_time": "2026-01-01T08:00:10+08:00", "bin": 1, "outcome": "PASS",
             "origin": "GENERATED", "tests": [pass_100_meta, pass_210_meta,
                                              _pin_test(606, "SCAN Test")]},
            {"dut_id": "Product_A-L01-W01-D002", "product": "Product_A", "lot": "L01",
             "wafer": "W01", "device": "D002", "coordinate": [2, 1],
             "device_time": "2026-01-01T08:00:13+08:00", "bin": 4,
             "outcome": "CONTINUITY_FAIL", "origin": "GENERATED",
             "tests": [_pin_test(100, "Open/Short-",
                                 [{"name": "AMSDSM", "number": 40}, {"name": "TSTEN", "number": 37}], 0)]},
            {"dut_id": "Product_A-L01-W01-D003", "product": "Product_A", "lot": "L01",
             "wafer": "W01", "device": "D003", "coordinate": [3, 1],
             "device_time": "2026-01-01T08:00:16+08:00", "bin": 3,
             "outcome": "IDD_STATIC_FAIL", "origin": "GENERATED",
             "tests": [pass_100_meta, {
                 "number": 210, "name": "IDD_Static", "status": "FAIL",
                 "current_mA": "58.18", "lower_mA": "35.00", "upper_mA": "55.00",
             }]},
            {"dut_id": "Product_A-L01-W01-D004", "product": "Product_A", "lot": "L01",
             "wafer": "W01", "device": "D004", "coordinate": [4, 1],
             "device_time": "2026-01-01T08:00:19+08:00", "bin": 2,
             "outcome": "SCAN_FAIL", "origin": "GENERATED",
             "tests": [pass_100_meta, pass_210_meta,
                       _pin_test(606, "SCAN Test", [{"name": "TSTIN", "number": 75}], 100)]},
        ]
        self.write_metadata()

    def write_metadata(self) -> None:
        (self.root / "device_metadata.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in self.metadata),
            encoding="utf-8", newline="\n",
        )

    def write_manifest(self, report: dict) -> None:
        paths = [self.log_path, self.root / "device_metadata.jsonl",
                 self.root / "validation_report.json"]
        files = [{"path": path.relative_to(self.root).as_posix(),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
        config_hash = hashlib.sha256((self.root / "config.json").read_bytes()).hexdigest()
        coord_hash = hashlib.sha256((self.root / "coordinate_template.json").read_bytes()).hexdigest()
        manifest = {
            "origin": "GENERATED_TEST_RESULTS", "seed": CONFIG["seed"],
            "start_time": CONFIG["start_time"], "timezone": CONFIG["timezone"],
            "generator_sha256": hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "generate.py").read_bytes()).hexdigest(),
            "format_reference": {"path": "source/a595.txt", "sha256": "1" * 64},
            "files": files, "config_sha256": config_hash,
            "coordinate_template_sha256": coord_hash,
            "totals": {
                "dut_count": report["overall"]["dut_count"],
                "bins": report["overall"]["bin_counts"],
                "executed_tests": report["overall"]["executed_tests"],
            },
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_reconstructs_four_paths_and_verifies_manifest(self) -> None:
        report = validate_dataset(self.root, check_manifest=False)
        self.assertEqual(report["overall"]["bin_counts"], {"1": 1, "2": 1, "3": 1, "4": 1})
        self.assertEqual(report["overall"]["executed_tests"],
                         {"100": 4, "210": 3, "606": 2, "total": 9})
        self.write_manifest(report)
        self.assertEqual(validate_dataset(self.root), report)

    def test_rejects_written_idd_contradiction(self) -> None:
        self.log_path.write_text(
            self.log_path.read_text(encoding="utf-8").replace("43.46 ma", "55.46 ma", 1),
            encoding="utf-8", newline="\n",
        )
        with self.assertRaisesRegex(ValidationError, "current, limits, and"):
            validate_dataset(self.root, check_manifest=False)

    def test_rejects_written_pin_count_mismatch(self) -> None:
        self.log_path.write_text(
            self.log_path.read_text(encoding="utf-8").replace("Failing Pins:  2 ", "Failing Pins:  1 "),
            encoding="utf-8", newline="\n",
        )
        with self.assertRaises(ValidationError):
            validate_dataset(self.root, check_manifest=False)

    def test_rejects_metadata_identity_drift(self) -> None:
        self.metadata[0]["coordinate"] = [2, 1]
        self.write_metadata()
        with self.assertRaisesRegex(ValidationError, "metadata coordinate differs"):
            validate_dataset(self.root, check_manifest=False)

    def test_rejects_manifest_hash_drift(self) -> None:
        report = validate_dataset(self.root, check_manifest=False)
        self.write_manifest(report)
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][0]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "hash mismatch"):
            validate_dataset(self.root)


if __name__ == "__main__":
    unittest.main()
