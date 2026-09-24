"""Independent, offline read-back validation for the Product_A wafer-sort logs.

The parser reads the generated text files again; it does not trust the generator's
in-memory DUT objects or the external metadata when deciding test outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    """A written dataset violates its stated format or evidence invariants."""


WAFER_RE = re.compile(r"Wafer:\s+(W\d+)\s+Start Time:\s+(.+)$")
LOT_RE = re.compile(r"Lot:\s+(L\d+)\s+Tester:\s+(\S+)\s+Program:\s+(\S+)$")
DEVICE_RE = re.compile(
    r"Device:\s+(\d+)\s+Station:\s+(\d+)\s+Site:\s+(\d+)\s+Date:\s+(.+)$"
)
SEQUENCER_RE = re.compile(r"Sequencer:\s+(\S+)$")
PIN_TEST_RE = re.compile(
    r"(100 Open/Short-|606 SCAN Test)\s+"
    r"(?:Halt Vector:\s+(\d+)\s+Halt Cycle:\s+(\d+)\s+)?"
    r"Failing Pins:\s+(\d+)(?:\s+(\(F\)))?$"
)
PIN_DETAIL_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*(\d+)")
IDD_RE = re.compile(
    r"210 IDD_Static\s+curr\s+(-?\d+\.\d{2})\s+ma\s+<\s+"
    r"(-?\d+\.\d{2})\s+ma\s+(?:(\(F\))\s+)?<\s+"
    r"(-?\d+\.\d{2})\s+ma$"
)
BIN_RE = re.compile(r"Bin:\s+(\d+)\s+Wafer Coordinates:\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$")
END_RE = re.compile(r"End of Wafer:\s+(W\d+)\s+Part Count:\s+(\d+)\s+Finish Time:\s+(.+)$")
DATE_FORMAT = "%a %b %d %H:%M:%S %Y"
TEST_NAMES = {100: "Open/Short-", 210: "IDD_Static", 606: "SCAN Test"}
OUTCOME_BY_FAILED_TEST = {100: "CONTINUITY_FAIL", 210: "IDD_STATIC_FAIL", 606: "SCAN_FAIL"}
FORBIDDEN_ORIGIN_LABELS = re.compile(r"\b(?:synthetic|simulated|prototype)\b", re.IGNORECASE)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValidationError(message)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Cannot read JSON {path}: {exc}") from exc


def _decimal(value: Any, context: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValidationError(f"Invalid decimal for {context}: {value!r}") from exc


def _date(value: str, context: str) -> datetime:
    try:
        return datetime.strptime(value.strip(), DATE_FORMAT)
    except ValueError as exc:
        raise ValidationError(f"Invalid tester timestamp at {context}: {value!r}") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _test_limit_pair(config: dict[str, Any]) -> tuple[Decimal, Decimal]:
    limits = config["idd_static"]
    if isinstance(limits, dict):
        lower = limits.get("lower_mA")
        upper = limits.get("upper_mA")
    else:
        lower, upper = limits
    _require(lower is not None and upper is not None, "Config lacks IDD_Static limits")
    return _decimal(lower, "lower IDD limit"), _decimal(upper, "upper IDD limit")


def _source_lines(path: Path) -> list[tuple[int, str]]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"Cannot read tester log {path}: {exc}") from exc
    _require(b"\r" not in raw, f"{path}: expected LF line endings")
    _require(not FORBIDDEN_ORIGIN_LABELS.search(text), f"{path}: origin label in raw log")
    return [(number, line.strip()) for number, line in enumerate(text.splitlines(), 1) if line.strip()]


def _parse_log(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    lines = _source_lines(path)
    _require(bool(lines), f"{path}: empty wafer log")
    index = 0

    def current() -> tuple[int, str]:
        _require(index < len(lines), f"{path}: unexpected end of log")
        return lines[index]

    def consume(pattern: re.Pattern[str], label: str) -> re.Match[str]:
        nonlocal index
        line_no, value = current()
        match = pattern.fullmatch(value)
        _require(match is not None, f"{path}:{line_no}: expected {label}, got {value!r}")
        index += 1
        return match

    wafer_match = consume(WAFER_RE, "wafer header")
    wafer, start_text = wafer_match.groups()
    start = _date(start_text, f"{path}:wafer start")
    _require(wafer == path.stem, f"{path}: wafer header does not match filename")
    lot = path.parent.name
    devices: list[dict[str, Any]] = []
    last_device_time: datetime | None = None
    while index < len(lines) and not lines[index][1].startswith("End of Wafer:"):
        lot_match = consume(LOT_RE, "lot/tester/program header")
        record_lot, tester, program = lot_match.groups()
        _require(record_lot == lot, f"{path}: lot header {record_lot} does not match directory {lot}")
        _require(tester == config.get("tester", "ate-01"), f"{path}: unexpected tester {tester}")
        _require(program == config.get("program", "product_a_ws"), f"{path}: unexpected program {program}")
        device_match = consume(DEVICE_RE, "device header")
        device_number, station, site, date_text = device_match.groups()
        _require(int(device_number) >= 1, f"{path}: invalid device number")
        _require(int(station) == int(config.get("station", 1)), f"{path}: unexpected station")
        _require(int(site) == int(config.get("site", 0)), f"{path}: unexpected site")
        device_time = _date(date_text, f"{path}:device {device_number}")
        _require(device_time > start, f"{path}: device timestamp must follow wafer start")
        if last_device_time is not None:
            _require(device_time >= last_device_time, f"{path}: device timestamps are not chronological")
        last_device_time = device_time
        sequencer = consume(SEQUENCER_RE, "sequencer").group(1)
        _require(sequencer == config.get("sequencer", "product_a_ws_seq"), f"{path}: unexpected sequencer")
        tests: list[dict[str, Any]] = []
        while index < len(lines):
            line_no, row = current()
            if row.startswith("Bin:"):
                break
            if row.startswith("100 ") or row.startswith("606 "):
                match = consume(PIN_TEST_RE, "pin-test row")
                label, vector, cycle, count_text, flag = match.groups()
                number = 100 if label.startswith("100 ") else 606
                count = int(count_text)
                failed = flag is not None
                _require(failed == (count > 0), f"{path}:{line_no}: pin count and (F) disagree")
                _require((vector is not None) == failed and (cycle is not None) == failed,
                         f"{path}:{line_no}: halt fields and status disagree")
                if number == 606:
                    pattern_no, pattern_line = current()
                    _require(pattern_line == "Pattern: scan", f"{path}:{pattern_no}: missing scan pattern")
                    index += 1
                pins: list[dict[str, Any]] = []
                if failed:
                    detail_no, detail_header = current()
                    _require(detail_header == "Failed Pins:", f"{path}:{detail_no}: missing Failed Pins section")
                    index += 1
                    while len(pins) < count:
                        detail_no, detail = current()
                        matches = list(PIN_DETAIL_RE.finditer(detail))
                        _require(bool(matches), f"{path}:{detail_no}: invalid pin detail")
                        last_end = 0
                        for pin_match in matches:
                            _require(not detail[last_end:pin_match.start()].strip(),
                                     f"{path}:{detail_no}: unexpected text in pin detail")
                            name, pin_number = pin_match.groups()
                            pins.append({"name": name, "number": int(pin_number)})
                            last_end = pin_match.end()
                        _require(not detail[last_end:].strip(),
                                 f"{path}:{detail_no}: unexpected text after pin detail")
                        index += 1
                _require(len(pins) == count, f"{path}:{line_no}: wrong number of listed pins")
                _require(len({p["name"] for p in pins}) == count, f"{path}:{line_no}: duplicate pin name")
                _require(len({p["number"] for p in pins}) == count, f"{path}:{line_no}: duplicate pin number")
                for pin in pins:
                    _require(config["pin_map"].get(pin["name"]) == pin["number"],
                             f"{path}:{line_no}: pin is inconsistent with config pin map")
                if number == 606 and failed:
                    minimum = int(config["scan_vector_min"])
                    maximum = int(config["scan_vector_max"])
                    _require(minimum <= int(vector) <= maximum and minimum <= int(cycle) <= maximum,
                             f"{path}:{line_no}: scan vector/cycle outside configured range")
                tests.append({
                    "number": number, "name": TEST_NAMES[number],
                    "status": "FAIL" if failed else "PASS",
                    "failing_pins": pins,
                    "halt_vector": int(vector) if vector is not None else None,
                    "halt_cycle": int(cycle) if cycle is not None else None,
                    **({"pattern": "scan"} if number == 606 else {}),
                })
            elif row.startswith("210 "):
                match = consume(IDD_RE, "IDD_Static row")
                lower_text, current_text, flag, upper_text = match.groups()
                lower, upper = _test_limit_pair(config)
                measured = _decimal(current_text, f"{path}:{line_no}:current")
                _require(_decimal(lower_text, "rendered lower limit") == lower and
                         _decimal(upper_text, "rendered upper limit") == upper,
                         f"{path}:{line_no}: limits disagree with config")
                expected_pass = lower < measured < upper
                _require((flag is None) == expected_pass,
                         f"{path}:{line_no}: rendered current, limits, and (F) disagree")
                tests.append({
                    "number": 210, "name": TEST_NAMES[210],
                    "status": "PASS" if expected_pass else "FAIL",
                    "current_mA": current_text, "lower_mA": lower_text, "upper_mA": upper_text,
                })
            else:
                raise ValidationError(f"{path}:{line_no}: unexpected row in DUT tests: {row!r}")
        bin_match = consume(BIN_RE, "final bin and coordinate")
        bin_number, x_text, y_text = bin_match.groups()
        observed_numbers = [test["number"] for test in tests]
        _require(bool(tests), f"{path}: device {device_number} has no tests")
        failed_tests = [test["number"] for test in tests if test["status"] == "FAIL"]
        _require(len(failed_tests) <= 1, f"{path}: device {device_number} has multiple failed tests")
        expected_numbers = [100, 210, 606]
        if failed_tests:
            expected_numbers = expected_numbers[:expected_numbers.index(failed_tests[0]) + 1]
        _require(observed_numbers == expected_numbers,
                 f"{path}: device {device_number} has incorrect test order or tests after first failure")
        outcome = OUTCOME_BY_FAILED_TEST[failed_tests[0]] if failed_tests else "PASS"
        expected_bin = int(config["outcome_bins"][outcome])
        _require(int(bin_number) == expected_bin,
                 f"{path}: device {device_number} bin {bin_number} contradicts {outcome}")
        device = f"D{int(device_number):03d}"
        devices.append({
            "dut_id": f"{config['product']}-{lot}-{wafer}-{device}",
            "product": config["product"], "lot": lot, "wafer": wafer, "device": device,
            "coordinate": [int(x_text), int(y_text)], "device_time": device_time,
            "bin": expected_bin, "outcome": outcome, "tests": tests,
        })
    end_match = consume(END_RE, "end-of-wafer line")
    end_wafer, count_text, finish_text = end_match.groups()
    finish = _date(finish_text, f"{path}:wafer finish")
    _require(end_wafer == wafer, f"{path}: end wafer ID does not match header")
    _require(int(count_text) == len(devices), f"{path}: part count contradicts device blocks")
    _require(last_device_time is not None and finish >= last_device_time,
             f"{path}: finish precedes device timestamps")
    if index != len(lines):
        raise ValidationError(f"{path}:{lines[index][0]}: unexpected content after wafer end")
    return {"lot": lot, "wafer": wafer, "start": start, "finish": finish, "devices": devices}


def _read_metadata(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"Cannot read metadata {path}: {exc}") from exc
    for line_number, row in enumerate(rows, 1):
        _require(bool(row.strip()), f"{path}:{line_number}: blank metadata row")
        try:
            item = json.loads(row)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        _require(isinstance(item, dict) and isinstance(item.get("dut_id"), str),
                 f"{path}:{line_number}: missing DUT identity")
        dut_id = item["dut_id"]
        _require(dut_id not in result, f"{path}:{line_number}: duplicate DUT {dut_id}")
        result[dut_id] = item
    return result


def _compare_metadata(parsed: dict[str, Any], metadata: dict[str, Any], offset: Any) -> None:
    dut_id = parsed["dut_id"]
    _require(metadata.get("origin") == "GENERATED", f"{dut_id}: metadata must identify generated origin")
    for key in ("dut_id", "product", "lot", "wafer", "device", "coordinate", "bin", "outcome"):
        _require(metadata.get(key) == parsed[key], f"{dut_id}: metadata {key} differs from written log")
    try:
        stamp = datetime.fromisoformat(metadata["device_time"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"{dut_id}: invalid metadata device_time") from exc
    _require(stamp.tzinfo is not None and stamp.utcoffset() == offset,
             f"{dut_id}: metadata timezone disagrees with config")
    _require(stamp.replace(tzinfo=None) == parsed["device_time"],
             f"{dut_id}: metadata device_time differs from written log")
    meta_tests = metadata.get("tests")
    _require(isinstance(meta_tests, list) and len(meta_tests) == len(parsed["tests"]),
             f"{dut_id}: metadata executed-test count differs from written log")
    for expected, actual in zip(parsed["tests"], meta_tests):
        _require(isinstance(actual, dict), f"{dut_id}: malformed metadata test")
        for key, value in expected.items():
            _require(actual.get(key) == value,
                     f"{dut_id}: metadata test {expected['number']} field {key} differs from written log")


def _check_manifest(dataset_dir: Path, manifest: dict[str, Any], report: dict[str, Any],
                    config_path: Path, coordinate_path: Path, expected_logs: list[Path]) -> None:
    config = _read_json(config_path)
    template = _read_json(coordinate_path)
    _require(manifest.get("origin") == "GENERATED_TEST_RESULTS", "Manifest lacks generated origin")
    _require(manifest.get("generator_sha256") == _sha256(Path(__file__).with_name("generate.py")),
             "Manifest generator hash mismatch")
    reference = manifest.get("format_reference")
    _require(isinstance(reference, dict) and
             reference.get("path") == template.get("source_file") and
             reference.get("sha256") == template.get("source_sha256"),
             "Manifest format reference differs from coordinate template")
    _require(manifest.get("seed") == config.get("seed") and
             manifest.get("start_time") == config.get("start_time") and
             manifest.get("timezone") == config.get("timezone"),
             "Manifest generation settings differ from config")
    file_entries = manifest.get("files")
    _require(isinstance(file_entries, list), "Manifest lacks files list")
    seen: set[str] = set()
    for entry in file_entries:
        _require(isinstance(entry, dict) and isinstance(entry.get("path"), str) and
                 isinstance(entry.get("sha256"), str), "Malformed manifest file entry")
        relative = Path(entry["path"])
        _require(not relative.is_absolute() and ".." not in relative.parts,
                 f"Unsafe manifest path {entry['path']!r}")
        path = (dataset_dir / relative).resolve()
        _require(path.is_relative_to(dataset_dir.resolve()), f"Manifest path escapes dataset: {relative}")
        _require(path.is_file(), f"Manifest file missing: {relative}")
        _require(entry["path"] not in seen, f"Duplicate manifest path: {relative}")
        seen.add(entry["path"])
        _require(_sha256(path) == entry["sha256"], f"Manifest hash mismatch: {relative}")
    required = {str(path.relative_to(dataset_dir)).replace("\\", "/") for path in expected_logs}
    required.update({"device_metadata.jsonl", "validation_report.json"})
    _require(required == seen,
             f"Manifest file set differs from dataset: missing {sorted(required - seen)}, extra {sorted(seen - required)}")
    _require(manifest.get("config_sha256") == _sha256(config_path), "Manifest config hash mismatch")
    _require(manifest.get("coordinate_template_sha256") == _sha256(coordinate_path),
             "Manifest coordinate template hash mismatch")
    totals = manifest.get("totals", {})
    overall = report["overall"]
    _require(totals.get("dut_count") == overall["dut_count"], "Manifest DUT total mismatch")
    _require(totals.get("bins") == overall["bin_counts"], "Manifest bin totals mismatch")
    _require(totals.get("executed_tests") == overall["executed_tests"],
             "Manifest executed-test totals mismatch")


def validate_dataset(dataset_dir: str | Path, config_path: str | Path | None = None,
                     coordinate_path: str | Path | None = None,
                     check_manifest: bool = True) -> dict[str, Any]:
    """Validate written logs and metadata, write/return a deterministic report.

    Use ``check_manifest=False`` during generation, before the manifest exists.
    With the default ``True``, the existing report and manifest are verified and
    neither file is changed.
    """
    dataset_dir = Path(dataset_dir).resolve()
    config_path = Path(config_path).resolve() if config_path else dataset_dir / "config.json"
    coordinate_path = Path(coordinate_path).resolve() if coordinate_path else dataset_dir / "coordinate_template.json"
    config = _read_json(config_path)
    template = _read_json(coordinate_path)
    _require(isinstance(config, dict), "Config must be a JSON object")
    _require(isinstance(config.get("product"), str), "Config lacks product")
    _require(isinstance(config.get("lots"), list), "Config lots must be a list")
    lots = config["lots"]
    wafers_per_lot = int(config["wafers_per_lot"])
    devices_per_wafer = int(config["devices_per_wafer"])
    _require(len(lots) == len(set(lots)) and lots, "Config lots are empty or duplicated")
    _require(wafers_per_lot > 0 and devices_per_wafer > 0, "Invalid configured dataset size")
    _require(int(config["scan_vector_min"]) >= 0 and
             int(config["scan_vector_max"]) >= int(config["scan_vector_min"]),
             "Invalid scan vector range")
    _require(len(set(config["pin_map"].values())) == len(config["pin_map"]),
             "Config pin numbers must be unique")
    template_rows = template["coordinates"] if isinstance(template, dict) else template
    _require(isinstance(template_rows, list) and len(template_rows) == devices_per_wafer,
             "Coordinate template has wrong device count")
    coordinate_by_device: dict[str, list[int]] = {}
    for entry in template_rows:
        _require(isinstance(entry, dict), "Malformed coordinate template entry")
        device = entry["device"]
        _require(device not in coordinate_by_device, f"Duplicate template device {device}")
        coordinate_by_device[device] = [int(entry["x"]), int(entry["y"])]
    expected_devices = {f"D{number:03d}" for number in range(1, devices_per_wafer + 1)}
    _require(set(coordinate_by_device) == expected_devices, "Coordinate template has missing or extra devices")
    _require(len({tuple(coord) for coord in coordinate_by_device.values()}) == devices_per_wafer,
             "Coordinate template contains duplicate positions")
    try:
        configured_start = datetime.fromisoformat(config["start_time"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError("Config start_time must be ISO 8601") from exc
    _require(configured_start.tzinfo is not None, "Config start_time must contain a timezone offset")
    _require(bool(config.get("timezone")), "Config must document timestamp timezone")
    offset = configured_start.utcoffset()
    metadata = _read_metadata(dataset_dir / "device_metadata.jsonl")
    expected_logs = [dataset_dir / "logs" / lot / f"W{number:02d}.txt"
                     for lot in lots for number in range(1, wafers_per_lot + 1)]
    actual_logs = set((dataset_dir / "logs").rglob("*.txt"))
    _require(actual_logs == set(expected_logs), "Log file set differs from configured lot/wafer matrix")
    all_ids: set[str] = set()
    overall_bins: Counter[str] = Counter()
    overall_outcomes: Counter[str] = Counter()
    overall_tests: Counter[str] = Counter()
    per_wafer: list[dict[str, Any]] = []
    previous_finish: datetime | None = None
    for path in expected_logs:
        wafer = _parse_log(path, config)
        _require(len(wafer["devices"]) == devices_per_wafer,
                 f"{path}: expected {devices_per_wafer} DUTs")
        _require({item["device"] for item in wafer["devices"]} == expected_devices,
                 f"{path}: missing, extra or duplicate DUT numbers")
        _require(len({tuple(item["coordinate"]) for item in wafer["devices"]}) == devices_per_wafer,
                 f"{path}: duplicate coordinates")
        if previous_finish is None:
            _require(wafer["start"] == configured_start.replace(tzinfo=None),
                     f"{path}: first wafer start differs from config")
        else:
            _require(wafer["start"] >= previous_finish, f"{path}: wafer time overlaps preceding wafer")
        previous_finish = wafer["finish"]
        bins: Counter[str] = Counter()
        outcomes: Counter[str] = Counter()
        tests: Counter[str] = Counter()
        for item in wafer["devices"]:
            dut_id = item["dut_id"]
            _require(dut_id not in all_ids, f"Duplicate DUT identity {dut_id}")
            all_ids.add(dut_id)
            _require(item["coordinate"] == coordinate_by_device[item["device"]],
                     f"{dut_id}: coordinate differs from template")
            _require(dut_id in metadata, f"{dut_id}: missing metadata row")
            _compare_metadata(item, metadata[dut_id], offset)
            bins[str(item["bin"])] += 1
            outcomes[item["outcome"]] += 1
            for test in item["tests"]:
                tests[str(test["number"])] += 1
        overall_bins.update(bins)
        overall_outcomes.update(outcomes)
        overall_tests.update(tests)
        per_wafer.append({
            "lot": wafer["lot"], "wafer": wafer["wafer"],
            "dut_count": len(wafer["devices"]),
            "bin_counts": {key: bins[key] for key in sorted(map(str, config["outcome_bins"].values()))},
            "outcome_counts": {key: outcomes[key] for key in sorted(config["outcome_bins"])},
            "executed_tests": {key: tests[key] for key in ("100", "210", "606")},
        })
    _require(set(metadata) == all_ids, "Metadata has extra DUTs absent from written logs")
    expected_count = len(lots) * wafers_per_lot * devices_per_wafer
    _require(len(all_ids) == expected_count, "Global DUT count mismatch")
    quotas = config["outcome_quotas"]
    _require({key: overall_outcomes[key] for key in quotas} == {key: int(value) for key, value in quotas.items()},
             "Written outcome counts differ from configured exact quotas")
    _require(sum(overall_outcomes.values()) == expected_count, "Outcome count does not match DUT count")
    _require(overall_tests["100"] == expected_count, "Continuity execution count mismatch")
    _require(overall_tests["210"] == expected_count - overall_outcomes["CONTINUITY_FAIL"],
             "IDD execution count mismatch")
    _require(overall_tests["606"] == expected_count - overall_outcomes["CONTINUITY_FAIL"] -
             overall_outcomes["IDD_STATIC_FAIL"], "Scan execution count mismatch")
    test_counts = {key: overall_tests[key] for key in ("100", "210", "606")}
    test_counts["total"] = sum(test_counts.values())
    report = {
        "schema_version": "1.0",
        "status": "passed",
        "overall": {
            "lot_count": len(lots), "wafer_count": len(expected_logs),
            "dut_count": expected_count,
            "bin_counts": {key: overall_bins[key] for key in sorted(map(str, config["outcome_bins"].values()))},
            "outcome_counts": {key: overall_outcomes[key] for key in sorted(config["outcome_bins"])},
            "executed_tests": test_counts,
        },
        "per_wafer": per_wafer,
    }
    report_path = dataset_dir / "validation_report.json"
    expected_bytes = _canonical_json(report)
    if check_manifest:
        _require(report_path.is_file(), "Validation report missing before manifest check")
        _require(report_path.read_bytes() == expected_bytes, "Validation report differs from independent read-back")
        _check_manifest(dataset_dir, _read_json(dataset_dir / "manifest.json"), report,
                        config_path, coordinate_path, expected_logs)
    else:
        report_path.write_bytes(expected_bytes)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--coordinates", type=Path)
    parser.add_argument("--skip-manifest", action="store_true", help="For generation before manifest creation")
    args = parser.parse_args()
    try:
        report = validate_dataset(args.dataset, args.config, args.coordinates,
                                  check_manifest=not args.skip_manifest)
    except (ValidationError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"Validation failed: {exc}\n")
    totals = report["overall"]
    print(f"Validated {totals['dut_count']} DUTs across {totals['wafer_count']} wafers; "
          f"{totals['executed_tests']['total']} executed tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
