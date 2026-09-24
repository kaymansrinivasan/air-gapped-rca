"""Deterministic first-attempt Product_A wafer-sort log generator."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path


VERSION = "product-a-log-generator-v1"
ROOT = Path(__file__).resolve().parent
FAILURES = ("CONTINUITY_FAIL", "IDD_STATIC_FAIL", "SCAN_FAIL")
TEST_NAMES = {100: "Open/Short-", 210: "IDD_Static", 606: "SCAN Test"}
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def stable_json(obj: object, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_number(seed: int, *parts: object) -> int:
    label = "|".join(str(x) for x in (VERSION, seed, *parts)).encode("ascii")
    return int.from_bytes(hashlib.sha256(label).digest()[:8], "big")


def format_time(value: datetime) -> str:
    return (f"{WEEKDAYS[value.weekday()]} {MONTHS[value.month - 1]} "
            f"{value.day:2d} {value:%H:%M:%S} {value.year}")


def checked_inputs(config: dict, template: dict) -> tuple[list[str], list[dict]]:
    lots = config["lots"]
    if lots != [f"L{i:02d}" for i in range(1, 6)]:
        raise ValueError("This version requires lots L01–L05")
    if config["product"] != "Product_A" or config["wafers_per_lot"] != 7:
        raise ValueError("This version requires Product_A and seven wafers per lot")
    if config["devices_per_wafer"] != 112:
        raise ValueError("This coordinate template requires 112 devices per wafer")
    quota = config["outcome_quotas"]
    if set(quota) != {"PASS", *FAILURES} or any(type(v) is not int or v < 0 for v in quota.values()):
        raise ValueError("Outcome quotas must be nonnegative integers for all four outcomes")
    if sum(quota.values()) != 5 * 7 * 112:
        raise ValueError("Outcome quotas must sum to 3,920")
    if any(quota[k] < 6 for k in FAILURES):
        raise ValueError("Each failure quota must permit the two three-device clusters")
    if config["outcome_bins"] != {"PASS": 1, "CONTINUITY_FAIL": 4,
                                   "IDD_STATIC_FAIL": 3, "SCAN_FAIL": 2}:
        raise ValueError("Prototype bin mapping must remain 1/4/3/2")
    if config["idd_static"] != {"lower_mA": "35.00", "upper_mA": "55.00",
                                 "strict_limits": True, "display_decimals": 2}:
        raise ValueError("This version uses the documented prototype IDD limits")
    if config["scan_vector_min"] < 0 or config["scan_vector_max"] < config["scan_vector_min"]:
        raise ValueError("Invalid scan vector range")
    start = datetime.fromisoformat(config["start_time"])
    if start.tzinfo is None or start.utcoffset() != timedelta(hours=8):
        raise ValueError("The fixed start time must have a +08:00 offset")
    if config["timezone"] != "Asia/Kuala_Lumpur":
        raise ValueError("The timezone label must match the documented +08:00 clock")
    pin_map = config["pin_map"]
    if len(pin_map) < 3 or len(set(pin_map.values())) != len(pin_map):
        raise ValueError("Pin map needs distinct pin numbers")
    coordinates = template["coordinates"]
    if len(coordinates) != 112 or [c["device"] for c in coordinates] != [f"D{i:03d}" for i in range(1, 113)]:
        raise ValueError("Coordinate template must map D001–D112 in order")
    if len({(c["x"], c["y"]) for c in coordinates}) != 112:
        raise ValueError("Coordinate pairs must be unique")
    if template["source_sha256"] != "50dd35d58a93ff75404743a49789115b7180808b870cab4d6223d56430c81928":
        raise ValueError("Coordinate template is not tied to the checked A595 reference")
    return lots, coordinates


def all_slots(config: dict, coordinates: list[dict]) -> list[dict]:
    slots = []
    for lot in config["lots"]:
        for wafer_number in range(1, config["wafers_per_lot"] + 1):
            wafer = f"W{wafer_number:02d}"
            for item in coordinates:
                device = item["device"]
                slots.append({"lot": lot, "wafer": wafer, "device": device,
                              "coordinate": [item["x"], item["y"]],
                              "dut_id": f'{config["product"]}-{lot}-{wafer}-{device}'})
    return slots


def assign_outcomes(config: dict, coordinates: list[dict], slots: list[dict]) -> tuple[dict[str, str], list[dict]]:
    """Reserve explicit adjacent triples, then fill exact quotas by stable hash rank."""
    seed = config["seed"]
    lookup = {(c["x"], c["y"]): c["device"] for c in coordinates}
    triples = []
    for x, y in lookup:
        if (x + 1, y) in lookup and (x + 2, y) in lookup:
            triples.append([lookup[(x + j, y)] for j in range(3)])
    if not triples:
        raise ValueError("Coordinate template has no adjacent three-device groups")
    wafer_keys = [(lot, f"W{i:02d}") for lot in config["lots"]
                  for i in range(1, config["wafers_per_lot"] + 1)]
    outcome = {}
    clusters = []
    used_wafers = set()
    for category in FAILURES:
        for group_number in range(2):
            candidates = sorted((key for key in wafer_keys if key not in used_wafers),
                                key=lambda key: hash_number(seed, "cluster-wafer", category, group_number, *key))
            lot, wafer = candidates[0]
            group = min(triples, key=lambda ds: hash_number(seed, "cluster-group", category, lot, wafer, *ds))
            for device in group:
                outcome[f'{config["product"]}-{lot}-{wafer}-{device}'] = category
            clusters.append({"outcome": category, "lot": lot, "wafer": wafer, "devices": group})
            used_wafers.add((lot, wafer))
    for category in FAILURES:
        needed = config["outcome_quotas"][category] - sum(v == category for v in outcome.values())
        available = (slot for slot in slots if slot["dut_id"] not in outcome)
        # The wafer term gives moderate wafer-level variation without collecting a category in one lot.
        ranked = sorted(available, key=lambda slot: (
            hash_number(seed, "dut", category, slot["dut_id"]) / 2**64
            - 0.16 * hash_number(seed, "wafer", category, slot["lot"], slot["wafer"]) / 2**64,
            slot["dut_id"]))
        for slot in ranked[:needed]:
            outcome[slot["dut_id"]] = category
    for slot in slots:
        outcome.setdefault(slot["dut_id"], "PASS")
    if Counter(outcome.values()) != Counter(config["outcome_quotas"]):
        raise AssertionError("Exact global quotas were not achieved")
    if any(not any(outcome[slot["dut_id"]] == category and slot["lot"] == lot for slot in slots)
           for category in FAILURES for lot in config["lots"]):
        raise AssertionError("A failure category did not reach every lot")
    return outcome, clusters


def failing_pins(config: dict, seed: int, dut_id: str, category: str) -> list[dict]:
    items = sorted(config["pin_map"].items(),
                   key=lambda item: hash_number(seed, "pin", category, dut_id, item[0]))
    count = 1 + hash_number(seed, "pin-count", category, dut_id) % 3
    return [{"name": name, "number": number} for name, number in items[:count]]


def idd_value(seed: int, dut_id: str, fail: bool) -> str:
    if not fail:
        cents = 3800 + hash_number(seed, "idd-pass", dut_id) % 1401  # 38.00–52.00 mA
    elif hash_number(seed, "idd-side", dut_id) % 2 == 0:
        cents = 2700 + hash_number(seed, "idd-low", dut_id) % 601  # 27.00–33.00 mA
    else:
        cents = 5700 + hash_number(seed, "idd-high", dut_id) % 1301  # 57.00–70.00 mA
    return f"{cents // 100}.{cents % 100:02d}"


def make_tests(config: dict, slot: dict, category: str) -> list[dict]:
    seed, dut_id = config["seed"], slot["dut_id"]
    continuity_fail = category == "CONTINUITY_FAIL"
    pins = failing_pins(config, seed, dut_id, "continuity") if continuity_fail else []
    tests = [{"number": 100, "name": TEST_NAMES[100],
              "status": "FAIL" if continuity_fail else "PASS", "failing_pins": pins,
              "halt_vector": 0 if continuity_fail else None,
              "halt_cycle": 0 if continuity_fail else None}]
    if continuity_fail:
        return tests
    idd_fail = category == "IDD_STATIC_FAIL"
    tests.append({"number": 210, "name": TEST_NAMES[210],
                  "status": "FAIL" if idd_fail else "PASS",
                  "lower_mA": config["idd_static"]["lower_mA"],
                  "current_mA": idd_value(seed, dut_id, idd_fail),
                  "upper_mA": config["idd_static"]["upper_mA"]})
    if idd_fail:
        return tests
    scan_fail = category == "SCAN_FAIL"
    pins = failing_pins(config, seed, dut_id, "scan") if scan_fail else []
    vector = (config["scan_vector_min"] +
              hash_number(seed, "scan-vector", dut_id) %
              (config["scan_vector_max"] - config["scan_vector_min"] + 1)) if scan_fail else None
    tests.append({"number": 606, "name": TEST_NAMES[606],
                  "status": "FAIL" if scan_fail else "PASS", "pattern": config["scan_pattern"],
                  "failing_pins": pins, "halt_vector": vector, "halt_cycle": vector})
    return tests


def pin_line(pins: list[dict]) -> str:
    return "               " + "    ".join(f'{p["name"]:>6} : {p["number"]:<2}' for p in pins)


def test_lines(test: dict) -> list[str]:
    number, name, fail = test["number"], test["name"], test["status"] == "FAIL"
    if number == 210:
        marker = " (F)" if fail else "    "
        return [f'   210 IDD_Static         curr         {test["lower_mA"]} ma < '
                f'{test["current_mA"]:>8} ma{marker} <  {test["upper_mA"]} ma']
    if fail:
        first = (f"   {number:3d} {name:<20} Halt Vector: {test['halt_vector']:>6}   "
                 f"Halt Cycle: {test['halt_cycle']:>6}   Failing Pins: {len(test['failing_pins']):>2} (F)")
    else:
        first = f"   {number:3d} {name:<20} Failing Pins:  0"
    lines = [first]
    if number == 606:
        lines.append("       Pattern: scan")
    if fail:
        lines.extend(["          Failed Pins:", pin_line(test["failing_pins"])])
    return lines


def create_dataset(config_path: Path, coordinate_path: Path, output: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    template = json.loads(coordinate_path.read_text(encoding="utf-8"))
    _, coordinates = checked_inputs(config, template)
    if output.exists() and any((output / name).exists() for name in
                               ("logs", "device_metadata.jsonl", "validation_report.json", "manifest.json")):
        raise FileExistsError("Output contains generated files; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    slots = all_slots(config, coordinates)
    outcomes, clusters = assign_outcomes(config, coordinates, slots)
    start = datetime.fromisoformat(config["start_time"])
    file_paths = []
    records = []
    current_wafer_start = start
    for lot in config["lots"]:
        for wafer_number in range(1, config["wafers_per_lot"] + 1):
            wafer = f"W{wafer_number:02d}"
            wafer_slots = [slot for slot in slots if slot["lot"] == lot and slot["wafer"] == wafer]
            lines = ["", f"Wafer: {wafer}   Start Time: {format_time(current_wafer_start)}", "", ""]
            for index, slot in enumerate(wafer_slots):
                moment = (current_wafer_start + timedelta(seconds=config["first_device_delay_seconds"]
                          + index * config["device_interval_seconds"]))
                category = outcomes[slot["dut_id"]]
                tests = make_tests(config, slot, category)
                record = {**slot, "product": config["product"], "origin": "GENERATED",
                          "outcome": category, "bin": config["outcome_bins"][category],
                          "device_time": moment.isoformat(), "tests": tests}
                records.append(record)
                lines.extend([f'Lot: {lot:<21} Tester: {config["tester"]:<16} Program: {config["program"]}',
                              f'Device: {index + 1:03d}      Station: {config["station"]}   '
                              f'Site: {config["site"]}    Date: {format_time(moment)}',
                              "", f'Sequencer:  {config["sequencer"]}', ""])
                for test in tests:
                    lines.extend(test_lines(test))
                    lines.append("")
                x, y = slot["coordinate"]
                lines.extend([f'Bin:  {record["bin"]}     Wafer Coordinates: ( {x} , {y})', "", ""])
            finish = (current_wafer_start + timedelta(seconds=config["first_device_delay_seconds"]
                      + len(wafer_slots) * config["device_interval_seconds"]))
            lines.extend([f'End of Wafer: {wafer}  Part Count: {len(wafer_slots)}   '
                          f'Finish Time: {format_time(finish)}', "", ""])
            relative = Path("logs") / lot / f"{wafer}.txt"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes("\n".join(lines).encode("utf-8"))
            file_paths.append(relative.as_posix())
            current_wafer_start = finish + timedelta(seconds=config["wafer_gap_seconds"])
    metadata = output / "device_metadata.jsonl"
    metadata.write_bytes(b"".join(stable_json(r) for r in records))
    file_paths.append("device_metadata.jsonl")

    # Validation reconstructs the records from the written text, before the manifest exists.
    from validate import validate_dataset
    report = validate_dataset(output, config_path, coordinate_path, check_manifest=False)
    file_paths.append("validation_report.json")
    expected_tests = {"100": len(slots),
                      "210": len(slots) - config["outcome_quotas"]["CONTINUITY_FAIL"],
                      "606": len(slots) - config["outcome_quotas"]["CONTINUITY_FAIL"]
                      - config["outcome_quotas"]["IDD_STATIC_FAIL"]}
    expected_tests["total"] = sum(expected_tests.values())
    manifest = {
        "schema_version": "product-a-log-manifest-v1",
        "origin": "GENERATED_TEST_RESULTS",
        "generator_version": VERSION,
        "seed": config["seed"],
        "start_time": config["start_time"],
        "timezone": config["timezone"],
        "format_reference": {"path": template["source_file"], "sha256": template["source_sha256"],
                             "role": "format, coordinate layout and selected example test definitions only"},
        "generator_sha256": hash_file(Path(__file__)),
        "config_sha256": hash_file(config_path),
        "coordinate_template_sha256": hash_file(coordinate_path),
        "totals": {"dut_count": len(slots), "executed_tests": expected_tests,
                   "bins": {str(config["outcome_bins"][k]): v
                            for k, v in config["outcome_quotas"].items()}},
        "generated_failure_clusters": clusters,
        "files": [{"path": rel, "sha256": hash_file(output / rel)} for rel in sorted(file_paths)],
    }
    (output / "manifest.json").write_bytes(stable_json(manifest, pretty=True))
    validate_dataset(output, config_path, coordinate_path, check_manifest=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--coordinates", type=Path, default=ROOT / "coordinate_template.json")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    report = create_dataset(args.config, args.coordinates, args.output)
    print(json.dumps(report["overall"], sort_keys=True))


if __name__ == "__main__":
    main()
