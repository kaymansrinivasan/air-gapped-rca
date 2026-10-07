"""Deterministic source and symptom checks for structured RCA suggestions.

This verifier can reject mismatched sources, pins and current direction. It
cannot establish a physical cause or prove semantic entailment from a log.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.rca_local.core import EvidenceError, require


def _source_record(root: Path, item: dict, match: dict) -> dict:
    source = item["citation"]
    path = (root / source["file"]).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(),
            "Citation file is outside the evidence root or missing.")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == source["sha256"],
            "Cited source changed during verification.")
    lines = source["lines"]
    require(type(lines["start"]) is int and type(lines["end"]) is int
            and 1 <= lines["start"] <= lines["end"] <= len(raw.splitlines()),
            "Citation line range is invalid.")
    record = json.loads(raw)
    require(source == match["sources"]["investigation"],
            "Citation does not identify the matched investigation.")
    require(record.get("record_id") == source["record_id"]
            and record.get("stage") == "investigation"
            and record.get("role") == "historical"
            and record.get("case_id") == item["case_id"]
            and record.get("dut_id") == item["historical_dut"],
            "Citation does not identify the claimed historical record.")
    allowed = (record["possible_causes"] if item["kind"] == "cause"
               else [check["check"] for check in record["checks"]])
    require(item["text"] in allowed, "Suggested text is absent from its cited source.")
    return record


def _direction(current: dict) -> str | None:
    """Read a high/low IDD observation; questions are never evidence."""
    text = current.get("observation_text") or current.get("retrieval_text", "")
    upper = bool(re.search(r"\b(?:exceed(?:s|ed)?|above|over)\b.{0,35}\b(?:upper|high|maximum|limit)\b"
                           r"|\b(?:upper|high|maximum)\b.{0,35}\b(?:exceed|above|fail)",
                           text, re.I))
    lower = bool(re.search(r"\b(?:below|under)\b.{0,35}\b(?:lower|low|minimum|limit)\b"
                           r"|\b(?:lower|low|minimum)\b.{0,35}\b(?:below|under|fail)",
                           text, re.I))
    measured = re.search(r"measured (?:static )?current\s*:\s*(\d+(?:\.\d+)?)\s*mA", text, re.I)
    limits = re.search(r"(?:allowed range|limits?)\s*:\s*(\d+(?:\.\d+)?)\s*[–-]\s*"
                       r"(\d+(?:\.\d+)?)\s*mA", text, re.I)
    tester_line = re.search(
        r"\b210\s+IDD_Static\b[^\n]*?(\d+(?:\.\d+)?)\s*ma\s*<\s*"
        r"(\d+(?:\.\d+)?)\s*ma\s*\(F\)\s*<\s*(\d+(?:\.\d+)?)\s*ma",
        text, re.I)
    if tester_line:
        low, value, high = map(float, tester_line.groups())
        require(low < high, "Current limits are inconsistent.")
        return "high" if value > high else "low" if value < low else None
    if measured and limits:
        value, low, high = float(measured[1]), float(limits[1]), float(limits[2])
        require(low < high, "Current limits are inconsistent.")
        numeric = "high" if value > high else "low" if value < low else None
        require(not numeric or (not upper or numeric == "high") and
                (not lower or numeric == "low"), "Current direction conflicts with measurements.")
        return numeric
    if upper and not lower:
        return "high"
    if lower and not upper:
        return "low"
    return None


def _historical_direction(match: dict) -> str | None:
    text = " ".join(match["records"]["investigation"]["possible_causes"]).lower()
    high = "high" in text or "excessive static current" in text
    low = "low" in text
    return "high" if high and not low else "low" if low and not high else None


def compatible_matches(bundle: dict) -> list[dict]:
    """Limit model choices to histories with the observed IDD direction.

    The original retrieval bundle remains intact for audit and display. The
    verifier below independently repeats this check on the model's selections.
    An unknown current direction does not license a high or low hypothesis.
    """
    current = bundle["current"]
    if current["test_result"]["test_number"] != 210:
        return bundle["matches"]
    direction = _direction(current)
    return [match for match in bundle["matches"]
            if direction is not None and _historical_direction(match) == direction]


def verify_selection(root: Path, bundle: dict, status: str, selected: dict):
    """Return status, supported selections, and a per-item rejection audit.

    Bad or changed citations raise EvidenceError. Inapplicable historical
    suggestions are removed. With no remaining cause and check, refuse.
    """
    root = Path(root)
    current = bundle["current"]
    require(current.get("role") == "query_only" and current.get("index_as_history") is False,
            "Only a query-only incident can be verified.")
    number = current["test_result"]["test_number"]
    pins = {p["name"].upper() for p in current["test_result"].get("failing_pins", [])}
    direction = _direction(current) if number == 210 else None
    matches = {m["records"]["observation"]["case_id"]: m for m in bundle["matches"]}
    kept = {"cause_ids": [], "check_ids": []}
    rejected = []
    for field in kept:
        for item in selected[field]:
            require(item["kind"] == ("cause" if field == "cause_ids" else "check"),
                    "Selection type changed.")
            match = matches.get(item["case_id"])
            require(match is not None, "Selection is not from a retrieved case.")
            record = _source_record(root, item, match)
            observation = match["records"]["observation"]
            historical_test = observation.get("original_chunk", {}).get("failed_test")
            require(historical_test == number and record["case_id"] == observation["case_id"],
                    "Citation belongs to a different failed test or case.")
            reason = None
            if number == 210:
                expected = _historical_direction(match)
                if not direction or expected != direction:
                    reason = "Current high/low symptom is unknown or differs from historical case."
            if field == "cause_ids" and not reason:
                old_text = observation.get("retrieval_text", "")
                pin_section = old_text.split("Failed Pins:", 1)[1].split("\n\n", 1)[0] if "Failed Pins:" in old_text else ""
                historical_pins = {p.upper() for p in re.findall(
                    r"([A-Za-z0-9_+-]+)\s*:\s*\d+", pin_section)}
                mentioned = {p for p in historical_pins if re.search(
                    r"(?<![A-Za-z0-9_])" + re.escape(p) + r"(?![A-Za-z0-9_])",
                    item["text"], re.I)}
                if mentioned and not mentioned <= pins:
                    reason = "Pin-specific cause does not match the current observed pins."
            if reason:
                rejected.append({"id": item["id"], "reason": reason})
            else:
                kept[field].append(item)
    if status == "refuse" or not kept["cause_ids"] or not kept["check_ids"]:
        return "refuse", {"cause_ids": [], "check_ids": []}, rejected
    return "unconfirmed", kept, rejected
