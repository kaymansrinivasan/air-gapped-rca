"""Audit saved benchmark runs without repeating model inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import threading

from src.rca_local.core import EvidenceError
from src.verify import _source_record, verify_selection


def parse_tegrastats_power(line):
    """Read input-board power in watts from Jetson tegrastats output."""
    match = re.search(r"\bVDD_IN\s+(\d+)mW(?:/\d+mW)?\b", line)
    return int(match[1]) / 1000 if match else None


class JetsonPowerSampler:
    """Capture the Jetson's total input rail for one benchmark request."""

    def __enter__(self):
        self.values = []
        self.process = subprocess.Popen(["tegrastats", "--interval", "1000"],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        text=True)
        def read_lines():
            for line in self.process.stdout:
                value = parse_tegrastats_power(line)
                if value is not None:
                    self.values.append(value)
        self.thread = threading.Thread(target=read_lines, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.thread.join(timeout=3)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_power(path, ids):
    samples = {key: [] for key in ids}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["case_id", "watts"]:
            raise ValueError("Power CSV columns must be case_id,watts.")
        for row in reader:
            if row["case_id"] not in samples:
                raise ValueError("Unknown case in power samples.")
            watts = float(row["watts"])
            if not math.isfinite(watts) or watts <= 0:
                raise ValueError("Power readings must be positive finite watts.")
            samples[row["case_id"]].append(watts)
    if any(not values for values in samples.values()):
        raise ValueError("Every evaluated case needs measured power samples.")
    return samples


def audit(result_path, data_root, power_csv=None, power_method=None, electricity_per_kwh=None):
    result_path, data_root = result_path.resolve(), data_root.resolve()
    result = read(result_path)
    rows = result["results"]
    if not rows or len(rows) != result["summary"]["count"]:
        raise ValueError("Benchmark result count is inconsistent.")
    backend = result["backend"]
    if backend not in ("jetson_tensorrt_edge_llm", "rb3_qnn_htp"):
        raise ValueError("Unknown benchmark backend.")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation ID.")
    if (power_csv is None) != (power_method is None):
        raise ValueError("Provide both --power-csv and --power-method.")
    if electricity_per_kwh is not None and (not math.isfinite(electricity_per_kwh) or electricity_per_kwh <= 0 or power_csv is None):
        raise ValueError("Cost requires measured power and a positive electricity price.")
    power = load_power(power_csv, ids) if power_csv else None
    details, tokens, decode_seconds, energies, means, peaks = [], [], [], [], [], []
    citation_total = citation_valid = 0
    for row in rows:
        run_id = row["run_id"]
        if not isinstance(run_id, str) or len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise ValueError("Invalid run ID.")
        folder = data_root / "artifacts/rca_app" / run_id
        answer = read(folder / "answer.json")
        if answer["run_id"] != run_id or answer["status"] != row["status"]:
            raise ValueError("Saved answer and benchmark row disagree.")
        selected = {"cause_ids": answer["causes"], "check_ids": answer["checks"]}
        if any([x["id"] for x in selected[key]] != row[key] for key in selected):
            raise ValueError("Saved citations and benchmark selections disagree.")
        items = selected["cause_ids"] + selected["check_ids"]
        entry = {"id": row["id"], "run_id": run_id, "citation_count": len(items),
                 "completion_tokens": None, "decode_tokens_per_second": None,
                 "mean_power_watts": None, "peak_power_watts": None, "energy_joules": None}
        if row["status"] == "unconfirmed":
            bundle = read(folder / "evidence.json")
            status, retained, rejected = verify_selection(data_root, bundle, "unconfirmed", selected)
            if status != "unconfirmed" or rejected or any(
                    [x["id"] for x in retained[key]] != row[key] for key in selected):
                raise EvidenceError("Saved answer failed re-verification: " + row["id"])
            matches = {(m["records"]["observation"]["case_id"], m["records"]["observation"]["dut_id"]): m
                       for m in bundle["matches"]}
            for item in items:
                match = matches.get((item["case_id"], item["historical_dut"]))
                if match is None:
                    raise EvidenceError("Citation has no matched historical case: " + row["id"])
                _source_record(data_root, item, match)
                citation_valid += 1
        elif items:
            raise ValueError("A non-answer contains cited suggestions.")
        citation_total += len(items)
        if backend == "rb3_qnn_htp" and (folder / "response.json").exists():
            response = read(folder / "response.json")
            n = response.get("usage", {}).get("completion_tokens")
            rate = response.get("qwen35_metrics", {}).get("decode_tokens_per_second")
            if type(n) is int and n > 0 and isinstance(rate, (int, float)) and math.isfinite(rate) and rate > 0:
                entry["completion_tokens"], entry["decode_tokens_per_second"] = n, rate
                tokens.append(n)
                decode_seconds.append(n / rate)
        if power is not None:
            values = power[row["id"]]
            mean = statistics.mean(values)
            entry.update(mean_power_watts=mean, peak_power_watts=max(values),
                         energy_joules=mean * row["latency_seconds"])
            means.append(mean)
            peaks.append(max(values))
            energies.append(entry["energy_joules"])
        details.append(entry)
    summary = {"citation_validity": citation_valid / citation_total if citation_total else None,
               "valid_citations": citation_valid, "total_citations": citation_total,
               "decode_tokens_per_second": sum(tokens) / sum(decode_seconds) if decode_seconds else None,
               "decode_sample_count": len(decode_seconds), "power_method": power_method,
               "average_power_watts": statistics.mean(means) if means else None,
               "peak_power_watts": max(peaks) if peaks else None,
               "mean_energy_joules_per_triage": statistics.mean(energies) if energies else None,
               "cost_per_1000_triages": statistics.mean(energies) / 3_600_000 * 1000 * electricity_per_kwh
                   if energies and electricity_per_kwh else None,
               "electricity_price_per_kwh": electricity_per_kwh}
    return {"schema_version": "airgap-rca-run-audit-v1", "backend": backend,
            "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "manifest_sha256": result["manifest_sha256"], "label_origin": result["label_origin"],
            "question_count": len(rows), "summary": summary, "cases": details,
            "limitations": ["Citation validity means source membership and symptom checks, not diagnostic correctness.",
                            "RB3 decode speed comes from saved QNN metrics; Jetson decode speed is not in saved responses.",
                            "Power CSV samples must be physical readings recorded during each corresponding request.",
                            "Proxy agreement is not independent engineering accuracy."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--power-csv", type=Path)
    parser.add_argument("--power-method")
    parser.add_argument("--electricity-per-kwh", type=float)
    args = parser.parse_args()
    report = audit(args.result, args.data_root, args.power_csv, args.power_method, args.electricity_per_kwh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report["summary"], indent=2))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()

