"""Offline evaluation runner for a frozen, reviewed RCA question manifest.

Validation works without a model. --run uses the local Jetson pipeline and
records raw outcomes. Accuracy and false-answer rate need reviewed gold labels.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time


def load_manifest(path: Path):
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != "airgap-rca-eval-v1" or data.get("frozen_inputs") is not True:
        raise ValueError("Expected a frozen airgap-rca-eval-v1 manifest.")
    items = data.get("items")
    if not isinstance(items, list) or len(items) < 50:
        raise ValueError("The evaluation needs at least 50 items.")
    ids, duts = set(), set()
    data_root = path.resolve().parent.parent
    no_answer = 0
    for item in items:
        current, gold = item["current"], item["gold"]
        if item["id"] in ids or current["dut_id"] in duts:
            raise ValueError("Evaluation IDs and DUTs must be unique.")
        ids.add(item["id"])
        duts.add(current["dut_id"])
        if current.get("role") != "query_only" or current.get("index_as_history") is not False:
            raise ValueError("Evaluation incidents must remain query-only.")
        if not current.get("question") or current.get("test_result", {}).get("result") != "FAIL":
            raise ValueError("Every item needs a question and an observed failed test.")
        provenance = item["provenance"]
        source = (data_root / provenance["source"]).resolve()
        if not source.is_relative_to(data_root) or not source.is_file():
            raise ValueError("Evaluation source is missing or outside the repository.")
        lines = source.read_text(encoding="utf-8").splitlines()
        first, last = provenance["source_lines"]["start"], provenance["source_lines"]["end"]
        if type(first) is not int or type(last) is not int or not 1 <= first <= last <= len(lines):
            raise ValueError("Evaluation source line range is invalid.")
        excerpt = "\n".join(lines[first - 1:last]).strip()
        if hashlib.sha256(excerpt.encode()).hexdigest() != provenance["excerpt_sha256"]:
            raise ValueError("Evaluation source excerpt changed.")
        if item["category"] == "no_answer":
            no_answer += 1
            if gold["expected_status"] != "refuse":
                raise ValueError("No-answer probes require a refusal label.")
        if gold["review_status"] not in ("pending_engineer_review", "approved"):
            raise ValueError("Gold label review status is invalid.")
    if no_answer < 10:
        raise ValueError("The evaluation needs at least 10 no-answer probes.")
    return data, hashlib.sha256(raw).hexdigest()


def summarize(items, results):
    statuses = Counter(row["status"] for row in results)
    elapsed = sorted(row["latency_seconds"] for row in results)
    summary = {
        "count": len(results), "statuses": dict(statuses),
        "median_latency_seconds": statistics.median(elapsed),
        "p95_latency_seconds": elapsed[math.ceil(.95 * len(elapsed)) - 1],
        "no_answer_refusal_count": sum(
            result["status"] == "refuse" for item, result in zip(items, results)
            if item["category"] == "no_answer"),
        "no_answer_total": sum(item["category"] == "no_answer" for item in items),
    }
    # A pending label is not ground truth. Never fill these with a proxy score.
    summary["accuracy"] = None
    summary["false_answer_rate"] = None
    summary["citation_validity"] = None
    summary["power_watts"] = None
    summary["energy_joules_per_triage"] = None
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate", action="store_true", help="Check manifest without inference.")
    parser.add_argument("--run", action="store_true", help="Run all items through the local Jetson backend.")
    parser.add_argument("--data-root", type=Path, help="RCA_DATA_ROOT containing the local index.")
    parser.add_argument("--output", type=Path, help="New JSON result file; never overwritten.")
    args = parser.parse_args()
    data, manifest_hash = load_manifest(args.manifest)
    counts = Counter(item["category"] for item in data["items"])
    print(json.dumps({"items": len(data["items"]), "categories": counts,
                      "manifest_sha256": manifest_hash,
                      "approved_gold": sum(x["gold"]["review_status"] == "approved"
                                           for x in data["items"])}, indent=2))
    if not args.run:
        if not args.validate:
            parser.error("Choose --validate or --run.")
        return
    if not args.output or not args.data_root:
        parser.error("--run requires --output and --data-root.")
    if args.output.exists():
        raise SystemExit("Result file already exists. Choose a new --output.")
    os.environ["RCA_DATA_ROOT"] = str(args.data_root.resolve())
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", ANONYMIZED_TELEMETRY="False")
    from src.rca_local.pipeline import run_pipeline
    results = []
    for item in data["items"]:
        start = time.perf_counter()
        answer = run_pipeline(item["current"])
        row = {"id": item["id"], "run_id": answer["run_id"], "status": answer["status"],
               "latency_seconds": time.perf_counter() - start,
               "generation_attempts": answer.get("generation_attempts", 0),
               "cause_ids": [x["id"] for x in answer.get("causes", [])],
               "check_ids": [x["id"] for x in answer.get("checks", [])]}
        results.append(row)
        print(item["id"], row["status"], f'{row["latency_seconds"]:.2f}s', flush=True)
    output = {"manifest_sha256": manifest_hash, "backend": "jetson_tensorrt_edge_llm",
              "summary": summarize(data["items"], results), "results": results,
              "notes": "Pending engineer labels: accuracy, citation validity and FAR are not scored."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
