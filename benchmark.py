"""Offline evaluation runner for a frozen RCA question manifest.

Validation works without a model. --run uses the local Jetson pipeline and
records raw outcomes. AI proxy labels produce provisional agreement, never
engineer-validated diagnostic accuracy.
"""
import argparse
import csv
from collections import Counter
from contextlib import nullcontext
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


def load_gold(path: Path, manifest: dict, manifest_hash: str, allow_proxy=False):
    gold = json.loads(path.read_text(encoding="utf-8"))
    if gold.get("schema_version") != "airgap-rca-reviewed-gold-v1" or gold.get("manifest_sha256") != manifest_hash:
        raise ValueError("Gold labels are for a different manifest.")
    entries = gold.get("items")
    if not isinstance(entries, list) or len(entries) != len(manifest["items"]):
        raise ValueError("Gold labels must cover every frozen question.")
    origin = gold.get("label_origin", "human_engineer_review")
    proxy = origin == "ai_proxy_from_six_synthetic_scenarios"
    if proxy and not allow_proxy:
        raise ValueError("AI proxy labels require --allow-proxy-gold; they are not engineer review.")
    if not proxy and origin != "human_engineer_review":
        raise ValueError("Unknown gold-label origin.")
    labels = {}
    for entry in entries:
        identity = entry["id"]
        if identity in labels:
            raise ValueError("Duplicate gold label.")
        if proxy:
            if entry.get("review_status") != "proxy" or not entry.get("labeler") or entry.get("reviewer"):
                raise ValueError("Proxy label must not impersonate an engineer review.")
        elif entry.get("review_status") != "approved" or not entry.get("reviewer"):
            raise ValueError("unreviewed or anonymous gold label.")
        status = entry.get("expected_status")
        causes, checks = entry.get("acceptable_cause_ids"), entry.get("acceptable_check_ids")
        if status not in ("unconfirmed", "refuse") or not isinstance(causes, list) or not isinstance(checks, list):
            raise ValueError("Invalid gold label.")
        if any(not isinstance(x, str) or not x for x in causes + checks):
            raise ValueError("Gold IDs must be strings.")
        if (status == "refuse" and (causes or checks)) or (status == "unconfirmed" and (not causes or not checks)):
            raise ValueError("Gold cause/check choices conflict with expected status.")
        labels[identity] = entry
    if set(labels) != {item["id"] for item in manifest["items"]}:
        raise ValueError("Gold and manifest question IDs differ.")
    return labels, origin


def summarize(items, results, gold, origin="human_engineer_review"):
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
    correct = []
    false_answers = 0
    answered = 0
    top1_correct = 0
    top2_correct = 0
    for item, result in zip(items, results):
        label = gold[item["id"]]
        causes = result["cause_ids"]
        checks = result["check_ids"]
        acceptable = set(label["acceptable_cause_ids"])
        valid = (result["status"] == label["expected_status"] == "refuse"
                 or result["status"] == label["expected_status"] == "unconfirmed"
                 and bool(causes) and bool(checks) and set(causes) <= acceptable
                 and set(checks) <= set(label["acceptable_check_ids"]))
        correct.append(valid)
        if result["status"] == "unconfirmed":
            answered += 1
            false_answers += not valid
            top1_correct += bool(causes and causes[0] in acceptable)
            top2_correct += bool(set(causes[:2]) & acceptable)
    proxy = origin == "ai_proxy_from_six_synthetic_scenarios"
    prefix = "proxy_" if proxy else ""
    summary[prefix + "exact_answer_agreement" if proxy else "exact_answer_accuracy"] = sum(correct) / len(correct)
    summary[prefix + "false_answer_rate_per_answer"] = false_answers / answered if answered else None
    summary[prefix + "top1_cause_hit_rate_on_answered"] = top1_correct / answered if answered else None
    summary[prefix + "top2_cause_hit_rate_on_answered"] = top2_correct / answered if answered else None
    if proxy:
        summary["exact_answer_accuracy"] = None
        summary["false_answer_rate_per_answer"] = None
        summary["top1_cause_accuracy_on_answered"] = None
        summary["top2_cause_hit_rate_on_answered"] = None
    summary["top3_cause_hit_rate_on_answered"] = None  # Current selection allows only two causes.
    summary["answered_count"] = answered
    summary["false_answer_count"] = false_answers
    summary["citation_validity"] = None
    summary["power_watts"] = None
    summary["energy_joules_per_triage"] = None
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate", action="store_true", help="Check manifest without inference.")
    parser.add_argument("--run", action="store_true", help="Run all items through the selected local backend.")
    parser.add_argument("--backend", choices=("jetson", "rb3"), default="jetson",
                        help="Local generation backend (default: jetson).")
    parser.add_argument("--retrieval-mode", choices=("hybrid", "vector_only"), default="hybrid",
                        help="Use the full retrieval or vector-only ablation; writes the mode into results.")
    parser.add_argument("--data-root", type=Path, help="RCA_DATA_ROOT containing the local index.")
    parser.add_argument("--output", type=Path, help="New JSON result file; never overwritten.")
    parser.add_argument("--gold", type=Path, help="Separate labels for this manifest.")
    parser.add_argument("--allow-proxy-gold", action="store_true",
                        help="Allow explicitly provisional AI proxy labels.")
    parser.add_argument("--jetson-power", action="store_true",
                        help="Sample Jetson VDD_IN with tegrastats during each question; saves a matching power CSV.")
    args = parser.parse_args()
    data, manifest_hash = load_manifest(args.manifest)
    counts = Counter(item["category"] for item in data["items"])
    print(json.dumps({"items": len(data["items"]), "categories": counts,
                      "manifest_sha256": manifest_hash,
                      "embedded_labels_pending_review": sum(
                          x["gold"]["review_status"] != "approved" for x in data["items"])}, indent=2))
    try:
        labels, origin = load_gold(args.gold, data, manifest_hash, args.allow_proxy_gold) if args.gold else (None, None)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.run:
        if not args.validate:
            parser.error("Choose --validate or --run.")
        if labels is not None:
            print("Labels:", len(labels), "origin:", origin)
        return
    if not args.output or not args.data_root or not args.gold:
        parser.error("--run requires --output, --data-root and --gold labels.")
    if labels is None:
        parser.error("--run requires --gold labels.")
    if args.output.exists():
        raise SystemExit("Result file already exists. Choose a new --output.")
    if args.jetson_power and args.backend != "jetson":
        parser.error("--jetson-power requires --backend jetson.")
    power_path = args.output.with_suffix(".power.csv") if args.jetson_power else None
    if power_path and power_path.exists():
        raise SystemExit("Power CSV already exists. Choose a new --output.")
    os.environ["RCA_DATA_ROOT"] = str(args.data_root.resolve())
    os.environ["RCA_BACKEND"] = args.backend
    os.environ["RCA_RETRIEVAL_MODE"] = args.retrieval_mode
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", ANONYMIZED_TELEMETRY="False")
    from src.rca_local.pipeline import run_pipeline
    from src.measure import JetsonPowerSampler
    results = []
    if power_path:
        power_path.parent.mkdir(parents=True, exist_ok=True)
        with power_path.open("x", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(["case_id", "watts"])
    for item in data["items"]:
        with JetsonPowerSampler() if args.jetson_power else nullcontext() as sampler:
            start = time.perf_counter()
            answer = run_pipeline(item["current"])
            elapsed = time.perf_counter() - start
        if power_path:
            with power_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerows((item["id"], value) for value in sampler.values)
        row = {"id": item["id"], "run_id": answer["run_id"], "status": answer["status"],
               "latency_seconds": elapsed,
               "generation_attempts": answer.get("generation_attempts", 0),
               "cause_ids": [x["id"] for x in answer.get("causes", [])],
               "check_ids": [x["id"] for x in answer.get("checks", [])]}
        results.append(row)
        print(item["id"], row["status"], f'{row["latency_seconds"]:.2f}s', flush=True)
    output = {"manifest_sha256": manifest_hash,
              "gold_sha256": hashlib.sha256(args.gold.read_bytes()).hexdigest(),
              "label_origin": origin,
              "backend": {"jetson": "jetson_tensorrt_edge_llm", "rb3": "rb3_qnn_htp"}[args.backend],
              "retrieval_mode": args.retrieval_mode,
              "power_samples_csv": str(power_path) if power_path else None,
              "power_method": "Jetson tegrastats VDD_IN, 1000 ms interval (board input rail)" if power_path else None,
              "summary": summarize(data["items"], results, labels, origin), "results": results,
              "notes": ("AI proxy agreement only; not independent engineer-validated diagnostic accuracy."
                        if origin == "ai_proxy_from_six_synthetic_scenarios" else
                        "Engineer-reviewed labels applied; citation validity and power require separate audits.")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
