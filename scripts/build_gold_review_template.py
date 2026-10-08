"""Create an engineer review worksheet without running the model."""
import argparse
import json
from pathlib import Path

from benchmark import load_manifest
from src.rca_local.core import load_history, make_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, digest = load_manifest(args.manifest)
    root = args.manifest.resolve().parent.parent
    rows = []
    for item in manifest["items"]:
        current = item["current"]
        history = load_history(root, current)
        matches = [case for case in history.values()
                   if case["records"]["observation"]["original_chunk"]["failed_test"]
                   == current["test_result"]["test_number"]]
        catalog = make_catalog({"current": current, "matches": matches})
        rows.append({
            "id": item["id"], "dut_id": current["dut_id"],
            "failed_test": current["test_result"]["test_number"],
            "question": current["question"],
            "source": item["provenance"],
            "expected_status": "refuse" if item["category"] == "no_answer" else None,
            "acceptable_cause_ids": [], "acceptable_check_ids": [],
            "reviewer": "", "review_status": "pending_engineer_review",
            "review_note": "",
            "candidate_options": [
                {"id": key, "kind": entry["kind"], "text": entry["text"],
                 "historical_case": entry["case_id"]}
                for key, entry in catalog.items()
            ],
        })
    data = {"schema_version": "airgap-rca-reviewed-gold-v1",
            "manifest_sha256": digest, "items": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("Review template:", args.output, "items:", len(rows))


if __name__ == "__main__":
    main()
