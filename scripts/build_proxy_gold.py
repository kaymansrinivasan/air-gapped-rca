"""Draft a provisional answer key from the six synthetic history scenarios.

This is an AI-derived proxy, not an ATE engineer review or a physical diagnosis.
It may be used to exercise a board comparison, but cannot validate accuracy.
"""
import argparse
import json
from pathlib import Path

from benchmark import load_manifest
from src.rca_local.core import load_history, make_catalog
from src.verify import _direction


def proxy_row(root, item):
    current = item["current"]
    number = current["test_result"]["test_number"]
    pins = {x["name"].upper() for x in current["test_result"]["failing_pins"]}
    history = load_history(root, current)
    matches = [case for case in history.values()
               if case["records"]["observation"]["original_chunk"]["failed_test"] == number]
    catalog = make_catalog({"current": current, "matches": matches})
    causes, checks, rationale = set(), set(), []
    if item["category"] == "no_answer":
        rationale.append("The supplied IDD excerpt omits the value and limits; "
                         "high versus low current cannot be distinguished.")
    elif number == 100:
        # Cases 01–02 have 'Not reviewed' in their guidance_source.
        causes.add("product_a_case_02_cause_2")  # General interface/tester hypothesis.
        checks.update(key for key, option in catalog.items() if option["kind"] == "check")
        if "TSTIN" in pins:
            causes.update(("product_a_case_01_cause_1", "product_a_case_01_cause_2"))
        if {"XCKP", "DTO2P", "CCK"} <= pins:
            causes.add("product_a_case_02_cause_1")
        rationale.append("Continuity pin-specific hypotheses require matching named pins. "
                         "The general contact/tester hypothesis and comparison checks "
                         "remain unconfirmed. Case 01–02 guidance was recorded as 'Not reviewed'.")
    elif number == 210:
        direction = _direction(current)
        case = {"high": "product_a_case_03", "low": "product_a_case_04"}.get(direction)
        if case:
            causes.update(key for key, option in catalog.items()
                          if option["kind"] == "cause" and option["case_id"] == case)
            checks.update(key for key, option in catalog.items()
                          if option["kind"] == "check" and option["case_id"] == case)
            rationale.append(f"Measured static current is {direction}; only the same-direction "
                             f"history ({case}) supplies these provisional hypotheses and checks.")
        else:
            rationale.append("The current observation does not establish high versus low current.")
    elif number == 606:
        causes.update(("product_a_case_05_cause_2", "product_a_case_06_cause_2"))
        checks.update(key for key, option in catalog.items() if option["kind"] == "check")
        if "DTOP" in pins:
            causes.add("product_a_case_05_cause_1")
        if {"TSTIN", "TSTEN"} <= pins:
            causes.add("product_a_case_06_cause_1")
        rationale.append("Specific scan-path hypotheses require matching pins. General "
                         "interface/timing possibilities and comparison checks remain unconfirmed.")
    else:
        raise ValueError(f"Unsupported failed test: {number}")
    choices = set(catalog)
    if not causes <= choices or not checks <= choices:
        raise ValueError(f"Proxy cited an unavailable source for {item['id']}.")
    status = "unconfirmed" if causes and checks else "refuse"
    return {
        "id": item["id"], "dut_id": current["dut_id"], "failed_test": number,
        "expected_status": status,
        "acceptable_cause_ids": sorted(causes),
        "acceptable_check_ids": sorted(checks),
        "review_status": "proxy", "reviewer": "",
        "labeler": "AI proxy based on six synthetic scenario records",
        "review_note": " ".join(rationale)
                       + " No current DUT cause or physical outcome is confirmed.",
        "source_case_ids": sorted({catalog[key]["case_id"] for key in causes | checks}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, digest = load_manifest(args.manifest)
    root = args.manifest.resolve().parent.parent
    rows = [proxy_row(root, item) for item in manifest["items"]]
    if len(rows) != 50 or sum(row["expected_status"] == "refuse" for row in rows) != 10:
        raise ValueError("Unexpected proxy label distribution.")
    output = {
        "schema_version": "airgap-rca-reviewed-gold-v1",
        "manifest_sha256": digest,
        "label_origin": "ai_proxy_from_six_synthetic_scenarios",
        "limitations": [
            "Not an independent or engineer-reviewed answer key.",
            "Continuity scenario guidance (cases 01 and 02) was recorded as Not reviewed.",
            "The other scenario guidance covered plausible causes and proposed checks, not the simulated outcomes.",
            "Agreement with these labels is a proxy metric, not diagnostic accuracy.",
        ],
        "items": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("Provisional proxy labels:", len(rows), "refusals:", 10, "saved:", args.output)


if __name__ == "__main__":
    main()
