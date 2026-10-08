"""Freeze 50 distinct synthetic failures from the committed Product_A logs.

Run only to prepare a new evaluation version. Do not regenerate a frozen
manifest after model results have been inspected.
"""
import hashlib
import json
from pathlib import Path
import re

from src.rca_local.core import observation_from_form

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "L01/W04", "L02/W01", "L02/W03", "L03/W02", "L03/W03",
    "L04/W01", "L04/W04", "L05/W02", "L05/W06", "L05/W07",
)
TARGETS = {"continuity": (100, 12), "idd_static": (210, 14),
           "scan": (606, 14), "no_answer": (210, 10)}
QUESTION = {
    100: "What could explain this continuity failure, and what comparison should I perform?",
    210: "What could explain this static-current failure, and which check should I do next?",
    606: "Which possible scan-failure causes and distinguishing checks fit this observation?",
}


def failures():
    groups = {100: [], 210: [], 606: []}
    for relative in FILES:
        path = ROOT / "syn_data/product_a_v1/logs" / (relative + ".txt")
        raw = path.read_text(encoding="utf-8")
        lot, wafer = relative.split("/")
        starts = list(re.finditer(r"^Lot:\s+", raw, re.M))
        for i, start in enumerate(starts):
            text = raw[start.start():starts[i + 1].start() if i + 1 < len(starts) else len(raw)].strip()
            device = re.search(r"^Device:\s*(\d+)\b", text, re.M)
            if not device:
                continue
            failed = [number for number in groups if re.search(
                rf"^\s+{number}\b[^\n]*\(F\)", text, re.M)]
            if len(failed) != 1:
                continue
            number = failed[0]
            dut = f"Product_A-{lot}-{wafer}-D{int(device[1]):03d}"
            first = text.find("Failed Pins:\n")
            pin_block = text[first:].split("\n\n", 1)[0] if first >= 0 else ""
            pins = list(dict.fromkeys(re.findall(r"\b([A-Z][A-Z0-9_]*)\s*:\s*\d+", pin_block)))
            groups[number].append({
                "dut_id": dut, "failed_test": number, "observation": text,
                "pins": ",".join(pins), "source": path.relative_to(ROOT).as_posix(),
                "source_lines": {
                    "start": raw[:start.start()].count("\n") + 1,
                    "end": raw[:start.start() + len(text)].count("\n") + 1,
                },
                "excerpt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            })
    for number in groups:
        # Fixed ordering is independent of filesystem traversal.
        groups[number].sort(key=lambda x: hashlib.sha256(
            ("airgap-eval-v1|" + x["dut_id"]).encode()).hexdigest())
    return groups


def main():
    groups = failures()
    chosen = (groups[100][:12] + groups[210][:14] + groups[606][:14]
              + groups[210][14:24])
    if any(len(groups[number]) < minimum for number, minimum in
           ((100, 12), (210, 24), (606, 14))):
        raise SystemExit("Not enough distinct failures for the frozen manifest.")
    if len({entry["dut_id"] for entry in chosen}) != 50:
        raise SystemExit("Evaluation DUTs are duplicated.")
    history = {json.loads(path.read_text())["dut_id"] for path in
               (ROOT / "syn_data/product_a_scenarios_v1").glob("case_*/observation.json")}
    if any(entry["dut_id"] in history for entry in chosen):
        raise SystemExit("A historical or query-only case was selected.")
    items = []
    for i, entry in enumerate(chosen):
        no_answer = i >= 40
        form = {
            "product": "Product_A", "dut_id": entry["dut_id"],
            "failed_test": entry["failed_test"], "tester": "ate-01",
            "program": "product_a_ws", "pins": entry["pins"],
            "observation": ("210 IDD_Static: FAIL. The current value and limits were "
                            "omitted from this excerpt." if no_answer else entry["observation"]),
            "question": ("Can you distinguish high from low current and identify a supported "
                         "cause from this excerpt?" if no_answer else QUESTION[entry["failed_test"]]),
            "synthetic": True,
        }
        current = observation_from_form(form)
        category = ("no_answer" if no_answer else
                    {100: "continuity", 210: "idd_static", 606: "scan"}[entry["failed_test"]])
        items.append({
            "id": f"PA-EVAL-{i + 1:03d}", "category": category,
            "current": current,
            "provenance": {k: entry[k] for k in ("source", "source_lines", "excerpt_sha256")},
            "gold": {
                "expected_status": "refuse" if no_answer else None,
                "acceptable_cause_ids": [], "acceptable_check_ids": [],
                "review_status": "pending_engineer_review",
            },
        })
    manifest = {
        "schema_version": "airgap-rca-eval-v1",
        "frozen_inputs": True,
        "notes": "50 unique synthetic logged failures; 10 hide the IDD measurement and "
                 "are intended as no-answer probes. Gold labels require engineer review. "
                 "Do not report accuracy or false-answer rate from pending labels.",
        "items": items,
    }
    output = ROOT / "eval/product_a_v1_50.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"{output}: {len(items)} unique DUTs; 10 no-answer probes; labels pending review")


if __name__ == "__main__":
    main()
