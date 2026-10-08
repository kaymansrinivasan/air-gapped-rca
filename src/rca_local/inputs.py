"""Conservative extraction for single-DUT Product_A text excerpts."""
import json
import re

TESTS = {100: "Continuity / Open–Short", 210: "IDD_Static", 606: "Scan"}
NAME = r"(?:Open/Short-?|Continuity|IDD_Static|Scan)"
TEST_LINE = re.compile(rf"\b(?:(?P<before>\d{{2,4}})\s+{NAME}|{NAME}\s*\((?P<after>\d{{2,4}})\))", re.I)


def inspect_log(text):
    from .core import require
    require(isinstance(text, str) and len(text) <= 12000, "Use a single-DUT excerpt of at most 12,000 characters.")
    hints = {}
    identities = set(re.findall(r"\bProduct_A-L\d+-W\d+-D\d+\b", text))
    identities.update(re.findall(r"^\s*DUT\s*:\s*([A-Za-z0-9_.-]+)", text, re.M | re.I))
    require(len(identities) <= 1, "This excerpt contains multiple DUT IDs. Submit one DUT at a time.")
    if identities:
        hints["dut_id"] = next(iter(identities))
    if "Product_A" in text:
        hints["product"] = "Product_A"
    for field, label in (("tester", "Tester"), ("program", "Program")):
        values = set(re.findall(rf"\b{label}:\s*([A-Za-z0-9_.-]+)", text))
        require(len(values) <= 1, f"Multiple {label.lower()} values found; use a single test attempt.")
        if values:
            hints[field] = next(iter(values))
    tests = list(TEST_LINE.finditer(text))
    failures = set()
    for i, match in enumerate(tests):
        end = tests[i + 1].start() if i + 1 < len(tests) else len(text)
        segment = re.split(r"[\n;]", text[match.end():end], maxsplit=1)[0]
        if re.search(r"\bFAIL(?:ED)?\b|\(F\)|Failing Pins:\s*[1-9]\d*", segment, re.I):
            failures.add(int(match.group("before") or match.group("after")))
    require(len(failures) <= 1, "Multiple failed tests found; use one failure per request.")
    if failures:
        number = next(iter(failures))
        require(number in TESTS, f"Log uses unsupported failed-test number {number}. Check the original test program; do not relabel a real log.")
        hints["failed_test"] = number
    return hints


def form_options(root):
    """Offer only values present in local history, plus UI unknown/custom options."""
    testers, programs = set(), set()
    for path in sorted((root / "syn_data/product_a_scenarios_v1").glob("case_*/observation.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("role") != "historical":
            continue
        chunk = record.get("original_chunk", {})
        text = record.get("retrieval_text", "")
        for field, label, values in (("tester", "Tester", testers), ("program", "Program", programs)):
            value = chunk.get(field)
            if not value:
                match = re.search(rf"\b{label}:\s*([A-Za-z0-9_.-]+)", text)
                value = match.group(1) if match else None
            if isinstance(value, str) and value:
                values.add(value)
    return {"products": ["Product_A"], "tests": [{"number": n, "name": name} for n, name in TESTS.items()],
            "testers": sorted(testers), "programs": sorted(programs)}
