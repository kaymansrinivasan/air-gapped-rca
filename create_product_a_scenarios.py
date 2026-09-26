import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHUNK_FILE = (
    ROOT / "artifacts/product_a_chunks/product_a_dut_chunks.jsonl"
)
OUTPUT = ROOT / "syn_data/product_a_scenarios_v1"

# These are proposed synthetic histories based on the reviewed procedures.
# The engineer has NOT reviewed the newly simulated outcomes below.
SCENARIOS = [
    {
        "case_id": "case_01",
        "dut_id": "Product_A-L01-W03-D012",
        "review_row": 10,
        "review_status": "Not reviewed",
        "possible_causes": [
            "DUT-related electrical connection abnormality on TSTIN",
            "ATE interface/contact issue on the TSTIN path",
        ],
        "checks": [
            {
                "check": (
                    "Test a known-good reference die using comparable "
                    "program and test conditions on the suspect setup."
                ),
                "result": (
                    "The reference die also fails continuity on TSTIN."
                ),
            },
            {
                "check": (
                    "Repeat the comparison on an independently verified "
                    "compatible setup."
                ),
                "result": (
                    "Both the reference die and D012 pass continuity "
                    "on the comparison setup."
                ),
            },
        ],
        "assessment": (
            "The evidence points toward the original interface/contact "
            "path rather than a persistent D012 defect."
        ),
        "action": (
            "Clean and re-establish the probe contact on the original "
            "setup. Keep the test program and limits unchanged."
        ),
        "retest_checks": [
            {
                "check": "Retest the reference die on the original setup.",
                "result": "Continuity passes with zero failing pins.",
            },
            {
                "check": "Retest D012 three times on the original setup.",
                "result": (
                    "Continuity passes in all three attempts, "
                    "with zero failing pins."
                ),
            },
        ],
        "conclusion": (
            "The simulated comparisons and recovery after contact "
            "maintenance support a contact-related failure."
        ),
        "limitation": (
            "The precise contact defect is not established. "
            "Passing continuity does not establish a full-device pass."
        ),
    },
    {
        "case_id": "case_02",
        "dut_id": "Product_A-L01-W03-D005",
        "review_row": 11,
        "review_status": "Not reviewed",
        "possible_causes": [
            "DUT-related abnormality affecting XCKP, DTO2P and CCK",
            "Shared interface/contact or tester-resource issue",
        ],
        "checks": [
            {
                "check": (
                    "Run a known-good reference die on the original "
                    "setup under comparable conditions."
                ),
                "result": "The reference die passes continuity.",
            },
            {
                "check": (
                    "Retest D005 on the original and an independently "
                    "verified compatible setup."
                ),
                "result": (
                    "D005 fails continuity on XCKP, DTO2P and CCK "
                    "on both setups."
                ),
            },
        ],
        "assessment": (
            "The failure follows D005, supporting a DUT-associated "
            "abnormality. A specific internal defect is not identified."
        ),
        "action": (
            "Perform contact maintenance and repeat the comparison "
            "before placing D005 on hold for further analysis."
        ),
        "retest_checks": [
            {
                "check": "Retest the reference die after maintenance.",
                "result": "The reference die passes continuity.",
            },
            {
                "check": "Retest D005 after maintenance.",
                "result": (
                    "XCKP, DTO2P and CCK continue to fail continuity."
                ),
            },
        ],
        "conclusion": (
            "The simulated evidence supports a DUT-associated "
            "continuity failure; contact maintenance did not resolve it."
        ),
        "limitation": (
            "No physical failure analysis was simulated. "
            "The exact physical cause remains unresolved."
        ),
    },
    {
        "case_id": "case_03",
        "dut_id": "Product_A-L01-W01-D100",
        "review_row": 15,
        "review_status": "Plausible",
        "possible_causes": [
            "DUT-related excessive static current",
            "ATE measurement or interface issue producing a high reading",
        ],
        "checks": [
            {
                "check": (
                    "Verify the intended supply setting and measurement "
                    "configuration, then measure a known-good reference."
                ),
                "result": (
                    "The configuration matches the intended setup; "
                    "the reference measures 45.20 mA within 35–55 mA."
                ),
            },
            {
                "check": (
                    "Measure D100 on the original and an independently "
                    "verified compatible setup under matched conditions."
                ),
                "result": (
                    "D100 measures 65.40 mA and 65.70 mA respectively, "
                    "both above the 55 mA upper limit."
                ),
            },
        ],
        "assessment": (
            "The elevated current follows D100, supporting a "
            "DUT-associated high-current condition."
        ),
        "action": (
            "Verify contact integrity, repeat the measurement, "
            "and hold D100 for further electrical analysis."
        ),
        "retest_checks": [
            {
                "check": "Measure D100 after contact verification.",
                "result": (
                    "IDD_Static is 65.50 mA and remains above 55 mA."
                ),
            },
            {
                "check": "Repeat the reference measurement.",
                "result": "The reference measures 45.10 mA and passes.",
            },
        ],
        "conclusion": (
            "The simulated evidence supports persistent "
            "DUT-associated excessive static current."
        ),
        "limitation": (
            "The evidence does not distinguish leakage, an internal "
            "short, or another physical mechanism."
        ),
    },
    {
        "case_id": "case_04",
        "dut_id": "Product_A-L02-W02-D006",
        "review_row": 16,
        "review_status": "Plausible",
        "possible_causes": [
            "DUT-related abnormally low static current",
            "ATE measurement or supply-setup issue",
        ],
        "checks": [
            {
                "check": (
                    "Measure a known-good reference on the suspect "
                    "setup and compare the supply configuration "
                    "against the intended test configuration."
                ),
                "result": (
                    "The reference also reads below 35 mA. "
                    "The applied force-voltage setting differs "
                    "from the intended setting."
                ),
            },
            {
                "check": (
                    "Measure D006 and the reference on an independently "
                    "verified setup using the intended configuration."
                ),
                "result": (
                    "D006 measures 44.60 mA and the reference "
                    "measures 45.00 mA; both pass."
                ),
            },
        ],
        "assessment": (
            "A supply-setup mismatch is a plausible explanation "
            "for the low-current readings on the original setup."
        ),
        "action": (
            "Restore the intended force-voltage setting on the original "
            "setup and verify the applied supply before retesting. "
            "Keep the 35–55 mA acceptance limits unchanged."
        ),
        "retest_checks": [
            {
                "check": "Measure the reference on the corrected setup.",
                "result": "The reference measures 45.10 mA and passes.",
            },
            {
                "check": "Measure D006 on the corrected setup.",
                "result": "D006 measures 44.80 mA and passes.",
            },
        ],
        "conclusion": (
            "Recovery after correcting the supply configuration "
            "supports a setup-related low-current reading."
        ),
        "limitation": (
            "The voltage mismatch and resulting measurements are "
            "invented scenario assumptions, not recorded log facts."
        ),
    },
    {
        "case_id": "case_05",
        "dut_id": "Product_A-L01-W02-D005",
        "review_row": 20,
        "review_status": "Plausible",
        "possible_causes": [
            "DUT-related scan-path or logic issue involving DTOP",
            "ATE interface/contact or timing/setup issue",
        ],
        "checks": [
            {
                "check": (
                    "Repeat the scan test on the suspect DUT "
                    "and a known-good reference."
                ),
                "result": (
                    "The suspect DUT repeats the DTOP failure near "
                    "cycle 2347. The reference also shows DTOP "
                    "mismatches on the original setup."
                ),
            },
            {
                "check": (
                    "Compare both devices on an independently verified "
                    "compatible setup with matching pattern and timing."
                ),
                "result": "Both devices pass the scan test.",
            },
        ],
        "assessment": (
            "The comparison points toward the original interface "
            "or contact path."
        ),
        "action": (
            "Clean and re-establish probe contact on the original "
            "setup while keeping scan patterns and timing unchanged."
        ),
        "retest_checks": [
            {
                "check": "Retest the reference on the original setup.",
                "result": "The scan test passes.",
            },
            {
                "check": (
                    "Retest Product_A-L01-W02-D005 three times "
                    "on the original setup."
                ),
                "result": (
                    "The scan test passes in all three attempts "
                    "with no DTOP mismatch."
                ),
            },
        ],
        "conclusion": (
            "The simulated recovery after contact maintenance "
            "supports a contact-related scan failure."
        ),
        "limitation": (
            "The precise physical contact defect is not established."
        ),
    },
    {
        "case_id": "case_06",
        "dut_id": "Product_A-L01-W02-D001",
        "review_row": 21,
        "review_status": "Plausible",
        "possible_causes": [
            "DUT-related scan/control-path issue involving TSTIN and TSTEN",
            "Shared tester-resource or interface issue",
        ],
        "checks": [
            {
                "check": (
                    "Run a known-good reference on the original setup "
                    "using the same scan program and conditions."
                ),
                "result": "The reference passes the scan test.",
            },
            {
                "check": (
                    "Retest D001 on the original and an independently "
                    "verified compatible setup with matching conditions."
                ),
                "result": (
                    "TSTIN and TSTEN fail at cycle 4030 "
                    "on both setups."
                ),
            },
        ],
        "assessment": (
            "The repeatable signature follows D001, supporting "
            "a DUT-associated scan failure."
        ),
        "action": (
            "Verify contact integrity and repeat the scan test. "
            "Hold D001 for further scan diagnosis if failure persists."
        ),
        "retest_checks": [
            {
                "check": "Retest the reference after contact verification.",
                "result": "The reference passes the scan test.",
            },
            {
                "check": "Retest D001 after contact verification.",
                "result": (
                    "TSTIN and TSTEN still fail at cycle 4030."
                ),
            },
        ],
        "conclusion": (
            "The simulated evidence supports a persistent "
            "DUT-associated scan failure; the maintenance action "
            "did not resolve it."
        ),
        "limitation": (
            "The exact scan-chain defect or internal physical "
            "mechanism remains unresolved."
        ),
    },
]


def make_record(case_id, dut_id, stage, role):
    return {
        "schema_version": "1.0",
        "record_id": f"product_a_{case_id}_{stage}",
        "case_id": f"product_a_{case_id}",
        "dut_id": dut_id,
        "stage": stage,
        "role": role,
        "synthetic": True,
    }


def main():
    # Refuse to overwrite an earlier scenario set.
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Review it before creating a new version."
        )

    with CHUNK_FILE.open(encoding="utf-8") as file:
        rows = [
            (line_no, json.loads(line))
            for line_no, line in enumerate(file, start=1)
            if line.strip()
        ]

    chunks = {}
    for line_no, chunk in rows:
        dut_id = chunk["dut_id"]
        if dut_id in chunks:
            raise ValueError(f"Duplicate DUT ID in source: {dut_id}")
        chunks[dut_id] = (line_no, chunk)

    # Prepare every record before creating output files.
    records = []

    for scenario in SCENARIOS:
        case_id = scenario["case_id"]
        dut_id = scenario["dut_id"]
        line_no, chunk = chunks[dut_id]

        if not chunk.get("retrieval_text", "").strip():
            raise ValueError(f"Missing observation text: {dut_id}")

        observation = make_record(
            case_id, dut_id, "observation", "historical"
        )
        observation.update({
            "evidence_origin": "existing_synthetic_log_chunk",
            "source": {
                "file": CHUNK_FILE.relative_to(ROOT).as_posix(),
                "line": line_no,
            },
            "retrieval_text": chunk["retrieval_text"],
            # Preserve any original raw-log references and metadata.
            "original_chunk": chunk,
        })
        records.append((case_id, "observation", observation))

        review = {
            "file": "Case_cause(Reviewed).xlsx",
            "sheet": "Engineer review",
            "row": scenario["review_row"],
            "reviewer": "Ansari",
            "date": "2026-09-25",
            "status_as_recorded": scenario["review_status"],
            "scope": (
                "Possible causes and proposed investigation procedures; "
                "not the newly simulated findings or outcomes."
            ),
        }

        stage_contents = {
            "investigation": {
                "possible_causes": scenario["possible_causes"],
                "checks": scenario["checks"],
                "assessment": scenario["assessment"],
            },
            "action": {
                "action_taken": scenario["action"],
            },
            "retest": {
                "checks": scenario["retest_checks"],
                "conclusion": scenario["conclusion"],
                "limitation": scenario["limitation"],
            },
        }

        for stage, content in stage_contents.items():
            record = make_record(
                case_id, dut_id, stage, "historical"
            )
            record.update({
                "observation_id": observation["record_id"],
                "evidence_origin": "engineer_guided_synthetic_scenario",
                "physical_test_performed": False,
                "simulated_outcome_review_status": "not_reviewed",
                "guidance_source": review,
                **content,
            })
            records.append((case_id, stage, record))

    # A separate incoming incident, outside the five indexed lots.
    # No investigation, action, diagnosis, or retest is provided.
    query_case = make_record(
        "case_07",
        "Product_A-L06-W01-D001",
        "observation",
        "query_only",
    )
    query_case.update({
        "evidence_origin": "new_synthetic_query",
        "index_as_history": False,
        "source": {
            "type": "authored_synthetic_observation",
            "raw_log_file": None,
        },
        "context": {
            "product": "Product_A",
            "lot": "L06",
            "wafer": "W01",
            "tester": "ate-01",
            "program": "product_a_ws",
            "site": 0,
        },
        "test_result": {
            "test_number": 100,
            "test_name": "Open/Short-",
            "result": "FAIL",
            "failing_pins": [
                {"name": "TSTIN", "pin_number": 75}
            ],
            "later_tests_run": False,
            "bin": 4,
        },
        "retrieval_text": (
            "Product_A-L06-W01-D001; Product_A wafer-sort observation. "
            "Tester ate-01; program product_a_ws; site 0. "
            "Test 100 Open/Short- failed on TSTIN (75). "
            "Later tests were not run. Bin 4. "
            "No investigation or retest results are available."
        ),
    })
    records.append(("case_07", "observation", query_case))

    for case_id, stage, record in records:
        folder = OUTPUT / case_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{stage}.json"

        with path.open("x", encoding="utf-8") as file:
            json.dump(record, file, indent=2, ensure_ascii=False)
            file.write("\n")

    print("Historical cases: 6")
    print("Observation-only query cases: 1")
    print(f"JSON files created: {len(records)}")
    print(f"Saved to: {OUTPUT}")
    print("Chroma was not modified.")


if __name__ == "__main__":
    main()
