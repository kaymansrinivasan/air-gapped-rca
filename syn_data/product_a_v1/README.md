# Product_A wafer-sort test logs, version 1

This directory contains **generated test results** for an offline RCA prototype. No Product_A DUT was measured. The A595 readable log supplied the layout, test names and numbers, one coordinate layout, and the example `IDD_Static` limits. It did **not** supply Product_A measurements, real failure rates, investigations, corrective actions, or confirmed causes.

## Contents

- `logs/L01/W01.txt` through `logs/L05/W07.txt`: 35 UTF-8/LF readable wafer logs, one per wafer.
- `device_metadata.jsonl`: one external record per DUT with the complete identity, coordinate, result and actually executed tests. The `origin` field is `GENERATED`.
- `config.json`: seed, fixed time, exact outcome quotas, test limits, pin map, pattern range, and setup assumptions.
- `coordinate_template.json`: the 112 unique positions from A595 wafer 02, mapped to `D001`–`D112`.
- `generate.py`, `validate.py`, and `tests/`: standard-library generator, independent log reader/validator, and focused checks.
- `validation_report.json`: counts read back from the written log files.
- `manifest.json`: generated origin, assumptions, source and generator hashes, output hashes and dataset totals. It does not hash itself.
- `EXAMPLES.md`: four short excerpts from the generated logs, one per outcome.

## Dataset and execution rules

The 5 lots contain 7 wafers each, with 112 DUTs per wafer: **3,920 DUTs**. A canonical identity is `Product_A-L01-W01-D001`. The raw device header displays `001`; its product, lot and wafer come from the directory and headers, and the external metadata records the full identity. Device numbering restarts on each wafer.

Every DUT begins with test 100 `Open/Short-`. A passing DUT then executes test 210 `IDD_Static`, followed by test 606 `SCAN Test` if current is within limits. Execution stops at the first failed test. Bins **4, 3, 2, 1** respectively mean continuity failure, IDD failure, scan failure and all three tests passed. These are this prototype's bin definitions.

| Outcome | DUTs | Executed tests per DUT |
| --- | ---: | --- |
| Bin 1: pass | 3,528 | 100, 210, 606 |
| Bin 4: continuity failure | 98 | 100 |
| Bin 3: IDD failure | 98 | 100, 210 |
| Bin 2: scan failure | 196 | 100, 210, 606 |

The expected executed-test counts are 3,920 for test 100, 3,822 for test 210, and 3,724 for test 606: **11,466 test results**. No later test row is written after a failure. The quotas are chosen demonstration counts, not an observed manufacturing yield.

The generated IDD limits are strict: `35.00 < current < 55.00 mA`. A passing displayed value is 38.00–52.00 mA; a failing displayed value is 27.00–33.00 or 57.00–70.00 mA. These are assumed prototype limits, not verified Product_A specifications. The assumed nominal 2.00 V supply is only configuration metadata; it is not another result row. Scan uses the declared `scan` pattern, pin map and vector range 1–8191. Passing pin checks have zero failed pins; failure counts equal the distinct listed pins.

The fixed start is `2026-01-01T08:00:00+08:00` in **Asia/Kuala_Lumpur**. The raw log displays A595-style dates without a timezone suffix; `config.json` and metadata supply the offset. Device times follow their wafer start chronologically. This deliberately corrects a chronology inconsistency in the A595 source, whose first wafer 02 device timestamp precedes its stated wafer start.

The seed fixes a stable SHA-256 ranking for outcome placement and values. The generator reserves two groups of three adjacent failing DUT coordinates for each failure category, then fills exact quotas across the remaining DUTs with stable rankings and moderate wafer variation. These generated spatial patterns are **not evidence of a physical mechanism**. There are no retests, diagnoses, investigations or corrective actions in this dataset.

The A595 source files and earlier case histories were removed from the active repository. The coordinate template and manifest retain their original reference path and SHA-256 as historical provenance; that path now resolves through Git history rather than the current tree. Product_A results are generated independently from the included configuration and coordinate template.

