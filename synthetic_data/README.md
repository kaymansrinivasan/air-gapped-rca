# A595 synthetic RCA workflow pilot

This pilot demonstrates the data flow from a real failure to a simulated investigation, action and retest. It does not establish the actual causes of the A595 failures, validate real-world diagnosis accuracy, connect to a live tester, or run an AI model.

## Contents and provenance

- 6 synthetic incidents anchored to 5 real A595 failed devices; 001 and 002 deliberately share device 80's initial evidence but represent mutually exclusive fictional histories.
- 18 staged questions and separate designer-authored reference answers.
- 24 replay events, 7 synthetic retest records, including three full 123-test passing replays, three correctly linked failing targeted repeats and one deliberately mismatched passing targeted record.
- Original text and STDF files are copied byte-for-byte under `source/` with hashes. The case files quote the real failure and its original source line and byte-offset references.
- Investigation, action and retest documents carry `origin: SIMULATED`. Reference answers are explicitly designer-authored, not model predictions or engineer-certified labels.
- Synthetic device keys identify fictional twins. `source_anchor` identifies the real measured device used as a starting point; it does not assert that the fictional continuation happened to that real device.
- Passing retests reuse measurements from a nearby passing die on the same wafer as a numerical template. Failed repeats reuse the failure signature. This is controlled scenario construction, not a calibrated physical model or a statistical reconstruction of factory behavior.
- The wrong-device record is an intentional negative case. It must be excluded from proof of recovery. A targeted check never establishes a complete device pass.

## Start here

Read `walkthrough.md` for the complete six-case demonstration. For a staged feed, run these commands after extracting the ZIP (Python 3, standard library only):

```text
python scripts/validate_dataset.py
python scripts/replay_demo.py --incident SYN-A595-001
python scripts/replay_demo.py --incident SYN-A595-001 --speed 60
python scripts/replay_demo.py --incident SYN-A595-005 --show-reference
python scripts/replay_demo.py --incident SYN-A595-006
```

The default replay emits all stages without delay. `--speed 60` replays each simulated 60-second interval in one wall-clock second. The emitted JSON contains stage-appropriate evidence and questions. `--show-reference` additionally displays the separately stored reference response. It does not call or evaluate a model. These replay timings are invented; the original timestamps are not used because the source has known timestamp inconsistencies.

## Connecting your RCA assistant

1. Ingest only the evidence emitted for the current incident/stage. Do not preload future events.
2. Keep `reference/`, this README, the walkthrough, validation reports and scenario titles out of the assistant's retrieval context. They reveal the authored scenarios and expected outcomes.
3. Pass the emitted question to your assistant. Require its answer to state observation, hypotheses, evidence, missing information, and an appropriately qualified conclusion.
4. Require citations by evidence ID and a supporting field or source line. Resolve them through `evidence_index.json`; hashes provide file integrity, not proof that a fictional observation happened.
5. At retest, join the incident and full synthetic device key, check the scope and outcomes, and distinguish successful correction, persistent failure and unresolved diagnosis.
6. Compare the answer with the withheld reference for demonstration checks: correct failure extraction, evidence provenance, identity checks, available-at-this-stage citations, alarm handling, appropriate abstention, and closure scope. Do not report these six authored scenarios as a meaningful estimate of real-world RCA accuracy.

## Extending to Galaxy and other STDF files

Those files have not yet been included. Decode each supplied file separately; retain raw bytes, source hashes, lot/wafer/device/site/run identifiers, original flags, units, limits and software-bin meanings. Do not assume bin numbers or test numbers mean the same thing across products. Deduplicate identical measurements and keep alternate encodings of one run together. A canonical key should include dataset and source hash as well as lot, wafer, part, site and run/attempt identity.

This release is demonstration-only; no train/test split is claimed. If later used for training or evaluation, keep original devices, template-donor devices, duplicate files, related runs and all counterfactual variants in the same lineage group. Do not randomly split individual test rows. Use independent real, engineer-reviewed incidents to assess real-world diagnostic performance.

## Engineering limits and references

The scenario mechanisms require review by a test engineer familiar with the actual product and equipment. No complete test program, scan-cell map, factory alarm definition or physical failure-analysis record was recovered. No physical cause is inferred solely from a bin, current value, pin or cycle. No production commands or process settings are prescribed.

Probe contamination and cleaning are a general mechanism discussed by [FormFactor](https://www.formfactor.com/blog/2022/removing-probing-debris-from-springs-dram-flash-applications/); that background does not establish contamination in A595. Explicit synthetic labels and traceable provenance are consistent with the transparency approaches discussed by [NIST](https://www.nist.gov/publications/reducing-risks-posed-synthetic-content-overview-technical-approaches-digital-content). Neither source validates these fictional cases.

The original data attribution is retained in `source/source_manifest.json`. No additional rights are asserted for the supplied source data.
