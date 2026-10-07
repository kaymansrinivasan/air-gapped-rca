# Jetson evaluation checkpoint

The structured investigation result now passes through `src/verify.py` after
model ID selection. It checks the exact investigation record and file hash,
source line range, failed test, pin-specific causes, and the direction of
IDD_Static failures. Inapplicable selections are removed. If no supported
cause and check remain, the answer refuses and `verification.json` records
why. A source integrity failure rejects the request.

These deterministic checks do **not** establish a physical cause, prove that a
general cause fits, or validate free-form chat. Chat remains experimental.

## Frozen question inputs

`eval/product_a_v1_50.json` holds 50 distinct synthetic Product_A DUTs from
ten committed wafer logs. The 40 full-observation questions cover continuity
(12), IDD_Static (14), and scan (14). Ten further IDD_Static questions omit
the current measurement, making high/low cause selection unsupported from
their supplied observation. All 50 are query-only, and none is one of the
seven scenario DUTs. Each entry identifies the source file and line span and
stores a hash of its original excerpt. The whole manifest has a SHA-256
identity printed by the validation command.

The set is frozen as **inputs**, not as a reviewed diagnostic answer key.
Repeated no-answer probes all exercise missing IDD direction; expand this
class in a later version if engineer review identifies additional realistic
no-answer cases. Do not regenerate or edit this manifest after inspecting
model results; create a new version instead.

From the repository root:

```bash
python3 benchmark.py --validate --manifest eval/product_a_v1_50.json
```

With the Jetson venv, local Chroma index, tokenizer, and TensorRT runtime
already in place, run the fixed inputs offline:

```bash
export RCA_DATA_ROOT=/home/orin_nano/Documents/airgap-rca
/home/orin_nano/Documents/airgap-rca/.venv-llamaindex-jetson/bin/python \
  benchmark.py --run --manifest eval/product_a_v1_50.json \
  --data-root "$RCA_DATA_ROOT" \
  --output "$RCA_DATA_ROOT/artifacts/evaluation/jetson_v1.json"
```

The runner does not overwrite an existing result. It records status, selected
source IDs, generation attempts, and wall-clock latency per question, plus
median and p95 latency. It reports how many of the ten incomplete-observation
probes refused. It deliberately leaves accuracy, false-answer rate, citation
validity, power and energy unset until reviewed gold labels and measurement
instrumentation are added. The model output and verification audit remain in
`artifacts/rca_app/<run_id>/`.

## Before comparing boards

1. Ask an ATE engineer to review each question and its acceptable causes,
   checks, and refusal expectation **before** using the model results as
   labels. Record the review separately so the question manifest stays fixed.
2. Inspect false answers and refusals on Jetson. Record the Orin Nano device
   and its engine format, since the original project guide names AGX Orin.
3. Add measured power and generated-token accounting. Test a cable-removed
   cold boot, recording the command, versions, and results.
4. Implement the RB3 Genie/QNN backend behind the same pipeline contract,
   confirm accelerator execution, then run the exact manifest and report its
   hash. Document Jetson FP16 versus RB3 W8 differences.
