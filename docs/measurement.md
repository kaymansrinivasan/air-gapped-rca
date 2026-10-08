# Measurement checkpoint for both boards

The fixed 50-question inputs are in `eval/product_a_v1_50.json`. Jetson and RB3
have already run them with the AI-derived proxy key. Preserve those original
result files and all matching `artifacts/rca_app/<run_id>/` folders. Their
reported 100% **proxy agreement** is not independent engineering accuracy.

## Audit the existing runs without another inference pass

On Jetson, from the working code tree after pulling the audit branch:

```bash
cd ~/Documents/airgap-rca-minimal
export RCA_DATA_ROOT=$HOME/Documents/airgap-rca
$RCA_DATA_ROOT/.venv-llamaindex-jetson/bin/python -m src.measure \
  --result "$RCA_DATA_ROOT/artifacts/evaluation/jetson_proxy_v2.json" \
  --data-root "$RCA_DATA_ROOT" \
  --output "$RCA_DATA_ROOT/artifacts/evaluation/jetson_proxy_v2_audit.json"
```

On RB3:

```bash
cd ~/Documents/airgap-rca-rb3
.venv-rca/bin/python -m src.measure \
  --result artifacts/evaluation/rb3_proxy_v1.json \
  --data-root "$PWD" \
  --output artifacts/evaluation/rb3_proxy_v1_audit.json
```

The audit checks every displayed cause and check against its exact historical
file, hash, line range and symptom rules. A changed or unsupported answer stops
the audit. It calculates citation validity for source membership, and RB3
decode throughput from the saved QNN server's `decode_tokens_per_second` metric.
Jetson's saved responses do not contain a decode-only timing metric; leave that
field unset until it is measured. The audit never describes citation matching
as proof that the diagnosis is physically correct.

## Capture power on Jetson

The historical 50-question results have no contemporaneous power readings.
A new run with a distinct filename is necessary:

```bash
cd ~/Documents/airgap-rca-minimal
export RCA_DATA_ROOT=$HOME/Documents/airgap-rca
$RCA_DATA_ROOT/.venv-llamaindex-jetson/bin/python benchmark.py --run \
  --backend jetson --retrieval-mode hybrid --jetson-power \
  --manifest eval/product_a_v1_50.json \
  --gold eval/product_a_v1_50_proxy_gold.json --allow-proxy-gold \
  --data-root "$RCA_DATA_ROOT" \
  --output "$RCA_DATA_ROOT/artifacts/evaluation/jetson_proxy_power_v3.json"
```

This writes `jetson_proxy_power_v3.power.csv` with one `case_id,watts` reading
per `tegrastats` VDD_IN sample during each request. VDD_IN is board input
power, including CPU and peripherals; it is not GPU-only power. The app measures
latency around each request while the sampler runs concurrently. Follow with:

```bash
$RCA_DATA_ROOT/.venv-llamaindex-jetson/bin/python -m src.measure \
  --result "$RCA_DATA_ROOT/artifacts/evaluation/jetson_proxy_power_v3.json" \
  --data-root "$RCA_DATA_ROOT" \
  --power-csv "$RCA_DATA_ROOT/artifacts/evaluation/jetson_proxy_power_v3.power.csv" \
  --power-method 'tegrastats VDD_IN board input rail, one sample per second' \
  --output "$RCA_DATA_ROOT/artifacts/evaluation/jetson_proxy_power_v3_audit.json"
```

The audit requires a sample for every case. Inspect a missing-sample error
instead of fabricating power values. Energy uses measured mean watts multiplied
by the corresponding question's end-to-end latency. Add
`--electricity-per-kwh PRICE` only after documenting the currency and actual
electricity tariff; then it reports energy cost for 1,000 triages.

On RB3, use a calibrated inline input meter or documented board input telemetry
to collect watt readings **during each numbered case**. Save a two-column CSV
with header `case_id,watts`, with repeated rows for each case as sampled. Run
the same `src.measure` audit with `--power-csv`, `--power-method` and a new
`--output`. Do not substitute the RB3 server's NPU execution flag for power.

## Retrieval and comparison

The default pipeline now combines Chroma vector rank with exact log-token rank
using weighted reciprocal rank fusion, after same-product and same-test
filtering. A controlled vector-only run uses `--retrieval-mode vector_only` and
a different output file. The historical library has only six curated cases
(two per test), so this ablation has limited scope. The earlier Jetson v2 and
RB3 v1 runs were **vector-only** before this change; do not present a new hybrid
run as though it used their exact retrieval configuration.

Three further gates require separate evidence: an ATE engineer or independently
documented key for diagnostic accuracy, network-disconnected cold boot and
request on both devices, and controlled verification/refusal ablations. Free
form chat is not covered by the structured 50-question evaluation.
