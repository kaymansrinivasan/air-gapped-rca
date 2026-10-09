# Final benchmark batch

The October 8 hybrid runs completed 50 questions on each board. They establish
provisional proxy acceptance, latency and source membership. They do not close
all of the evaluation requirements in the project guide. The remaining software
measurements are consolidated into one resumable batch per board.

## Run

On each board, fetch the measurement branch and fast-forward its local checkout.
The explicit fetch works with the RB3 clone's narrow fetch refspec:

```bash
git switch feature/measurement-hybrid-audit &&
git fetch origin refs/heads/feature/measurement-hybrid-audit &&
git merge --ff-only FETCH_HEAD
```

Jetson, in `~/Documents/airgap-rca-minimal`:

```bash
bash scripts/finish_benchmarks.sh jetson
```

RB3, in `~/Documents/airgap-rca-rb3`:

```bash
bash scripts/finish_benchmarks.sh rb3
```

The wrapper runs the software tests, checks/starts the RB3 server when needed,
and launches the suite with `nohup`. Run the boards concurrently, but do not send
interactive UI requests to a board while it benchmarks. Expect approximately
45–70 minutes on Jetson and 100–130 minutes on RB3, based on the previously
observed per-question latencies. Retries can extend this estimate.

Progress is saved after every question in
`RCA_DATA_ROOT/artifacts/evaluation/final_suite_v1/progress.json`.
Repeat the same command after a stopped process to resume completed questions.
The runner refuses a concurrent run or a resume after code, source or configuration
changes. A runtime-failed question already checkpointed stays in the evaluation;
do not silently delete it to improve the result. Fixing code needs a new output
directory. A process killed in the middle of a question must finish or be stopped
before resuming, to avoid competing model calls.

## Fixed protocol

All four conditions run all 50 frozen questions, without editing inputs or the
proxy key. Benchmark generation permits up to three ranked causes and two checks
to expose top-3 hit rate. This is a new explicit protocol; the interactive UI and
earlier 50-question runs use a two-cause limit. Do not pool the two protocols.

| Condition | Intervention |
| --- | --- |
| full | Hybrid retrieval, IDD applicability prefilter, pin filter, verifier and refusal |
| no_keyword | Vector ranking with no keyword contribution; every other setting matches full |
| no_verifier | No IDD prefilter, pin catalog filter or post-selection applicability check; schema, ID ownership and source hash integrity remain |
| no_refusal | Force model choices from retrieved history even without IDD support; record verifier rejection but return the model's raw choices when it would otherwise refuse |

In no_refusal, malformed output still counts as rejected. The runner never
manufactures a selection when the model refuses or fails. This is an operational
ablation of the abstention policy: output format and source ownership still apply.
In no_verifier, immutable source handling remains part of the evidence assembly
architecture. The ablation measures the applicability verifier, not arbitrary
fabricated source text. These details must accompany the ablation table.

Raw experiments are saved under `artifacts/rca_experiments/<run_id>/`, separate
from the app's case/chat directory. These controls are not web form fields or
environment switches available to the UI. The UI always uses its normal verified
pipeline. Post-run audits observe experimental errors without repairing answers.

## Outputs

The suite directory contains `full.json`, `no_keyword.json`, `no_verifier.json`,
`no_refusal.json`, `suite.json`, `metrics.csv`, `REPORT.md`, and SVG plots.
Tables are generated from result rows, not manually entered numbers. The suite
records source/code hashes, runtime commit/config hashes, package versions, OS,
device model and available power/clock queries in its environment metadata.
Inspect any failed metadata query before claiming complete hardware provenance.

After copying both completed `suite.json` files to one machine, generate the
cross-board quality/performance/ablation tables and three SVG figures:

```bash
python3 -m src.compare_suites --jetson jetson_suite.json --rb3 rb3_suite.json \
  --output-dir artifacts/evaluation/comparison
```

The comparison checks protocol, code, source, question and label hashes and
refuses mismatched runs. Keep both per-board environment snapshots beside the
generated tables.

Reported quality metrics include proxy top-1/top-3 hit rates, acceptance under
the supported-subset rubric, refusals, rejections, false answers under that rubric,
coverage, and source citation validity. A citation can be genuine while its cause
is inappropriate; those are scored separately. An all-refusal system therefore
has zero answer coverage, not perfect cause accuracy.

Jetson enables native TensorRT Edge-LLM `--dumpProfile --profileOutputFile`.
The parser reads `generation.generated_tokens` and `generation.tokens_per_second`
from the v0.10.0 profile format. RB3 reads its server's measured decode rate.
All available generation attempts, including retries, contribute to throughput.
These runtime timing boundaries may differ; end-to-end request latency is reported
separately. Profiling is enabled consistently across the four Jetson conditions.
Reference: https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.10.0/examples/utils/profileFormatter.cpp

Jetson power uses sampled board input VDD_IN, not GPU-only power. Each question's
energy is its mean sampled power multiplied by its end-to-end latency. Missing
power for any question leaves aggregate power/energy unset and records coverage.
The wrapper uses **an illustrative MYR 0.50/kWh scenario**, not a claim about the
lab's actual tariff. Energy-only cost is `mean_joules / 3600 * 0.50` per 1,000
triages; hardware, labour and cooling costs are excluded. RB3 power/energy/cost
stay unmeasured without contemporaneous board input readings from an inline
meter or a documented input sensor. An empty sensor-file search is insufficient
to assert no hardware sensor exists; further sensor investigation is deferred.

## Gates software cannot complete

- Independent ATE-reviewed labels remain pending. The AI-derived key is based
  on six synthetic histories and is not evidence of real diagnostic accuracy.
- Physically disconnect network connectivity, reboot each board from a local
  console, and run a new request. Save the output and record the five-minute
  demonstration requested in the guide. This action cannot be inferred from
  offline environment variables or local model execution while connected.
- Measure RB3 input power with suitable hardware if no documented telemetry is
  available; do not substitute CPU load or an NPU execution flag.
- Confirm recorded precision, power modes and clocks. Jetson's per-request engine
  load differs from RB3's persistent server and must be reported.
- Free-form chat remains experimental and requires separate claim assessment.
- Complete the cross-board charts and the 8–12-page paper after collecting the
  remaining measurements. Any unavailable metric must remain explicitly missing.

## Three observed failure modes

1. **Structured-output noncompliance:** earlier continuity output selected three
   cause IDs and four check IDs against a two-item limit; another output put K
   check IDs inside cause_ids. Validation rejected them. A bounded retry helps
   but does not establish diagnostic correctness.
2. **Historical symptom mismatch:** the earlier Jetson v1 run refused six IDD
   questions after selecting case_04 low-current suggestions for incompatible or
   unknown current direction. Applicability filtering before selection addressed
   the observed cases; the frozen proxy v2 run then had 40 suggestions/10 refusals.
3. **Free-form chat claim leakage and repetition:** observed chat mentioned holding
   historical D100 while discussing current D010 and repeated a long response to
   a one-line-summary request. The structured suite does not validate chat claims.

These are operator-observed failures retained in the project session and board
artifacts. They are not failure frequencies estimated on an independent dataset.
