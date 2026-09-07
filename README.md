# airgap-rca

Grounded root cause analysis on offline edge accelerators.

## Status

Repository skeleton only. The pipeline, dataset, models, verification rules,
and benchmark are not implemented or selected yet. No results are claimed.

## Goal

Take public or synthetic machine logs and a failure question, retrieve local
evidence, generate ranked candidate causes with exact citations, verify those
citations deterministically, and refuse when no supported candidate survives.
Run and benchmark the same pipeline on Jetson AGX Orin and Qualcomm RB3 Gen 2.

## Structure (guide section 11)

- `requirements.txt`: pinned dependencies, once selected and verified.
- `offline/inventory.md`: downloaded assets, sizes, checksums, and offline checks.
- `src/ingest.py`: chunks with original file and line provenance.
- `src/index.py`: local embeddings and keyword retrieval.
- `src/generate.py`: structured generation with citations.
- `src/verify.py`: deterministic citation checks and refusal.
- `src/measure.py`: latency, generation throughput, power, and energy.
- `eval/questions.jsonl`: evaluation questions; currently empty and not frozen.
- `benchmark.py`: future benchmark entry point.
- `results/jetson/` and `results/rb3/`: generated benchmark outputs.
- `paper/`: Markdown paper draft and future generated figures.
- `notes/decisions.md`: decisions and their reasons.
- `notes/hardware.md`: measured hardware and software configurations.

## First milestone

Inventory both boards, prepare offline dependencies, preserve log provenance,
and demonstrate one end-to-end Jetson request after network disconnection and
cold boot. Record observed results; do not mark this milestone complete yet.

## Working rules

Use public or synthetic data only. Keep evaluation answers out of retrieval.
Do not fine-tune, build a UI, or start the robot extension before the core gates.
Preserve raw logs and original line references. Citation validity alone does not
establish causal correctness. Pin the dataset version and record its source.

Send Monday, Wednesday, and Friday updates, always including repository and
paper links. Escalate a repeated blocker after 90 minutes and hardware blocks
within the same hour. Keep these updates separate from university reports.

## Setup

No third-party dependencies are required for this skeleton. Select and pin the
actual runtime dependencies after hardware checks. Download compatible wheels
and model assets before the offline test.

After extracting this folder, initialise version control if needed:

```bash
git init -b main
git add .
git commit -m "Create section 11 project skeleton"
```

Configure your own Git identity if Git requests it. A remote repository has not
been created or connected. Add its URL and the paper link below once available.

Repository: TBD
Paper draft: TBD
