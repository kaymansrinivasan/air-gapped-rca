# airgap-rca

Grounded root cause analysis on offline edge accelerators.

## Status

Repository skeleton only. The pipeline, dataset, models, verification rules,
and benchmark are not implemented or selected yet. No results are claimed.

## Structure

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

## Setup

No third-party dependencies are required for this skeleton.

