# airgap-rca

Grounded root cause analysis on offline edge accelerators.

## Status

Repository skeleton only. models, verification rules,
and benchmark are not implemented or selected yet. No claim

## Proposed workflow

![Air-gapped RCA workflow: immutable logs, chunks with source lines, local vector and keyword indexes, an engineer question, evidence retrieval, ranked causes and citations, deterministic citation checks, and a supported answer or refusal.](docs/images/immutable-log-evidence.png)

Proposed architecture; implementation and validation are pending.

## Setup

No third-party dependencies are required for this skeleton.

Files to change:
- src/ingest.py
- README.md
- notes/decisions.md
- .gitignore
New files to add:
- .gitattributes
- tests/test_ingest.py
- docs/block2_chunking.md
Keep all other existing files unchanged, including:
- Everything in syn_data/
- Everything in paper/
- Everything in offline/
- Everything in results/
- docs/images/immutable-log-evidence.png
- notes/hardware.md
- src/index.py
- src/generate.py
- src/verify.py
- src/measure.py
- benchmark.py
- requirements.txt
- LICENSE
