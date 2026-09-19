# airgap-rca

Grounded root cause analysis on offline edge accelerators.

## Status

First executable synthetic-workflow prototype: staged evidence ingestion, lexical
retrieval, local-model adapter, conservative citation verification, and refusal.
Fixture replay is tested; actual LLM/board benchmarks and engineer review are pending.

## Proposed workflow

![Air-gapped RCA workflow: immutable logs, chunks with source lines, local vector and keyword indexes, an engineer question, evidence retrieval, ranked causes and citations, deterministic citation checks, and a supported answer or refusal.](docs/images/immutable-log-evidence.png)

Target architecture. The current retrieval baseline uses lexical TF-IDF vectors;
learned semantic embeddings and hardware validation are pending.

## Setup

Python 3.10+; no third-party dependencies for the prototype.

```bash
python -m unittest discover -s tests -v
python rca.py --question-id SYN-A595-001-Q-INVESTIGATION
```

The default fixture mode tests the pipeline without an LLM. See the
[prototype guide](docs/prototype.md) for local-model commands, evidence rules,
historical-case restrictions, and known limitations.

