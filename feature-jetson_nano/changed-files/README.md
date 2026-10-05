# airgap-rca

Grounded root cause analysis on offline edge accelerators.

## Current status

The active dataset contains 3,920 synthetic Product_A wafer-sort observations
(five lots, 35 wafers) and six curated historical scenarios. Engineering
guidance informed possible causes and investigation procedures. The simulated
investigation, action and retest outcomes are **not physically tested and not
reviewed**. They are not confirmed real-world diagnoses.

LlamaIndex-to-Chroma retrieval and LlamaIndex-to-Qwen3.5-0.8B inference have
been demonstrated on the Jetson Orin Nano. Free-text RCA drafts produced wrong
citations and historical-DUT references and were rejected.

The new local app presents a browser form for a failure observation and an
engineer's question. Qwen selects evidence IDs; code supplies source-owned
text and citations. Invalid selections are rejected. Source validation is
not proof that a hypothesis explains the current DUT.

- [Local app setup, supported inputs and limitations](docs/local-app.md)
- [Observed Jetson integration progress](docs/jetson-progress.md)
- Start the app: `python -m src.rca_local` inside the Jetson environment.
- Run software tests: `python -m unittest discover -s tests -v`.

The new selection workflow still needs on-board validation. The full quality
benchmark, offline unplug/reboot test, RB3 integration and board comparison
are pending. This is a research prototype, not a production RCA service.

## Proposed workflow

![Air-gapped RCA workflow: immutable logs, chunks with source lines, local vector and keyword indexes, an engineer question, evidence retrieval, ranked causes and citations, deterministic citation checks, and a supported answer or refusal.](docs/images/immutable-log-evidence.png)

The diagram describes the whole intended system.

The diagram describes the target architecture; not all acceptance criteria
have been demonstrated.

