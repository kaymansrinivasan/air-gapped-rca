# airgap-rca

Grounded root cause analysis on offline edge accelerators.

## Current status

The active dataset is **synthetic Product_A wafer-sort data**: five lots, seven wafers per lot, and 112 DUTs per wafer. Its first-attempt readable logs and STDF V4 files are available. No Product_A DUT was physically measured, and no Product_A investigation, corrective action, retest or confirmed-cause record has been created.

The earlier A595 demonstration and its A595-specific Block 2 chunker have been retired from the active tree. Git history retains them. Block 2 for Product_A will be designed after its new incident scenarios and evidence are reviewed with ATE engineers. Retrieval, models, answer verification and edge-board benchmarking remain unimplemented.

## Data

- [Product_A readable logs, metadata and validation](syn_data/product_a_v1/README.md)
- [Product_A STDF files for viewers](syn_data/STDF_a_v1/README.md)

The generated observations and chosen failure rates are demonstration data, not manufacturing measurements or physical root-cause evidence.

## Proposed workflow

![Air-gapped RCA workflow: immutable logs, chunks with source lines, local vector and keyword indexes, an engineer question, evidence retrieval, ranked causes and citations, deterministic citation checks, and a supported answer or refusal.](docs/images/immutable-log-evidence.png)

The diagram describes the intended architecture; later stages are not yet demonstrated.
