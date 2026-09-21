# First A595 demonstration split

This records the agreed separation before Block 2 (chunking).

| Role | Cases | Allowed evidence |
| --- | --- | --- |
| Historical collection | SYN-A595-001 through SYN-A595-004 | All four records per case: failure, investigation, action and retest (16 files). |
| Incoming query | SYN-A595-005 | observed_failure.json only (1 file). |
| Evaluator only | SYN-A595-005 | Its investigation, action and retest remain hidden from retrieval and model input. |
| Reserved | SYN-A595-006 | All five records excluded from this demonstration. |

The exact file allowlists and source Git blob hashes are in [first_demo.json](first_demo.json). Paths are relative to the repository root. The source revision is pinned there. Original records remain in their existing locations.

## How Block 2 must use this split

1. Read this configuration as control data, never as model evidence.
2. Build historical chunks only from `historical.files`. Preserve evidence IDs, case/device identity, original-versus-simulated labels, source paths and line references.
3. Prepare current evidence only from `query.files`, separately from the historical index. Use `query.question` for this demonstration.
4. Exclude every unlisted file from retrieval and model input. Do not recursively ingest `syn_data/` or the repository.
5. In particular, do not feed scenario titles, scenario/evidence indexes, replay files, answer keys, this README, the split configuration, or full source logs to the model. They can reveal future events or evaluation information. A citation verifier may resolve only the exact original-source excerpts cited by permitted evidence.
6. After recording the model response, the evaluator may inspect the selected failure-stage answer-key row and the held-out case-005 records. Do not use later events to judge what the model could have known at the failure stage.
7. Check file hashes against the pinned source revision before loading; if records change, explicitly review and version the split.

The configuration defines the boundary; a runtime loader enforcing it is still pending. No chunks, vector index, retrieval, model run or benchmark have been produced by this step. "Hidden" means excluded from model input, not deleted or access-restricted in GitHub.

## What this demonstration can show

The historical collection contains continuity, supply-current and high-delay examples, but no matching scan-failure history. Case 005 therefore tests whether the system can describe a new failure, recognize insufficient historical support and withhold an unsupported confirmed cause. It does not establish successful diagnosis from a close historical match.

All follow-up histories are synthetic and await engineer review. Cases 001 and 002 share one source-device failure and remain together; four historical scenarios represent three distinct original failures. Case 005 is new to this demonstration's historical collection, not a newly collected real-world incident.

The original `syn_data/manifest.json` is a pre-existing package manifest; it is not a current repository inventory and does not cover these new split files. It also references documentation/scripts absent from the current repository. This split records its own source revision and file identities without rewriting that original manifest.
