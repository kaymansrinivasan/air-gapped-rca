# Historical cases and a new incoming case

All six original cases stay as historical examples. Case 007 is a separate fictional incoming failure.

| Role | Cases | Files |
| --- | --- | --- |
| Historical collection | SYN-A595-001 through SYN-A595-006 | All 25 existing failure, investigation, action and retest records. |
| Current input | SYN-A595-007 | One observed_failure.json containing only currently available observations and provenance. |

[first_demo.json](first_demo.json) lists the exact paths for each role. Version 2 replaces the previous arrangement that used case 005 as the query and reserved case 006. No original incident file has been edited.

## Case 007

[Open the new case](../incidents/SYN-A595-007/observed_failure.json).

A fictional device has a fail–pass–fail sequence on test 100 (Open/Short-), with AMSDSM pin 40 reported on the failing attempts. These are three executions of one check, not three full-device test runs. No investigation, corrective intervention, assigned cause, or answer key has been created.

The test/pin vocabulary comes from the historical continuity example; the new sequence and identifiers are authored. No real device was tested. This example can exercise retrieval of relevant histories and cautious suggestions. It is not independent evidence of diagnostic accuracy.

The actual tester, product and setup details are unspecified. Their compatibility with a historical case must not be assumed. Historical cases 001 and 002 offer different follow-up outcomes for the same original symptom; neither establishes the cause of case 007.

## Next: Block 2

- Chunk the 25 historical records for later indexing, retaining case/device identity, evidence IDs, origin labels, file paths and source lines.
- Prepare case 007 separately as current evidence. Its own JSON file and line range are its citation source; it has no real STDF record or original-source line range.
- Use the engineer question from the configuration.
- Keep answer keys, replay files, scenario summaries, this README and configuration metadata out of searchable evidence/model input. They are administrative or evaluation material.
- Verify each listed Git blob hash before loading. The historical source commit identifies the unchanged original records; the new query is identified by its own blob hash in this version's repository tree.

This is input preparation for retrieval, not a training split. No historical case is held out or reserved now. The runtime loader, chunks, indexes and model pipeline are still pending.

All historical follow-up records remain explicitly synthetic. Six historical scenarios come from five distinct original failures because 001 and 002 share an original failure.

## Existing manifest

The original syn_data/manifest.json describes an earlier package and is not a current repository inventory. The hashes for the updated indexes, case 007 and these configuration files have been refreshed or added. It still lists previously absent documentation/scripts from the earlier package. The input configuration provides the exact files and per-file Git blob hashes for this demonstration.
