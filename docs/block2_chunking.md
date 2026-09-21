# Block 2: evidence chunks with source citations

This implements input preparation for the synthetic A595 demonstration. It follows
the paper's immutable-source, checksum and inclusive-line-range requirements and
the split's explicit evidence allowlist. It does not perform retrieval, train or
run a model, infer a cause, or implement the final answer verifier.

## What becomes searchable later

The unchanged [first-demo split](../syn_data/splits/first_demo.json) selects all
25 historical evidence records from cases 001–006. Case 007 is a separate incoming
failure with no assigned cause. Its three observations stay together. The loader
does not recursively scan directories or read answer keys, replay questions,
scenario summaries or administrative documentation as evidence.

| Evidence | Boundary | Reason |
| --- | --- | --- |
| Measured failure record | Whole record | Keep test, pins, value, limits, flags and provenance together. |
| Simulated investigation | Whole record | Keep checks, findings, interpretation and simulation label together. |
| Simulated action | Whole record | Keep the action, its scope and linked evidence together. |
| Short simulated retest | Whole record | Preserve targeted/full-run distinction and actual device key. |
| Long simulated retest | Groups of complete test objects | Bound retrieval text while keeping each test's result, units, limits, flags and donor provenance intact. |
| Current case 007 | Whole record, separate output | Preserve fail–pass–fail, missing setup information and the absence of investigation. |

Long retests are grouped in source order by existing test-number hundred band,
record kind and unit, then capped at a target of 6,000 characters including the
repeated record header/footer. These are mechanical grouping keys, not asserted
physical failure families. No test object is split. There is no test overlap;
only record context repeats. Every source line and all 373 retest results survive.

The character target is a development default, not a measured optimum or a token
limit. Whole failure/investigation/action records remain atomic even if oversized.
An individual oversized test is also kept intact. Minified retests with objects
on the same line remain whole because disjoint line citations cannot be assigned.
`over_target` identifies these cases; nothing is silently truncated. Block 3 must
budget with its selected tokenizer and reject or handle oversized items explicitly.

## Run offline

Use Python 3.10+ from the repository root:

```sh
python src/ingest.py --output artifacts/first_demo
python -m unittest discover -s tests -v
```

Optional arguments are `--repo-root`, `--split` (repository-relative path) and
`--target-chars`. Output paths are relative to the shell working directory. Use a
new output directory for each run. Ingestion validates the complete input before
writing outputs; `chunk_manifest.json` is written last as the completion marker.
Output inside `syn_data` is rejected. The implementation uses the standard library
and makes no network calls. This is a host-side check, not an offline cold-boot
qualification of either edge board.

The input hashes require byte-preserving checkout. Clone with
`git -c core.autocrlf=false clone ...`; do not normalize or re-save source JSON/logs
to fix a hash failure. Check the expected Git blob and restore the original bytes.
The repository's `.gitattributes` also disables line-ending conversion for
`syn_data/**`, protecting subsequent Windows checkouts.

## Output and citation contract

| File | Consumer |
| --- | --- |
| `historical_chunks.jsonl` | Block 3 historical keyword/vector indexes. |
| `current_case_chunks.jsonl` | Separate current evidence supplied with the query. Never add it to the historical index. |
| `question.txt` | Exactly the engineer question selected by the split, with a final LF. |
| `chunk_manifest.json` | Audit/reproducibility metadata; never searchable model evidence. |

Each JSONL row contains:

- Stable `chunk_id`, algorithm version, role, stage, incident and evidence IDs.
- `device_key` copied from that record, `incident_device_key` from its failure,
  and a derived `device_matches_incident` flag. A mismatch is retained, not repaired.
- Unchanged `origin`, `source_anchor` and evidence links. A
  `shared_source_failure_key` exposes related scenarios without collapsing them.
- `source_id`/`source_file` (repository-relative POSIX path), SHA-256 of original
  file bytes, inclusive one-based `source_lines.start/end`, and exact `text`.
- `context`: separately addressed verbatim record header/footer spans for split
  retests. They retain device identity, simulation label, run scope, total count,
  outcome and evidence links with each test group.
- `test_indices` (zero-based positions), `test_numbers`, and `json_pointers`.
  For a whole record, the JSON pointer is the empty root pointer.
- `retrieval_text`: primary and context spans assembled in source order for
  indexing. It is a composite view; it is **not** one contiguous source quote.
  Cite the primary or context span containing the claim, not the entire composite.
- `original_citations`: verified readable-log excerpts attached only to original
  measured failures, with preserved STDF references. These are secondary citations;
  the chunk's primary citation still identifies its own JSON file.

Line 1 is the first source line. LF terminates a line; CRLF bytes and all whitespace
are retained. A final LF does not create an extra empty line. Hashes cover original
file bytes, not normalized or reserialized text. Chunk identity incorporates the
algorithm version, role, file path, content hash and primary line range. Repeating
the same build yields identical output bytes. A source change creates new IDs;
an unapproved change fails the configured Git blob hash check first.

For illustration, case 001's measured continuity excerpt resolves to
`syn_data/source/a595_tester_log.txt`, lines 12704–12706. Its invented investigation
must instead cite `syn_data/incidents/SYN-A595-001/simulated_investigation.json`.
The original tester log cannot substantiate that fictional investigation.
Donor lines inside simulated retests remain template provenance; they are not
reclassified as actual retest evidence or added as independent measured incidents.

For retrieval and eventual generation, pass role, incident/device identity,
origin, `test_indices` and citation spans alongside the selected content. A chunk
from a full-catalogue retest represents only the listed test indices; the source's
`test_count: 123` describes its parent run, not the number of tests in every chunk.
The later answer verifier must distinguish quote integrity from causal support.

## Validation and demo results

At input commit `9acc7720d8a196a5b9d69141da868c265587bb94`, default output is:

| Role/stage | Source records | Chunks |
| --- | ---: | ---: |
| Historical failures | 6 | 6 |
| Historical investigations | 6 | 6 |
| Historical actions | 6 | 6 |
| Historical retests | 7 | 73 |
| Current failure (007) | 1 | 1 |
| Total | 26 | 92 |

All 373 synthetic retest results are represented exactly once by primary test
groups; no default chunk exceeds the target. Cases 001 and 002 still share their
original failure key. The other-device pass in case 006 remains searchable with
`device_matches_incident: false`. Case 007 has no original-source citation and no
invented diagnosis. Six historical scenarios still represent five distinct
original failures, not six independent real incidents.

Tests cover source alteration, original-log alteration, bad citations even after
updating the JSON blob hash, forbidden/duplicate/missing files, current/history
mixing, future-current evidence, complete test objects, exact line extraction,
identity mismatches, duplicate JSON keys, paths, Unicode/CRLF/minified JSON,
deterministic outputs, unchanged source bytes and network-free execution.

The manifest inventories the configured evidence plus resolved original source
hashes. It does not claim to validate every file in the old package manifest.
The frozen split's `runtime_enforcement: PENDING_BLOCK_2_IMPLEMENTATION` field
records its pre-implementation state; this module and the generated manifest
provide the actual enforcement/result. The split and all incident/source files
are left byte-identical so existing provenance remains valid.

STDF file hashes and offset bounds are checked, but binary records are not decoded
again by this module. The structural tests do not establish physical causality,
engineer approval, retrieval quality, model answer accuracy or board performance.
The next block is local keyword/vector indexing of historical chunks, with case
007 supplied separately as current evidence.
