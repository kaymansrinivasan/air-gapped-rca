# Offline RCA prototype: first executable slice

This implementation exercises the diagram with the committed synthetic cases.
It is a development prototype, not a validated ATE diagnostic system. No raw
source files, synthetic records, or reference answers are changed.

## Run locally (Python 3.10+; standard library only)

From the repository root:

```bash
python -m unittest discover -s tests -v
python rca.py --question-id SYN-A595-001-Q-INVESTIGATION
python rca.py --question-id SYN-A595-006-Q-RETEST
python benchmark.py > replay-results.jsonl
```

Default `fixture` mode reads explicit scenario interpretation fields to generate
candidates. It is a deterministic test generator, not a language model. It never
reads `reference/answer_key.jsonl`. Its answer/refusal counts are plumbing outcomes,
not accuracy, false-answer rate, or proof of diagnosis.

To use an ALREADY RUNNING local OpenAI-compatible model server:

```bash
python rca.py --question-id SYN-A595-001-Q-INVESTIGATION --mode local --endpoint http://127.0.0.1:8100/v1 --model qwen3vl-2b
python benchmark.py --mode local --endpoint http://127.0.0.1:8100/v1 --model qwen3vl-2b > model-replay-results.jsonl
```

Replace the endpoint/model with the actual local server configuration. This code
does not install or download weights. It disables HTTP proxies, rejects redirects,
and permits loopback HTTP endpoints only. A model endpoint's own offline behaviour
still requires a disconnected cold-boot test on each board. Malformed JSON,
connection failures and invalid top-level schemas are errors, not refusals.

## Implemented flow

1. `src/ingest.py` reads only the evidence index and current question's allowlist.
   Future investigation/retest records are withheld. Evidence is chunked in 32-line
   windows with original JSON path, inclusive one-based lines and SHA-256.
   Paths, incident IDs, origin and synthetic device identity are checked.
   The known wrong-device retest is quarantined before retrieval.
2. `src/index.py` builds in-memory lexical TF-IDF vectors and token-match rankings
   from eligible chunks only, then fuses ranks. These are NOT semantic embeddings.
   Index persistence and a local learned embedding model remain future work.
3. `src/generate.py` provides the fixture generator and local chat adapter.
   Retrieved logs are supplied as untrusted data. The local model must return JSON.
4. `src/verify.py` checks citation availability, retrieved ranges, source hashes,
   exact quotes, role and explicit structured support. This narrow first policy
   accepts only an exact documented `interpretation_within_simulation` whose
   record has `CONFIRMED_WITHIN_SIMULATION_ONLY`. Free-form causal entailment is
   not implemented. Synthetic conclusions can never be labelled real confirmation.
5. `rca.py` outputs supported candidates or a cause refusal, rejected evidence,
   retrieved chunks, raw candidate objects and host stage timings for inspection.
   `benchmark.py` replays all questions, separating errors from refusals. Raw
   generation and retrieved evidence are diagnostic fields, NOT accepted answers.

The gate can reject a fabricated cause even if its citation points to a real line.
It does not independently prove that an authored investigation is physically true.

## Historical cases and new failures

`--history SYN-A595-003 ...` explicitly opts in historical incident IDs. Shared
source lineage is forbidden (including 001 versus 002). Program, test name and
kind must match the current failure before historical records are eligible.
Missing compatibility metadata excludes the case. Historical candidates remain
hypotheses. This minimum filter is not an engineer-approved similarity policy;
product/revision, conditions and pin mapping need a richer schema and review.

The existing six cases do not provide an independent compatible historical pair
under these rules. Do not misrepresent this staged replay as a held-out new-device
benchmark. The next data step is a separate, engineer-reviewed historical case and
new failure of the same test context with distinct source lineage.

To add a new structured incident, register its failure evidence under `incidents/`,
add its identity and lineage in `scenario_index.json`, map its evidence ID/path in
`evidence_index.json`, and add a question with ONLY currently available evidence to
`replay/questions.jsonl`. The CLI operates on this schema; arbitrary STDF parsing
or automatic raw-log conversion is not implemented. Existing source citations in
observed_failure.json are retained; this version cites JSON evidence lines and
does not independently validate binary STDF offsets or original log excerpts.

## Known limits and next steps

- No actual LLM run, Jetson/RB3 run, offline cold boot or power measurement has been
  performed for this change. The HTTP adapter is tested against a local mock.
- No semantic embedding model, persistent indexes or context-token budgeting yet.
  Top-k retrieval can omit relevant support, resulting in conservative refusals.
- No automatic repair, full retest-effectiveness narrative, or general causal
  ranking. At most three model-ordered documented causes survive the narrow gate.
- 18 related demonstration questions are not the frozen >=50-question evaluation.
  Engineer-reviewed scoring labels, coverage/FAR and independent incidents remain
  required. Fixture replay never reads the answer key or reports accuracy.
- The existing dataset manifest still refers to deleted documentation/scripts;
  this prototype uses an evidence allowlist and runtime source hashes rather than
  claiming that stale manifest has passed full bundle validation.

A useful engineer review can now use actual input, retrieved evidence, rejected
records, and accepted/refused output to discuss the desired evidence policy.
