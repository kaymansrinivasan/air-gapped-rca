# Local investigation app

## Engineer workflow

1. Choose the product and failed test; select a known tester or enter another.
   Test program offers local suggestions and also accepts typed values.
2. Ask the engineer's question in the main form.
3. Optionally expand **Add a log or observation** to paste text or load a
   `.txt`/`.log` file into the same field. DUT and failing pins are optional.
4. Click **Investigate failure**. Read the compact **Answer** view; use
   **Evidence** for historical cases and **Details** for limitations/audit data.
   Source buttons open the exact original wording in a dialog.

Only Product_A and Continuity (100), IDD_Static (210), Scan (606) are supported.
These IDs come from the actual scenario records. The UI receives test choices
from the server rather than maintaining a separate numeric list.

No log is required when the product, failed test and question are provided.
Missing measurements, pins, tester and program stay unknown. A missing DUT is
assigned an internal query ID for audit bookkeeping, never presented as a
physical DUT identity or indexed as history. A no-log result displays that its
input context is limited. The model still must use historical choices and may
refuse; field completeness is not proof of diagnostic sufficiency.

For recognized single-DUT Product_A text, **Read details from log** can fill
empty fields for DUT, product, failed test, tester and program. Pasting/loading
also attempts this extraction. Inspect populated fields before submitting.
The original log text is retained. Explicit conflicts between log and form are
rejected on the server before retrieval. Unsupported numeric IDs are not
silently mapped to supported tests. Multiple recognized DUTs/failures are
rejected. This is conservative text extraction, not an arbitrary STDF parser;
unknown formats and contradictions outside these patterns may not be detected.
Failing pins are not guessed. Tester suggestions come from local history and
are not a verified live equipment inventory.

## Setup on the Jetson

Install dependencies while a network or a prepared local package wheelhouse
is available:

```bash
python3 -m venv .venv-llamaindex-jetson
.venv-llamaindex-jetson/bin/python -m pip install -r requirements-jetson.txt
```

The user's existing environment already contains these package versions.
The app requires the existing Chroma database, MiniLM ONNX cache, six case
folders, Qwen engine/tokenizer and working TensorRT executable/plugin.

Start from the project root:

```bash
.venv-llamaindex-jetson/bin/python -m src.rca_local
```

The server binds **127.0.0.1:8765 only**. On the laptop, create an SSH tunnel
(replace JETSON_IP with the actual address):

```bash
ssh -L 8765:127.0.0.1:8765 orin_nano@JETSON_IP
```

Open `http://127.0.0.1:8765` in the laptop browser. Keep that SSH session open.
Use plain SSH rather than a Jetson browser to preserve board RAM. Close local
Jetson applications normally after saving their work. Do not open a public
internet port for this prototype.

## Paths and memory

Optional operator environment variables:

| Variable | Default |
| --- | --- |
| RCA_DATA_ROOT | This checkout; contains syn_data and artifacts/chroma_product_a |
| RCA_BUILD_DIR | ~/TensorRT-Edge-LLM-0.10/build-orin-cuda |
| RCA_ENGINE_DIR | ~/tensorrt-edgellm-workspace/Qwen3.5-0.8B/engines/llm-lowmem |
| RCA_CHAT_SAMPLE | ~/qwen-rca-recheck.json |

If testing in a separate checkout, reuse the existing data and environment:

```bash
export RCA_DATA_ROOT=/home/orin_nano/Documents/airgap-rca
/home/orin_nano/Documents/airgap-rca/.venv-llamaindex-jetson/bin/python -m src.rca_local
```

`RCA_CHAT_SAMPLE` is the successful one-user-message TensorRT response already
demonstrated on the board. It supplies the actual chat wrapper; token counting
includes that wrapper. The runtime output must match the counted formatting.
The model selects IDs only, with 160 reserved output tokens and a 64-token
margin. Over-budget prompts are rejected without truncating evidence.

Retrieval and generation run sequentially in separate child processes, so
ONNX/Chroma memory can be released before TensorRT loads. Only one UI request
runs at a time. Each request currently loads Qwen afresh; this favors memory
control over warm-server latency. A failed/timed-out job is not accepted as
a successful diagnosis.

## Answer construction

The retriever preserves same-product/same-failed-test candidate filtering,
query/history separation and observation freshness checks. It ranks eligible
observations through LlamaIndex's Chroma integration and opens all four linked
stages for the selected cases. This is vector retrieval with explicit filters;
hybrid keyword fusion is not yet implemented.

The model sees current symptoms, the question, historical findings and a
catalog of possible causes/investigation checks. It returns only selection
IDs. Code validates IDs and types, then attaches exact original text, source
record IDs, file/line references and hashes. The model cannot invent a citation
mapping. Causes explicitly naming unmatched historical failing pins are not
offered as current-cause choices. Complete historical records remain visible.

For selected procedures only, exact historical DUT identifiers are replaced
with the current DUT ID. The original wording remains visible, and the UI
labels this adaptation. Historical measurements/results are never rewritten
as current measurements. An engineer must establish comparable conditions.

This constrains fabrication; it does **not** establish semantic relevance,
coverage, correct ranking, causal confirmation or the completeness of the
selected checks. Qwen can still select a poor hypothesis or refuse. Invalid
output is rejected, never silently replaced with a successful-looking answer.
Different symptoms and conflicting histories require engineer review.

## Audit and offline behavior

Every request saves input, retrieved evidence, source hashes, catalog, prompt,
budget, model response, runtime logs and rendered answer under
`artifacts/rca_app/<run_id>/`. These local artifacts are ignored by Git.

No cloud LLM or remote vector store is configured. Chroma telemetry is disabled;
MiniLM downloads are blocked by the app's embedding wrapper. Missing assets
produce an error. The page has no external fonts, scripts or other resources.
These controls are not a network sandbox or an offline certification.

For an unplug/reboot test, start the app and use a local Jetson browser after
disconnecting networks, or use a separately defined isolated LAN with no
internet route. A laptop SSH session itself needs a network connection.
Record exactly which offline condition was tested.

## Validation status

Software tests cover source linkage, malformed selections, the observed bad
draft, duplicate/unknown IDs, invalid status/type, incoming-DUT leakage,
historical-DUT procedure adaptation, unrelated-pin candidates, and local HTTP
input handling. HTTP tests substitute a controlled pipeline result; they do
not run the GPU. Python compilation and browser JavaScript syntax are checked.

On 2026-10-06 the prior UI completed a new synthetic IDD request on the Jetson:
Product_A-L06-W02-D008. The displayed answer selected high-current Case 03
hypotheses and checks with citations, with Case 04 visible as separate history.
This is one functional run, not a quality benchmark.

This minimal UI revision is for review and still needs a Jetson run. Test both
log and no-log inputs, all three test types and unsupported questions. Preserve
prior invalid outputs; rejection is not the same as a correct evidence-based
refusal. The >=50-question evaluation, engineer review, memory/latency/energy
measurements and documented offline test remain required for project acceptance.

## Git workflow

The original app and ID correction were merged via PR #3 at `3a5ba46`.
Review this revision on a new `feature/minimal-investigation-ui` branch. Keep commits limited to source,
tests, dependency declarations and documentation. Do not stage a virtual
environment, model weights, engine, cache or generated run directory.
Commit tested milestones; merge into main after board acceptance.

The copied Jetson folder may contain untracked scripts from this conversation.
Check its Git status before switching branches. Do not force-checkout over
those files; preserve them or use a separate checkout and RCA_DATA_ROOT.

### Selection retry (2026-10-06)

A Jetson continuity request completed normally but selected three cause IDs and
four check IDs, exceeding the two-per-array contract. This is an output-contract
failure, not an out-of-memory or truncation error.

The prompt now separates cause and check choices and repeats the output rules
after the evidence. A selection-validation failure permits one fresh generation
with the same evidence and a format reminder. No selected IDs are silently removed.
Runtime, budget, and source-integrity failures still stop the request. Both attempts
use the existing token budget and strict source/type validation. A second invalid
selection is rejected. This may add one inference run and must be included in
end-to-end latency and energy measurements.

The first attempt remains in the run directory; a retry is stored under `retry_1/`.
Each completed validation writes `validation.json`; `answer.json` records
`generation_attempts`. Inspect both attempts when debugging; the top-level
`model.json` always holds the first response.

Validation: 31 automated tests passed in the development environment, including
mocked retry acceptance, repeated rejection, wrong IDs/types, source changes and
runtime/budget failures. Jetson model behavior still needs a real rerun.

### Conversational ATE chat (review build, 2026-10-06)

The excerpt-selector chat was replaced after a live Jetson follow-up ('give me
one line') failed the JSON-output contract. Chat now requests ordinary prose
from Qwen. The engineer can ask for an overview, explanation, short answer or
next checks. The latest successful assistant reply is included as text so
follow-ups can refer to it. JSON selection remains only in the separate initial
investigation flow.

The UI shows a spinning indicator with actual elapsed time while waiting. It
shows the completed response as text, with expandable source references below.
Enter sends a message; Shift+Enter starts a new line. Model HTML is never executed.
This file-based runtime does not stream tokens; the spinner stays visible until
the full response or an error arrives. Reduced-motion preferences are respected.

Conversation scope is prompted to stay within ATE testing and supplied case
evidence, decline unrelated requests and acknowledge missing information. This
is a model instruction, not a proven domain classifier. Unlike the former
extractive mode, conversational wording is generated by the model and can be
wrong. The UI labels it as AI explanation for engineer review. Available E-number
source references and source hashes are checked; these checks do not establish
entailment, scope compliance or diagnostic correctness. A reply without explicit
references is recorded as such. Its expandable evidence is labelled context,
not falsely attributed as model citations. Synthetic/unreviewed provenance stays
visible and historical results must not be represented as new measurements.

The current incident stays fixed during chat. Submit a new investigation for a
new DUT, changed measurements or new failure information. Rejected replies are
not used as conversational memory. Up to two successful exchanges enter each
prompt; only the older one can be omitted if the engine budget requires it. The
latest reply and all evidence must fit or generation stops without truncation.
The actual selected history count is saved in budget.json and model.json. Chat
reserves 256 output tokens plus 64 margin, within the existing engine capacity.
Initial investigation continues to reserve 160 output tokens.

Full transcripts, prompts, outputs and errors remain in
`artifacts/rca_app/<run_id>/chat/<turn_id>/`. Each chat supports ten turns with
questions up to 700 characters. One corrective retry is allowed for empty,
structured or invalid-reference replies; runtime, source and token-budget errors
stop the request. Include every attempt in latency and energy measurements.
The localhost/CSRF checks and shared single-job lock remain active. No new model
assets or network requests are introduced.

Development validation: 47 automated tests passed, including acceptance of
plain one-line replies, actual previous-reply context, refusal text, source
checks and prompt-budget fallback. Browser JavaScript syntax checked. These are
mocked-runtime tests; live Qwen quality, scope compliance and UI interaction
still require Jetson testing. Apply conversational-chat.patch after the previous
ate-case-chat.patch. Restart the server and hard-refresh the browser.

On-device checks: ask for an overview and next checks; ask 'give me one line';
ask 'why use a reference die?'; ask an unrelated holiday question. Confirm the
spinner, readable prose, reference controls, shorter reply and scoped refusal.
Record failures instead of treating source-reference checks as accuracy proof.

### Current-DUT chat correction (2026-10-07)

A live IDD query for D010 returned a proposed action targeting historical D100
and phrased a DUT-related explanation too strongly. The chat prompt now explicitly
separates failed measurements from established causes and directs new actions
to the current DUT. Targeted language checks reject the observed historical-DUT
action and affirmative causal-overclaim patterns, with the existing single retry.
Known-bad prior replies are retained in the audit but excluded from new prompts.
These heuristic checks do not constitute general semantic verification and may
miss other wording or reject some legitimate discussion. Evaluate natural-language
claims with an engineer; source references alone are insufficient.

Chat source links are grouped by source file/hash, labelled with case and stage,
and laid out on separate lines. Each dialog retains the group's evidence excerpts.
No diagnostic text or historical record is silently rewritten.

Development validation: 52 automated tests passed, including regressions for the
observed D100 instruction, causal overclaim, preserved historical descriptions,
retry and exclusion of known-bad previous replies. On-device recheck pending.
