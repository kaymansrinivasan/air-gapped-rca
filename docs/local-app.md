# Local investigation app

## Engineer workflow

1. Open the app in a laptop browser through an SSH tunnel.
2. Enter the current DUT, product and failed test. Add known failing pins,
   tester and test program.
3. Paste a single-DUT observation, or load a local `.txt`/`.log` excerpt.
4. Ask a failure-triage question and click **Investigate failure**.
5. Read unconfirmed hypotheses, proposed checks, source references and full
   historical evidence. A rejected answer is visibly rejected; historical
   evidence remains available separately for engineer inspection.

Engineers do not edit Python or type JSON. An operator starts the server once.
Only Product_A and the current Continuity (100), IDD_Static (210) and Scan
(606) prototype are supported. Test/product/DUT fields are engineer-confirmed
input; arbitrary STDF files, whole wafers and unknown log formats are not
automatically parsed. The form supports new DUTs, not just Case 07.

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

No new Jetson run or new diagnostic accuracy result is claimed for this code.
Before considering merge/release: run Case 07 on the board, inspect rejected
and unanswerable cases, check all three test types, measure peak RAM and latency,
complete the >=50-question evaluation and engineer review, and perform the
documented offline test. Preserve the prior rejected drafts as regression cases.

## Git workflow

Develop on `feature/llamaindex-jetson-ui`. Keep commits limited to source,
tests, dependency declarations and documentation. Do not stage a virtual
environment, model weights, engine, cache or generated run directory.
Commit tested milestones; merge into main after board acceptance.

The copied Jetson folder may contain untracked scripts from this conversation.
Check its Git status before switching branches. Do not force-checkout over
those files; preserve them or use a separate checkout and RCA_DATA_ROOT.
