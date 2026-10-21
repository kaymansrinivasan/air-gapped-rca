# Qwen2.5 shared W8A8 experiment

This branch targets Qwen2.5-0.5B-Instruct on Jetson Orin Nano GPU and QCS6490 NPU
with the same signed W8A8 ONNX model, weights, quantization parameters, tokenizer,
prompt token IDs and ONNX Runtime version (1.27.0). TensorRT EP and QNN EP are the
device-specific backends. This is a target, not an achieved result.

## What is implemented

`scripts/prepare_qwen25_reference.py` downloads one pinned original checkpoint,
verifies its upstream weight SHA256, records hashes for tokenizer and model files,
and runs all 24 layers through three short general prompts plus two application
checks. The application checks use the existing `core.build_prompt` and
`core.validate_selection` with the two continuity histories and the query-only
case; they also probe refusal when history is empty. These examine raw generation
before the app's existing retry. The actual app refuses empty eligible history
before calling the model; the empty-history probe is an additional model stress
test, not a claim to reproduce that app path. Retrieval and the complete UI
pipeline are not measured. It saves the exact
prompt template, input/output token IDs, first-token logits, dependency versions
and responses. Existing manifests are verified before reuse. Download interruption
can be resumed by rerunning the command. Reference reports use new directories.

Checkpoint: `Qwen/Qwen2.5-0.5B-Instruct`

Revision: `7ae557604adf67be50417f59c2c2f167def9a775`

Model weight SHA256: `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`

The original BF16 checkpoint is calculated in FP32 on the laptop solely to record
reference behavior. This does not change the target to FP32, emulate W8A8 on the
GPU, or qualify a CPU run as a GPU/NPU benchmark.

## Start in laptop WSL

Use the existing export environment, which already has PyTorch, Transformers,
NumPy, safetensors and huggingface_hub. This script does not install packages or
modify that environment. From this branch's checkout:

```bash
~/qwen-shared-w8a8/venv/bin/python -u scripts/prepare_qwen25_reference.py
```

Default output: `~/qwen25-shared-w8a8/`. This is separate from the Qwen3.5 files.
Download is about 1 GB. No GPU or NPU is needed for this preparation step.
Use `--offline` for a repeat using an already verified checkpoint.

## Remaining gates

1. Export the complete model and verify its ONNX reference outputs against the
   saved original-model logits and sequential generation. Export is not yet
   implemented on this branch.
2. Produce one W8A8 candidate, recording exactly which operations are quantized
   and any floating point exceptions. Assess answers and numerical error; a file
   called INT8 alone is not evidence of accurate quantization.
3. Execute that same candidate on both accelerators with CPU fallback disabled.
   Audit provider placement and arithmetic precision. QDQ graph placement alone
   is insufficient evidence of native INT8 matrix calculations.
4. Validate stability and held-out RCA answers, then integrate with the UI and
   measure identical workloads on both boards.

Stop at a failed gate and retain its report. Five reference prompts do not
establish RCA accuracy. This tool performs neither quantization nor accelerator
benchmarking, and does not claim Qwen2.5 will satisfy all the gates.

The existing Qwen3.5 branches and deployed UI are not modified by this tool.
The retrieval database and evidence format can be reused, but the LLM adapter,
prompt template, context budgeting and answer quality need checking after export.
