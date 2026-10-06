# Jetson integration progress — 2026-10-05

## Demonstrated on the user's Jetson Orin Nano

- Python 3.10.12, ARM64, 7.4 GiB RAM.
- LlamaIndex core 0.14.25, Chroma integration 0.6.0, Chroma 1.5.9.
- Existing collection: 3,920 synthetic Product_A DUT observations.
- Case 07 retrieval through LlamaIndex: Case 01 (distance 0.5303),
  Case 02 (0.6455), with four linked records each. No records inserted.
- Qwen3.5-0.8B via TensorRT-Edge-LLM 0.10.0 and LlamaIndex CustomLLM.
- Runtime: `~/TensorRT-Edge-LLM-0.10/build-orin-cuda/examples/llm/llm_inference`.
- Plugin: `~/TensorRT-Edge-LLM-0.10/build-orin-cuda/libNvInfer_edgellm_plugin.so`.
- Engine: `~/tensorrt-edgellm-workspace/Qwen3.5-0.8B/engines/llm-lowmem`.
- Engine limits: max input 2,048; KV capacity 2,048; batch size 1.
- The latest full-evidence prompt used 1,371 input tokens, 384 reserved
  output tokens and a 64-token safety margin.

## Failures retained as evidence, not successes

The initial runtime attempt lacked EDGELLM_PLUGIN_PATH and ran out of
memory with Firefox/VS Code consuming RAM. Correcting the path and using
SSH with those applications closed allowed inference to succeed.

Two free-text RCA generations were rejected:

1. The model copied a placeholder cause and returned limitations as a string.
2. The model cited an observation as support for an investigation cause and
   proposed retesting historical DUT D005. Validation rejected this output.

Retrieval and model execution working does not establish RCA accuracy.
Historical scenarios are synthetic; simulated investigation and retest
outcomes are not physically tested and are not reviewed. Engineer guidance
on possible causes/procedures does not validate the invented outcomes.

## Next implementation milestone

Use source-bound selectable evidence entries so the model cannot create
citation assignments. Provide a local browser interface for questions and
failure observations. Validate selections and show uncertainty/refusal to
the engineer. New code still requires a Jetson run and engineer assessment.
The 50-question evaluation, false-answer measurement, unplug/reboot test,
RB3 integration and board comparison are outstanding.
