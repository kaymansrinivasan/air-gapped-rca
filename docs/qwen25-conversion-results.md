# Qwen2.5 conversion experiment: quality gate failed

This records CPU conversion experiments observed in the assistant workspace on
2026-10-22. It is not a GPU/NPU benchmark or a deployable model release.

The user's checkpoint download succeeded. The downloaded branch provides the
pinned checkpoint/reference preparation script. A successful download does not
establish shared W8A8 support.

## Fixed checkpoint

- Model: Qwen/Qwen2.5-0.5B-Instruct
- Revision: 7ae557604adf67be50417f59c2c2f167def9a775
- Original safetensors SHA256:
  fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe
- Local comparison: PyTorch 2.6.0 CPU, Transformers 4.51.3,
  ONNX 1.20.1, ONNX Runtime 1.27.0, NumPy 1.26.4.

## What was actually tested

An experimental standard ONNX decoder contained all 24 layers and the complete
vocabulary head in one graph, with fixed 2048-token cache dimensions. Embedding
lookup, rotary inputs, cache transport and greedy sampling remained explicit
host work. There was no accelerator execution.

The original FP32 ONNX export was compared with the pinned Hugging Face model
on three prompts and up to eight teacher-forced greedy prediction steps per
prompt. All compared greedy tokens matched.

| Prompt | Original export relative L2 | Maximum absolute error |
| --- | ---: | ---: |
| Malaysia capital | 0.0000013488 | 0.0000543594 |
| Continuity, A versus B | 0.0000021120 | 0.0000333786 |
| Failure with insufficient information | 0.0000020496 | 0.0000476837 |

These measurements establish limited numerical export parity, not RCA accuracy.

## Two failed W8A8 candidates

Candidate 1 used signed symmetric per-tensor INT8 weights and activation QDQ
after every floating-point graph operation. Calibration used eight complete
test-engineering prompt trajectories. Quantization scales came from absolute
maxima; masked-score observers floored artificial negative mask values at -32.
This masking treatment was experimental, not an established deployment recipe.
Integer shape constants were not quantized. Host embedding lookup remained FP32.

Candidate 2 additionally applied an orthogonal residual basis change:
DCT(7) x Hadamard(128), deterministic row signs with seed 25, with RMSNorm
weights folded into adjacent projections. Its FP32 calculation also matched
the original checkpoint in the same checks: relative L2 errors
0.0000019198, 0.0000030488 and 0.0000021432, and matching greedy tokens.

The rotation reduced one observed layer-8 hidden absolute maximum from
1702.5057 to 81.4077. This supports the outlier diagnosis but does not establish
that outliers were the only cause of the quantized failure.

Both full INT8 models generated incoherent output on all three smoke prompts.

| Prompt | Candidate 1 first-logit relative L2 | Candidate 2 first-logit relative L2 |
| --- | ---: | ---: |
| Malaysia capital | 5.937718 | 1.680865 |
| Continuity, A versus B | 8.483408 | 2.343824 |
| Failure with insufficient information | 8.249866 | 3.437700 |

All first-token matches failed. Candidate 1 produced repeated symbols;
candidate 2 produced unrelated multilingual fragments. Neither answered the
capital question correctly.

Candidate 2 artifact fingerprints recorded in the tool output:
- ONNX: cdbb878325795bfa1267c72394131cbc6f395164b43555c2f08167626eff4c25
- Weight data: 176bacfc1472b2aefdbe4a2840cf22a0c947030dabb808019a402b7a348ccb16

The observed results reject these particular conversion recipes. They do not
prove that Qwen2.5 or shared W8A8 across GPU and NPU is impossible.

## Artifact limitation and current decision

Automated workspace maintenance removed the experimental local scripts,
model exports and raw reports before they could be committed. This note
preserves measurements present in the conversation's tool outputs; the exact
experiment is not a reproducible release in this branch. The user's WSL clone
and downloaded checkpoint were not touched.

Do not copy either failed candidate to the boards, integrate it into the UI,
or describe it as a working shared W8A8 model. No GPU/NPU support, native
arithmetic parity, stability or application accuracy has been demonstrated.

The next development requirement is an accuracy-preserving, explicitly
documented quantization recipe. It must pass complete-model answer checks
before hardware validation. Repeating these min/max recipes or recompiling
them on the boards does not resolve the measured quality failure.

The existing Qwen3.5 deployments and UI remain separate from this experiment.
