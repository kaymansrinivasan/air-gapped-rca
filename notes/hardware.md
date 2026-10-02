### RB3 Gen 2 — initial DSP validation

- OS: Ubuntu 24.04.4 LTS, ARM64; observed kernel `6.8.0-1080-qcom`.
- QAIRT packages: 2.46.0-0ubuntu1~bpo24.04.1; QNN-enabled ONNX Runtime environment installed.
- Detected DSP architecture: Hexagon V68.
- QNN platform validator: hardware supported, libraries found.
- DSP calculator unit test: passed.
- Evidence: rca-validation/validator.log.

### RB3 Gen 2 — Qwen3.5-0.8B, observed 2026-10-02

- Unofficial PrismPhi QCS6490 QNN/HTP port, source tag `v0.1.0-rc1`; its 2K asset profile was downloaded and hash-verified (15 files), then installed.
- Source repository documents the upstream Qwen checkpoint at `2fc06364715b967f1860aea9cf38778875588b17`, symmetric per-output-channel W8 for accepted language weights (mixed precision across operators), W8A16 vision, and FP32 CPU embedding lookup. The downloaded 2K model is a QNN artifact, not a GGUF model.
- A launcher displayed `ready: true` with model `qwen3.5-0.8b-q6a-2k`; an HTTP health response reported `backend.vision: QNNExecutionProvider / HTP` and QNN-assigned model components. A short text prompt returned `Ready`. These are setup and smoke-test observations, not a measured quality or throughput result.
- A previous launch froze the GUI; after reboot, an SSH-managed launch succeeded. RAM was approximately 5.2 GiB total with no swap on this board. Memory and recovery behavior require repeatable testing.
- Offline cold boot, end-to-end RAG integration, and matched benchmark runs: pending.

### Jetson Orin Nano Super — TensorRT Edge-LLM preparation, observed 2026-10-02

- Board identified as NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super; about 7.4 GiB RAM, 3.7 GiB swap, and 400 GB available NVMe at inspection. L4T R36 revision 5.2; CUDA compiler 12.6.68; TensorRT system packages 10.3.0.30.
- NVIDIA TensorRT Edge-LLM `v0.10.0` source and submodules checked out. An isolated Python environment generated `aarch64/sm_87` CuTe DSL `fmha,gdn` artifacts. CMake was configured with `EMBEDDED_TARGET=jetson-orin`, `CUDA_CTK_VERSION=12.6`, `ENABLE_CUTE_DSL=gdn`, and explicit `/usr/local/cuda/bin/nvcc`.
- The native build reached 100%; `llm_build --help` and `llm_inference --help` both ran. This establishes that the runtime binaries were built, not that the Qwen model was loaded or executed.
- **Unverified user-specified assumption for the status draft:** Qwen3.5-0.8B has since been loaded and run using TensorRT Edge-LLM. No Jetson model output, engine manifest, precision, or runtime trace was provided in this conversation. Do not promote this assumption to a measured result without capturing those records.
- The intended comparison uses the same upstream Qwen checkpoint and 2,048-token context. The first Jetson engine may use FP16; its actual precision is not yet established. Equal model family/context does not make the RB3 W8/QNN and Jetson TensorRT runs matched-precision experiments.
- Model/engine evidence, decoding settings, offline cold boot, accelerator trace, performance, power, and energy: pending documentation.

