# RB3 Gen 2 integration checkpoint

The shared `src.rca_local` app uses the same local Chroma/LlamaIndex retrieval,
source verifier, browser UI, and 50 frozen Product_A questions on both boards.
Set `RCA_BACKEND=rb3` to call the installed Qwen3.5-0.8B 2K QNN/HTP server
at `http://127.0.0.1:8000/v1/chat/completions`. The request remains on the
board; the server binds to loopback. The RB3 runtime is an independent model
preparation with different precision from the Jetson TensorRT engine.

The adapter uses the installed RB3 tokenizer and the server's single-user
chat wrapper to count tokens. It refuses if the prompt plus output budget and
64-token margin exceed 2048 tokens, if the server silently fits/truncates
messages or clamps output, if token counts differ, if generation stops early,
or if request metrics do not report full QNN HTP language prefill/decode with
fallback disabled. Exact raw response and budget are saved per run. This is a
runtime provenance check, not an independent silicon utilization measurement.
The original retrieved history stays in the audit; IDD direction-incompatible
cases cannot be offered to the model as causes/checks.

## RB3 provisioning

1. Keep the tested Qwen server running with `install-2k/launch.sh`. Check
   `/health` and a short text completion first. If a key is configured, set
   `QWEN35_API_KEY` in the RCA shell too. Do not print the key in logs.
2. Clone the feature branch of `kaymansrinivasan/air-gapped-rca` on RB3 and
   create an independent Python 3.12 virtual environment for the RCA app.
   Install `requirements-rb3.txt` while network/provisioned packages are
   available. The Qwen runtime keeps its own Python and model assets.
3. Put the Jetson-generated `artifacts/chroma_product_a` index under the RB3
   data root. The collection must contain the same 3,920 observations and
   its historical documents must exactly match the checked-in scenario
   records. Provision the 79 MB ONNX MiniLM cache under the RB3 account's
   Chroma cache before claiming an offline run; retrieval refuses download.
4. Set `RCA_DATA_ROOT` to this checkout and `RCA_BACKEND=rb3`. Run one case,
   inspect `answer.json`, `budget.json`, `response.json`, and
   `verification.json` under `artifacts/rca_app/<run_id>/`. Check `finish_reason`
   and `qwen35_metrics` in the raw response.
5. Run `python -m src.rca_local` for the same UI. The web server binds to
   `127.0.0.1:8765`; use SSH forwarding from a laptop.

For the fixed provisional comparison (not engineer-reviewed accuracy):

```bash
export RCA_DATA_ROOT="$PWD" RCA_BACKEND=rb3
.venv-rca/bin/python benchmark.py --run --backend rb3 \
  --manifest eval/product_a_v1_50.json \
  --gold eval/product_a_v1_50_proxy_gold.json --allow-proxy-gold \
  --data-root "$RCA_DATA_ROOT" \
  --output "$RCA_DATA_ROOT/artifacts/evaluation/rb3_proxy_v1.json"
```

Confirm manifest SHA-256
`06c67914b489811db1fdc96dad66bdd56a37028acb659da86505368d92b96d9a`
on both boards. Preserve the Jetson v2 result and per-run artifacts. Record
checkpoint revision, quantization, accelerator trace, request tokens,
latency, and power separately. Chat remains experimental; neither an AI
proxy score nor the runtime's backend declaration establishes diagnostic
correctness or actual electrical power.
