"""Short-lived retrieval and generation processes to reduce Jetson RAM overlap."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .core import EvidenceError, load_history, require


def settings():
    root = Path(os.environ.get("RCA_DATA_ROOT", Path(__file__).resolve().parents[2])).resolve()
    home = Path.home()
    build = Path(os.environ.get("RCA_BUILD_DIR", home / "TensorRT-Edge-LLM-0.10/build-orin-cuda"))
    engine = Path(os.environ.get("RCA_ENGINE_DIR", home / "tensorrt-edgellm-workspace/Qwen3.5-0.8B/engines/llm-lowmem"))
    sample = Path(os.environ.get("RCA_CHAT_SAMPLE", home / "qwen-rca-recheck.json"))
    return root, build, engine, sample


def retrieve(current):
    root, _, _, _ = settings()
    cases = load_history(root, current)
    eligible = {dut: c for dut, c in cases.items()
                if c["records"]["observation"]["original_chunk"]["failed_test"] == current["test_result"]["test_number"]
                and c["records"]["observation"]["original_chunk"].get("product") == current["context"]["product"]}
    bundle = {"current": current, "matches": [], "candidate_count": len(eligible),
              "method": "llamaindex_chroma_same_product_same_failed_test", "score_is_cause_probability": False}
    if not eligible:
        return bundle
    db = root / "artifacts/chroma_product_a"
    require((db / "chroma.sqlite3").is_file(), "Chroma database missing. Build or copy the local index first.")
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core.vector_stores.types import VectorStoreQuery, MetadataFilter, MetadataFilters, FilterOperator

    class LocalMiniLM(ONNXMiniLM_L6_V2):
        def _download(self, *args, **kwargs):
            raise EvidenceError("MiniLM assets are missing. Provision the cached ONNX model before using the offline app.")

    client = chromadb.PersistentClient(path=str(db), settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection("product_a_observations", embedding_function=None)
    stored = collection.get(ids=list(eligible), include=["documents", "metadatas"])
    docs = dict(zip(stored["ids"], stored["documents"]))
    metas = dict(zip(stored["ids"], stored["metadatas"]))
    for dut, case in eligible.items():
        require(docs.get(dut) == case["records"]["observation"]["retrieval_text"], "Indexed observation differs from historical source.")
        require((metas.get(dut) or {}).get("dut_id") == dut, "Indexed DUT metadata is missing.")
    embedder = LocalMiniLM(preferred_providers=["CPUExecutionProvider"])
    vector = [float(x) for x in embedder([current["retrieval_text"]])[0]]
    results = ChromaVectorStore(chroma_collection=collection).query(VectorStoreQuery(
        query_embedding=vector, similarity_top_k=min(2, len(eligible)),
        filters=MetadataFilters(filters=[MetadataFilter(key="dut_id", value=list(eligible), operator=FilterOperator.IN)]),
    ))
    ids, scores = results.ids or [], results.similarities or []
    require(bool(ids) and len(ids) == len(scores) and len(ids) == len(set(ids)), "Invalid or empty search result.")
    for rank, (dut, score) in enumerate(zip(ids, scores), 1):
        require(dut in eligible and math.isfinite(float(score)) and 0 < float(score) <= 1, "Invalid search score or case.")
        bundle["matches"].append({**eligible[dut], "rank": rank, "vector_distance": -math.log(float(score))})
    bundle["chroma_records"] = collection.count()
    return bundle


def select_prompt(job, template, old_text, tokenizer, limits, output_tokens, margin):
    """Select a complete prompt that fits; never clip evidence or the latest reply."""
    variants = job.get("prompt_options", [{"prompt": job["prompt"], "history_turns": None}])
    require(isinstance(variants, list) and 1 <= len(variants) <= 3, "Invalid prompt options.")
    for variant in variants:
        prompt = variant["prompt"]
        require(isinstance(prompt, str), "Prompt must be text.")
        formatted = template.replace(old_text, prompt, 1)
        count = len(tokenizer.encode(formatted, add_special_tokens=False).ids)
        if count <= limits["max_input_len"] and count + output_tokens + margin <= limits["max_kv_cache_capacity"]:
            return prompt, formatted, count, variant.get("history_turns")
    raise EvidenceError(f"Conversation exceeds the engine budget: {count} input + {output_tokens} output + {margin} margin. Start a new investigation with a shorter log or question. No evidence was truncated.")


def generate(job, run_dir):
    backend = os.environ.get("RCA_BACKEND", "jetson")
    require(backend in ("jetson", "rb3"), "Unknown RCA_BACKEND.")
    if backend == "rb3":
        from .rb3 import generate_rb3
        return generate_rb3(job, run_dir)
    _, build, engine, sample_path = settings()
    from tokenizers import Tokenizer
    from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
    from llama_index.core.llms.callbacks import llm_completion_callback

    require(sample_path.is_file(), "A successful TensorRT chat sample is required (RCA_CHAT_SAMPLE).")
    sample = json.loads(sample_path.read_text())["responses"][0]
    require(len(sample["messages"]) == 1 and sample["messages"][0]["role"] == "user", "Chat sample must have one user message.")
    old_text = sample["messages"][0]["content"][0]["text"]
    template = sample["formatted_complete_request"]
    require(bool(old_text) and template.count(old_text) == 1, "Cannot recover chat template from sample.")
    tokenizer = Tokenizer.from_file(str(engine / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    limits = json.loads((engine / "config.json").read_text())["builder_config"]
    output_tokens, margin = job.get("output_tokens", 160), 64
    require(type(output_tokens) is int and 1 <= output_tokens <= 384, "Invalid output token budget.")
    prompt, formatted, count, history_turns = select_prompt(job, template, old_text, tokenizer, limits, output_tokens, margin)
    del tokenizer
    (run_dir / "budget.json").write_text(json.dumps({"input_tokens": count, "output_tokens": output_tokens, "margin": margin, "limits": limits, "history_turns_used": history_turns}, indent=2))

    class JetsonQwen(CustomLLM):
        @property
        def metadata(self):
            return LLMMetadata(model_name="Qwen3.5-0.8B-TensorRT", context_window=limits["max_kv_cache_capacity"], num_output=output_tokens)

        @llm_completion_callback()
        def complete(self, prompt: str, formatted: bool = False, **kwargs):
            payload = {"batch_size": 1, "temperature": 0.0, "max_generate_length": output_tokens,
                       "apply_chat_template": True, "enable_thinking": False,
                       "requests": [{"messages": [{"role": "user", "content": prompt}]}]}
            request, response = run_dir / "request.json", run_dir / "response.json"
            request.write_text(json.dumps(payload, indent=2))
            env = os.environ.copy()
            env["EDGELLM_PLUGIN_PATH"] = str(build / "libNvInfer_edgellm_plugin.so")
            binary = build / "examples/llm/llm_inference"
            require(binary.is_file() and Path(env["EDGELLM_PLUGIN_PATH"]).is_file(), "TensorRT executable or plugin is missing; check RCA_BUILD_DIR.")
            with (run_dir / "runtime.log").open("w") as log:
                run = subprocess.run([str(binary), "--engineDir", str(engine), "--inputFile", str(request), "--outputFile", str(response)],
                                     env=env, cwd=str(build.parent), stdout=log, stderr=subprocess.STDOUT, timeout=600)
            require(run.returncode == 0 and response.is_file(), "TensorRT failed; see the saved runtime.log.")
            raw = json.loads(response.read_text())
            require(len(raw["responses"]) == 1, "Unexpected response count.")
            return CompletionResponse(text=raw["responses"][0]["output_text"].strip(), raw=raw)

        def stream_complete(self, prompt, **kwargs):
            raise NotImplementedError("File runtime uses complete responses.")

    result = JetsonQwen().complete(prompt)
    item = result.raw["responses"][0]
    require(item.get("finish_reason") == "end-of-sequence", "Model output was incomplete; no answer accepted.")
    require(item.get("formatted_complete_request") == formatted, "Runtime formatting differs from counted prompt; no answer accepted.")
    return {"text": result.text, "input_tokens": count, "finish_reason": item["finish_reason"], "history_turns_used": history_turns}


def main():
    mode, input_file, output_file = sys.argv[1:]
    target = Path(output_file)
    try:
        data = json.loads(Path(input_file).read_text())
        result = retrieve(data) if mode == "retrieve" else generate(data, target.parent)
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        target.with_suffix(".error.json").write_text(json.dumps({"error": str(exc)}))
        raise


if __name__ == "__main__":
    main()
