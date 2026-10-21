#!/usr/bin/env python3
"""Pin Qwen2.5-0.5B and record complete-model answers before W8A8 conversion.

This is a reference preparation tool, not an ONNX exporter or accelerator runner.
The deployment target is shared signed W8A8 via ONNX Runtime; it is unvalidated.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
WEIGHTS_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
FILES = (
    "config.json", "generation_config.json", "model.safetensors",
    "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json", "LICENSE",
)
PROMPTS = (
    "What is the capital of Malaysia? Answer in one short sentence.",
    "Historical device A failed continuity and passed after contact cleaning. "
    "New device B failed continuity on pin IO_03. B has no inspection or retest. "
    "Is dirty contact confirmed for B? State whether it is confirmed and suggest one next check.",
    "A device failed a test. No test name, measurements, limits or pin information "
    "is available. What is its root cause? Answer briefly using only these facts.",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, record):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def validate_config(config):
    expected = {
        "model_type": "qwen2", "hidden_size": 896, "num_hidden_layers": 24,
        "num_attention_heads": 14, "num_key_value_heads": 2,
        "vocab_size": 151936, "tie_word_embeddings": True,
    }
    for key, wanted in expected.items():
        if config.get(key) != wanted:
            raise ValueError(f"Unexpected checkpoint configuration: {key}={config.get(key)!r}; expected {wanted!r}")


def verify(checkpoint, manifest):
    if manifest.get("model_id") != MODEL or manifest.get("revision") != REVISION:
        raise ValueError("Existing manifest identifies a different model or revision")
    if set(manifest.get("files", {})) != set(FILES):
        raise ValueError("Manifest does not contain the expected complete file list")
    if manifest["files"]["model.safetensors"]["sha256"] != WEIGHTS_SHA256:
        raise ValueError("Weight digest does not match the pinned upstream checkpoint")
    for name in FILES:
        path = checkpoint / name
        expected = manifest["files"][name]
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Checkpoint integrity failed: {name}; existing files have not been replaced")
    validate_config(json.loads((checkpoint / "config.json").read_text()))


def prepare(root, offline):
    checkpoint = root / "checkpoint"
    manifest_path = root / "source_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        verify(checkpoint, manifest)
        print("Verified existing pinned checkpoint.", flush=True)
        return checkpoint, manifest
    if offline:
        raise ValueError("Offline mode requires a previously prepared source_manifest.json")
    from huggingface_hub import snapshot_download
    print(f"Downloading {MODEL} at {REVISION}; approximately 1 GB, resumable.", flush=True)
    snapshot_download(repo_id=MODEL, revision=REVISION, allow_patterns=list(FILES),
                      local_dir=str(checkpoint), max_workers=2)
    validate_config(json.loads((checkpoint / "config.json").read_text()))
    manifest = {
        "model_id": MODEL, "revision": REVISION,
        "files": {name: {"bytes": (checkpoint / name).stat().st_size,
                          "sha256": digest(checkpoint / name)} for name in FILES},
        "target": {"format": "ONNX", "runtime": "ONNX Runtime 1.27.0",
                   "quantization": "signed W8A8 candidate; scope and quality not yet validated",
                   "jetson_provider": "TensorRTExecutionProvider",
                   "rb3_provider": "QNNExecutionProvider",
                   "accelerator_validation": "not run"},
    }
    if manifest["files"]["model.safetensors"]["sha256"] != WEIGHTS_SHA256:
        raise ValueError("Downloaded weights do not match the upstream SHA256")
    save(manifest_path, manifest)
    print("Checkpoint pinned and file hashes recorded.", flush=True)
    return checkpoint, manifest


def app_jobs():
    """Raw selection probes: no retrieval, pipeline prechecks, or retry is run."""
    repo = Path(__file__).resolve().parents[1]
    source = repo / "src/rca_local/core.py"
    spec = importlib.util.spec_from_file_location("qwen25_app_core", source)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    current = json.loads((repo / "syn_data/product_a_scenarios_v1/case_07/observation.json").read_text())
    current["question"] = "What possible causes should be investigated, and which checks should the engineer perform next?"
    history = core.load_history(repo, current)
    matches = [v for v in history.values()
               if v["records"]["observation"]["case_id"] in ("product_a_case_01", "product_a_case_02")]
    if len(matches) != 2:
        raise ValueError("Expected the two existing continuity history cases")
    jobs = []
    for label, incident, evidence, expected in (
        ("app_continuity", current, matches, "unconfirmed"),
        ("empty_evidence_stress", {**current, "question": "What is the confirmed root cause?",
                             "retrieval_text": "A device failed. No test, pin, measurement, investigation or retest information is available."},
         [], "refuse"),
    ):
        bundle = {"current": incident, "matches": evidence}
        catalog = core.make_catalog(bundle)
        prompt, aliases = core.build_prompt(bundle, catalog)
        jobs.append((label, prompt, (core, aliases, catalog, expected)))
    return jobs


def reference(root, checkpoint, manifest, max_new_tokens, threads):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    out = root / "reference_runs" / str(time.time_ns())
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "started", "model_id": MODEL, "revision": REVISION,
        "source_manifest_sha256": digest(root / "source_manifest.json"),
        "model_safetensors_sha256": manifest["files"]["model.safetensors"]["sha256"],
        "scope": "complete original model CPU reference, before quantization",
        "reference_arithmetic": "FP32", "deployment_target": "shared signed W8A8",
        "accuracy_approved": False, "accelerator_validation": "not run",
        "benchmark_approved": False, "python": sys.version.split()[0],
        "selection_probe_scope": "raw model output before application retry; empty evidence is a stress probe, since the app normally refuses before calling the model",
        "packages": {n: importlib.metadata.version(n) for n in (
            "torch", "transformers", "numpy", "safetensors", "huggingface-hub")},
        "sampling": {"do_sample": False, "repetition_penalty": 1.0,
                     "max_new_tokens": max_new_tokens},
        "results": [],
    }
    save(out / "report.json", report)
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        print("Loading all 24 layers of the original model on CPU for reference answers.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32, attn_implementation="eager",
        ).eval()
        if len(model.model.layers) != 24:
            raise ValueError("Expected all 24 decoder layers")
        eos = model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        generation = GenerationConfig(
            do_sample=False, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
            use_cache=True, eos_token_id=eos, bos_token_id=model.config.bos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
        jobs = [(f"general_{i}", prompt, None) for i, prompt in enumerate(PROMPTS)] + app_jobs()
        for index, (label, prompt, app_check) in enumerate(jobs):
            print(f"Full-model reference prompt {index + 1}/{len(jobs)}: {label}", flush=True)
            text = tokenizer.apply_chat_template(
                [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
            input_ids = inputs["input_ids"][0].tolist()
            start = time.perf_counter()
            with torch.inference_mode():
                logits = model(**inputs, use_cache=False).logits[0, -1].float().numpy().copy()
                if not np.isfinite(logits).all():
                    raise RuntimeError("Nonfinite reference logits")
                output = model.generate(**inputs, generation_config=generation)
            generated = output[0, len(input_ids):].tolist()
            if not generated or generated[0] != int(logits.argmax()):
                raise RuntimeError("Reference forward and greedy generation disagree")
            logit_file = out / f"prompt_{index:02d}_first_logits.npy"
            np.save(logit_file, logits)
            result = {
                "label": label,
                "prompt": prompt, "formatted_prompt": text, "input_ids": input_ids,
                "output_ids": generated,
                "output_text": tokenizer.decode(generated, skip_special_tokens=True),
                "ended_with_eos": generated[-1] in eos_ids,
                "first_logits": logit_file.name, "first_logits_sha256": digest(logit_file),
                "elapsed_seconds_including_reference_forward": time.perf_counter() - start,
            }
            if app_check:
                core, aliases, catalog, expected = app_check
                try:
                    status, _ = core.validate_selection(result["output_text"], aliases, catalog)
                    result["app_validation"] = {"schema_valid": True, "status": status,
                                                "expected_status": expected, "status_matches": status == expected}
                except ValueError as exc:
                    result["app_validation"] = {"schema_valid": False, "expected_status": expected,
                                                "status_matches": False, "error": str(exc)}
            report["results"].append(result)
            save(out / "report.json", report)
            print(result["output_text"], flush=True)
            if "app_validation" in result:
                print("SELECTION CHECK:", label, json.dumps(result["app_validation"]), flush=True)
        report["status"] = "reference_complete_quantization_pending"
        report["raw_selection_checks_passed"] = all(r["app_validation"]["status_matches"]
            for r in report["results"] if "app_validation" in r)
        print("RAW SELECTION CHECKS PASSED:", report["raw_selection_checks_passed"], flush=True)
        return out
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save(out / "report.json", report)
        print("REFERENCE REPORT:", out / "report.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "qwen25-shared-w8a8")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 256 or args.threads < 1:
        parser.error("Use 1..256 output tokens and at least one CPU thread")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.root = args.root.expanduser().resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    checkpoint, manifest = prepare(args.root, args.offline)
    if not args.download_only:
        reference(args.root, checkpoint, manifest, args.max_new_tokens, args.threads)
    print("NEXT: shared W8A8 export and full-model accelerator validation. These have not run.", flush=True)


if __name__ == "__main__":
    main()
