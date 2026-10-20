#!/usr/bin/env python3
"""Bounded QNN AOT loading experiment for the existing shared Qwen bundle.

Compile each unchanged graph in a separate process, then run the existing
generation loop from external QNN contexts with resource sharing requested.
Requires ORT 1.27.0 and onnxruntime-qnn 2.3.0. No accuracy approval is implied.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_runner(path):
    spec = importlib.util.spec_from_file_location("_qwen_aot_runner", path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    if runner.ort.__version__ != "1.27.0":
        raise RuntimeError("Expected ONNX Runtime 1.27.0")
    if runner.options().graph_optimization_level != runner.ort.GraphOptimizationLevel.ORT_ENABLE_BASIC:
        raise RuntimeError("Expected the existing BASIC-optimization runner")
    return runner


def identity(args):
    import onnxruntime_qnn as qnn
    version = importlib.metadata.version("onnxruntime-qnn")
    if version != "2.3.0":
        raise RuntimeError(f"This experiment targets QNN plugin 2.3.0; found {version}")
    return {"schema": 1, "bundle": str(args.bundle),
            "manifest_sha256": sha(args.bundle / "manifest.json"),
            "runner_sha256": sha(args.runner), "ort": "1.27.0", "qnn": version,
            "backend_sha256": sha(qnn.get_qnn_htp_path()),
            "plugin_sha256": sha(qnn.get_library_path())}


def selected_specs(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    specs = manifest["layers"] + manifest["heads"]
    names = [s["name"] for s in specs]
    if len(manifest["layers"]) != 24 or len(manifest["heads"]) != 8 or len(set(names)) != 32:
        raise RuntimeError("Expected all 24 decoder layers and 8 vocabulary heads")
    if any(Path(name).name != name or name in (".", "..") for name in names):
        raise RuntimeError("Invalid graph name in manifest")
    return manifest, specs


def verify_sources(bundle, manifest):
    for name, expected in manifest["files"].items():
        if name.startswith("fp32/"):
            continue
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or sha(path) != expected:
            raise RuntimeError(f"Source checksum mismatch: {name}")


def cache_record(cache, name):
    folder = cache / name
    record = json.loads((folder / "ready.json").read_text())
    context = folder / "model_ctx.onnx"
    if record.get("schema") != 1 or "model_ctx.onnx" not in record["files"]:
        raise RuntimeError(f"Invalid compiled cache record: {name}")
    if not any(p.endswith(".bin") for p in record["files"]):
        raise RuntimeError(f"Missing external QNN binary: {name}")
    for relative, expected in record["files"].items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != expected:
            raise RuntimeError(f"Compiled cache checksum mismatch: {name}/{relative}")
    return context


def qnn_options(runner, context=None, name=None, prepare=False):
    import onnxruntime_qnn as qnn
    options = runner.options()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("ep.context_enable", "1" if prepare else "0")
    # Embedded contexts conflict with resource sharing in this plugin version.
    options.add_session_config_entry("ep.context_embed_mode", "0")
    if prepare:
        options.add_session_config_entry("ep.context_file_path", str(context))
        options.add_session_config_entry("ep.context_node_name_prefix", name + "_")
    devices = [d for d in runner.ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    if not devices:
        raise RuntimeError("QNN devices unavailable")
    values = {"backend_path": qnn.get_qnn_htp_path(), "offload_graph_io_quantization": "0"}
    values.update({"enable_htp_prepare_only": "1"} if prepare else
                  {"htp_share_resource_optimization": "1"})
    options.add_provider_for_devices(devices, values)
    return options


def prepare_graph(args, runner, name):
    import onnxruntime_qnn as qnn
    runner.ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
    folder = args.cache / name
    folder.mkdir(exist_ok=False)
    context = folder / "model_ctx.onnx"
    options = qnn_options(runner, context=context, name=name, prepare=True)
    session = runner.ort.InferenceSession(str(args.bundle / "int8" / (name + ".onnx")),
                                          sess_options=options, enable_fallback=False)
    session.disable_fallback()
    del session
    gc.collect()
    files = {str(p.relative_to(folder)): sha(p) for p in folder.rglob("*") if p.is_file()}
    if not context.is_file() or not any(p.endswith(".bin") for p in files):
        raise RuntimeError(f"QNN did not generate an external context for {name}")
    write_json(folder / "ready.json", {"schema": 1, "files": files})


class AotOrt:
    def __init__(self, original, cache, bundle, names):
        self.original, self.cache, self.bundle = original, cache, bundle
        self.names, self.loaded = set(names), []
        self.report_path = None

    def __getattr__(self, key):
        return getattr(self.original, key)

    def InferenceSession(self, model, *positional, **keywords):
        source = Path(model).resolve()
        if source.parent != self.bundle / "int8" or source.stem not in self.names:
            raise RuntimeError(f"Unexpected model requested by runner: {source}")
        options = keywords["sess_options"]
        self.report_path = Path(options.profile_file_prefix).parent / "run_report.json"
        if options.get_session_config_entry("session.disable_cpu_ep_fallback") != "1":
            raise RuntimeError("Model CPU fallback must remain disabled")
        context = cache_record(self.cache, source.stem)
        print("AOT LOAD:", source.stem, flush=True)
        result = self.original.InferenceSession(str(context), *positional, **keywords)
        result.disable_fallback()
        self.loaded.append(source.stem)
        return result


def run_cached(args, runner, specs):
    # The original generation loop still supplies tokens, states and all heads.
    # Configure options before its provider registration; add_provider_for_devices
    # in the runner supplies backend_path and offload_graph_io_quantization only.
    original_options = runner.options

    def shared_options():
        options = original_options()
        options.add_session_config_entry("ep.context_enable", "0")
        options.add_session_config_entry("ep.context_embed_mode", "0")
        options.add_session_config_entry("ep.qnnexecutionprovider.htp_share_resource_optimization", "1")
        return options

    runner.options = shared_options
    owner = AotOrt(runner.ort, args.cache, args.bundle, [s["name"] for s in specs])
    runner.ort = owner
    args.provider, args.precision = "qnn", "int8"
    primary_error = False
    try:
        runner.run(args)
    except BaseException:
        primary_error = True
        raise
    finally:
        if owner.report_path and owner.report_path.is_file():
            try:
                record = json.loads(owner.report_path.read_text())
                if record.get("error"):
                    record["status"] = "failed"
                record["qnn_loading_experiment"] = {
                    "cache": str(args.cache), "mode": "external AOT contexts",
                    "resource_sharing_requested": True, "loaded_graphs": owner.loaded,
                    "source_graphs_weights_scales": "unchanged",
                    "accuracy_acceptance": "not established"}
                write_json(owner.report_path, record)
                print("AOT-SHARED STATUS:", record["status"], flush=True)
                print("AOT-SHARED REPORT:", owner.report_path, flush=True)
            except Exception as exc:
                if not primary_error:
                    raise
                print("SECONDARY REPORT ERROR:", exc, file=sys.stderr, flush=True)


def main():
    root = Path.home() / "qwen-shared-w8a8"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=root / "full-model-tools/full_qwen_shared_basic.py")
    parser.add_argument("--bundle", type=Path, default=root / "full_model_portability_v4")
    parser.add_argument("--cache", type=Path, help="Reuse this experiment's verified cache directory")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt-ids", type=Path)
    parser.add_argument("--workspace-mib", type=int, default=64)
    parser.add_argument("--prepare", help=argparse.SUPPRESS)
    parser.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.workspace_mib < 1:
        parser.error("Output length and workspace must be positive")
    args.bundle, args.runner = args.bundle.resolve(), args.runner.resolve()
    runner = load_runner(args.runner)
    manifest, specs = selected_specs(args.bundle)
    if args.prepare:
        if args.cache is None or args.prepare not in {s["name"] for s in specs}:
            parser.error("Invalid internal compilation request")
        args.cache = args.cache.resolve()
        prepare_graph(args, runner, args.prepare)
        return
    verify_sources(args.bundle, manifest)
    current_identity = identity(args)
    if args.cache is None:
        args.cache = args.bundle / "diagnostics" / ("aot_shared_" + str(time.time_ns()))
        args.cache.mkdir(parents=True, exist_ok=False)
        write_json(args.cache / "identity.json", current_identity)
    else:
        args.cache = args.cache.resolve()
        if json.loads((args.cache / "identity.json").read_text()) != current_identity:
            raise RuntimeError("Cache identity differs from source/runtime; create a fresh cache")
    if args.execute:
        run_cached(args, runner, specs)
        return
    print("EXPERIMENT CACHE:", args.cache, flush=True)
    print("Source model unchanged. Compile each graph in a fresh process; then test the full model.", flush=True)
    base = [sys.executable, str(Path(__file__).resolve()), "--runner", str(args.runner),
            "--bundle", str(args.bundle), "--cache", str(args.cache)]
    for index, spec in enumerate(specs, 1):
        name = spec["name"]
        if (args.cache / name / "ready.json").is_file():
            cache_record(args.cache, name)
            print(f"[{index}/32] Verified cached {name}", flush=True)
            continue
        if (args.cache / name).exists():
            raise RuntimeError(f"Incomplete cache for {name}; start a fresh experiment without --cache")
        print(f"[{index}/32] Compile unchanged {name}", flush=True)
        subprocess.run(base + ["--prepare", name], check=True)
    command = base + ["--execute", "--max-new-tokens", str(args.max_new_tokens),
                      "--workspace-mib", str(args.workspace_mib)]
    if args.prompt_ids:
        command += ["--prompt-ids", str(args.prompt_ids.resolve())]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
