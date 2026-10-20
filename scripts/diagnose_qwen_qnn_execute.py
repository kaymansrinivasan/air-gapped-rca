#!/usr/bin/env python3
"""Capture a full-model QNN failure, then replay that exact call in a fresh process.

Diagnostic only: no ONNX files, quantization, runner source, or fallback settings
are changed. A successful isolated replay is not a model accuracy approval.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_runner(path):
    spec = importlib.util.spec_from_file_location("_qwen_diagnostic_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.ort.__version__ != "1.27.0":
        raise RuntimeError("Use the existing ONNX Runtime 1.27.0 environment")
    if module.options().graph_optimization_level != module.ort.GraphOptimizationLevel.ORT_ENABLE_BASIC:
        raise RuntimeError("Expected the basic-optimization runner; source was not changed")
    return module


class SessionCapture:
    def __init__(self, session, model, args):
        self.session, self.model, self.args = session, Path(model).resolve(), args
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.session, name)

    def run(self, output_names, feed, *extra, **kwargs):
        self.calls += 1
        # Copy before execution: a failed backend must not corrupt our fixture.
        snapshot = {k: np.array(v, copy=True) for k, v in feed.items()} if self.model.stem == self.args.target else None
        try:
            return self.session.run(output_names, feed, *extra, **kwargs)
        except Exception as exc:
            if snapshot is not None:
                try:
                    fixture = self.args.output / "failed_inputs.npz"
                    with fixture.open("wb") as stream:
                        np.savez(stream, **snapshot)
                    save(self.args.output / "fault.json", {
                        "status": "captured_before_failed_call",
                        "model": str(self.model), "model_sha256": sha(self.model),
                        "runner": str(self.args.runner), "runner_sha256": sha(self.args.runner),
                        "fixture_sha256": sha(fixture),
                        "layer": self.model.stem, "call_number": self.calls,
                        "error": f"{type(exc).__name__}: {exc}",
                        "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                                       "finite": bool(np.isfinite(v).all())} for k, v in snapshot.items()},
                        "cpu_fallback": "disabled", "precision": "existing INT8 QDQ",
                    })
                    print(f"DIAGNOSTIC: saved exact inputs for {self.model.stem}, call {self.calls}", flush=True)
                except Exception as capture_error:
                    print(f"DIAGNOSTIC CAPTURE ERROR: {capture_error}", flush=True)
            raise


class OrtCapture:
    def __init__(self, original, args):
        self.original, self.args = original, args

    def __getattr__(self, name):
        return getattr(self.original, name)

    def InferenceSession(self, model, *args, **kwargs):
        return SessionCapture(self.original.InferenceSession(model, *args, **kwargs), model, self.args)


def capture(args):
    runner = load_runner(args.runner)
    runner.ort = OrtCapture(runner.ort, args)
    run_args = argparse.Namespace(bundle=args.bundle, provider="qnn", precision="int8",
                                  max_new_tokens=1, prompt_ids=None, workspace_mib=64)
    try:
        runner.run(run_args)
    except Exception as exc:
        save(args.output / "capture_result.json", {"status": "full_run_failed",
                                                   "error": f"{type(exc).__name__}: {exc}"})
        return 2
    save(args.output / "capture_result.json", {"status": "failure_not_reproduced",
                                               "accuracy_approved": False})
    return 0


def verify_fixture(args):
    record = json.loads((args.output / "fault.json").read_text())
    fixture = args.output / "failed_inputs.npz"
    for path, expected in ((Path(record["model"]), record["model_sha256"]),
                           (args.runner, record["runner_sha256"]),
                           (fixture, record["fixture_sha256"])):
        if sha(path) != expected:
            raise RuntimeError(f"Changed file; replay refused: {path}")
    with np.load(fixture, allow_pickle=False) as data:
        feed = {k: data[k].copy() for k in data.files}
    return record, feed


def replay(args):
    record, feed = verify_fixture(args)
    runner = load_runner(args.runner)
    import onnxruntime_qnn as qnn
    ort = runner.ort
    ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
    devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    if not devices:
        raise RuntimeError("QNN device unavailable")
    opt = runner.options()
    opt.enable_profiling = True
    opt.profile_file_prefix = str(args.output / "isolated_profile")
    opt.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    opt.add_provider_for_devices(devices, {"backend_path": qnn.get_qnn_htp_path(),
                                         "offload_graph_io_quantization": "0"})
    result = {"status": "isolated_execution_failed", "layer": record["layer"],
              "original_call_number": record["call_number"], "accuracy_approved": False}
    session = None
    try:
        session = ort.InferenceSession(record["model"], sess_options=opt, enable_fallback=False)
        session.disable_fallback()
        outputs = session.run(None, feed)
        if not all(np.isfinite(v).all() for v in outputs):
            raise RuntimeError("Isolated execution returned non-finite outputs")
        result["status"] = "isolated_execution_passed"
        result["output_shapes"] = [list(v.shape) for v in outputs]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if session is not None:
            try:
                profile = session.end_profiling()
                events = json.loads(Path(profile).read_text())
                providers = sorted({e["args"]["provider"] for e in events if e.get("args", {}).get("provider")})
                result.update(profile=profile, observed_providers=providers)
                if result["status"] == "isolated_execution_passed" and providers != ["QNNExecutionProvider"]:
                    result.update(status="isolated_provider_audit_failed")
            except Exception as exc:
                result["profile_error"] = str(exc)
                if result["status"] == "isolated_execution_passed":
                    result["status"] = "isolated_provider_audit_failed"
        save(args.output / "replay_result.json", result)
    print("DIAGNOSTIC REPLAY:", json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "isolated_execution_passed" else 3


def main():
    root = Path.home() / "qwen-shared-w8a8"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=root / "full-model-tools/full_qwen_shared_basic.py")
    parser.add_argument("--bundle", type=Path, default=root / "full_model_portability_v4")
    parser.add_argument("--target", default="layer_14")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("auto", "capture", "replay"), default="auto")
    args = parser.parse_args()
    args.runner, args.bundle = args.runner.resolve(), args.bundle.resolve()
    args.output = (args.output or args.bundle / "diagnostics" / f"execute_{time.time_ns()}").resolve()
    args.output.mkdir(parents=True, exist_ok=args.stage != "auto")
    if args.stage == "capture":
        return capture(args)
    if args.stage == "replay":
        return replay(args)
    command = [sys.executable, str(Path(__file__).resolve()), "--runner", str(args.runner),
               "--bundle", str(args.bundle), "--target", args.target, "--output", str(args.output)]
    print("DIAGNOSTIC: one full-model reproduction, followed by exact-input replay in a fresh process.", flush=True)
    attempt = subprocess.run(command + ["--stage", "capture"])
    if not (args.output / "fault.json").is_file():
        print("DIAGNOSTIC: no target fixture captured; do not infer a cause. Output:", args.output, flush=True)
        return attempt.returncode
    attempt = subprocess.run(command + ["--stage", "replay"])
    print("DIAGNOSTIC OUTPUT:", args.output, flush=True)
    return attempt.returncode


if __name__ == "__main__":
    sys.exit(main())
