#!/usr/bin/env python3
"""Run the shared bundle with decoder sessions resident and one head open at a time.

Uses the existing basic runner and its source-model hash checks, arithmetic,
inputs, state updates and provider configuration. QNN head contexts are cached
within this run only; TensorRT uses the existing runner's engine cache. The
loading policy is the same for both accelerators. No accuracy claim is implied.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys


def config(options, key, value):
    try:
        if options.get_session_config_entry(key) == value:
            return
    except RuntimeError:
        pass
    options.add_session_config_entry(key, value)


class StreamedHead:
    def __init__(self, owner, model, positional, keywords):
        self.owner, self.model = owner, Path(model)
        self.positional, self.keywords = positional, dict(keywords)
        self.options = self.keywords["sess_options"]
        self.prefix = Path(self.options.profile_file_prefix)
        self.profiles, self.profile_errors = [], []
        self.calls = 0
        self.fallback_disabled = False

    def disable_fallback(self):
        self.fallback_disabled = True

    def run(self, output_names, feed, *extra, **kwargs):
        if self.owner.active_heads:
            raise RuntimeError("Only one output-head session may be held by this runner")
        if not self.fallback_disabled:
            raise RuntimeError("The original runner did not disable session fallback")
        self.calls += 1
        self.options.profile_file_prefix = str(self.prefix) + f"_open_{self.calls:04d}"
        model = self.model
        if self.owner.provider == "qnn" and self.owner.context_cache:
            cache = self.prefix.parent / "head_context_cache"
            cache.mkdir(exist_ok=True)
            cached_model = cache / f"{self.model.stem}_ctx.onnx"
            if cached_model.is_file():
                config(self.options, "ep.context_enable", "0")
                model = cached_model
            else:
                config(self.options, "ep.context_enable", "1")
                config(self.options, "ep.context_embed_mode", "1")
                config(self.options, "ep.context_file_path", str(cached_model))
        session, primary_error = None, False
        self.owner.active_heads += 1
        self.owner.max_heads = max(self.owner.max_heads, self.owner.active_heads)
        try:
            session = self.owner.original.InferenceSession(str(model), *self.positional, **self.keywords)
            session.disable_fallback()
            return session.run(output_names, feed, *extra, **kwargs)
        except BaseException:
            primary_error = True
            raise
        finally:
            try:
                if session is not None:
                    try:
                        self.profiles.append(session.end_profiling())
                    except Exception as exc:
                        self.profile_errors.append(str(exc))
                        if not primary_error:
                            raise
            finally:
                # Do not keep the session alive between vocabulary chunks.
                del session
                gc.collect()
                self.owner.active_heads -= 1

    def end_profiling(self):
        if self.profile_errors:
            raise RuntimeError("Head profiling failed: " + "; ".join(self.profile_errors))
        events = []
        for source in self.profiles:
            # Retain raw traces separately. This concatenation is for the
            # original provider audit, not a combined timing timeline.
            events.extend(json.loads(Path(source).read_text()))
        merged = Path(str(self.prefix) + "_provider_audit.json")
        merged.write_text(json.dumps(events))
        return str(merged)


class StreamedOrt:
    def __init__(self, original, provider, context_cache=True):
        self.original, self.provider, self.context_cache = original, provider, context_cache
        self.active_heads, self.max_heads = 0, 0
        self.heads, self.report_path = [], None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def InferenceSession(self, model, *positional, **keywords):
        prefix = Path(keywords["sess_options"].profile_file_prefix)
        self.report_path = prefix.parent / "run_report.json"
        if Path(model).stem.startswith("head_"):
            head = StreamedHead(self, model, positional, keywords)
            self.heads.append(head)
            return head
        return self.original.InferenceSession(model, *positional, **keywords)


def annotate_report(owner):
    if owner.report_path is None or not owner.report_path.is_file():
        return
    record = json.loads(owner.report_path.read_text())
    # An unused session after an execution error is not proof of CPU fallback.
    if record.get("error"):
        record["status"] = "failed"
    record["session_loading"] = {
        "policy": "resident decoders; open, execute and release each vocabulary head",
        "max_head_sessions_held_by_runner": owner.max_heads,
        "head_calls": {h.model.stem: h.calls for h in owner.heads},
        "raw_head_profiles": {h.model.stem: h.profiles for h in owner.heads},
        "merged_head_profiles": "provider audit only; separate session time bases",
        "qnn_head_context_cache": owner.provider == "qnn" and owner.context_cache,
    }
    record["timing_scope"] = "includes host state transport and output-head session loading"
    record["accuracy_acceptance"] = "not established"
    temporary = owner.report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(owner.report_path)
    print("STREAMED-HEAD STATUS:", record["status"], flush=True)
    print("STREAMED-HEAD REPORT:", owner.report_path, flush=True)


def main():
    root = Path.home() / "qwen-shared-w8a8"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=root / "full-model-tools/full_qwen_shared_basic.py")
    parser.add_argument("--bundle", type=Path, default=root / "full_model_portability_v4")
    parser.add_argument("--provider", choices=("cpu", "qnn", "trt"), required=True)
    parser.add_argument("--precision", choices=("fp32", "int8"), default="int8")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt-ids", type=Path)
    parser.add_argument("--workspace-mib", type=int, default=64)
    parser.add_argument("--no-head-context-cache", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.workspace_mib < 1:
        parser.error("Output length and workspace must be positive")
    spec = importlib.util.spec_from_file_location("_qwen_streamed_runner", args.runner.resolve())
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    if runner.ort.__version__ != "1.27.0":
        raise RuntimeError("Use the pinned ONNX Runtime 1.27.0 environment")
    if runner.options().graph_optimization_level != runner.ort.GraphOptimizationLevel.ORT_ENABLE_BASIC:
        raise RuntimeError("Expected the existing basic-optimization runner")
    owner = StreamedOrt(runner.ort, args.provider, not args.no_head_context_cache)
    runner.ort = owner
    print("LOADING POLICY: 24 resident decoder sessions; at most one output head held at a time.", flush=True)
    primary_error = False
    try:
        runner.run(args)
    except BaseException:
        primary_error = True
        raise
    finally:
        try:
            annotate_report(owner)
        except Exception as exc:
            if not primary_error:
                raise
            print("SECONDARY REPORT ERROR:", exc, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
