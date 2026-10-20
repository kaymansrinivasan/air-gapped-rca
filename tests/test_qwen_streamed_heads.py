import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_qwen_streamed_heads.py"
SPEC = importlib.util.spec_from_file_location("streamed_heads", SOURCE)
stream = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stream)


class StreamedHeadsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def options(self, name):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.intra_op_num_threads = 1
        options.enable_profiling = True
        options.profile_file_prefix = str(self.root / name)
        return options

    def test_real_cpu_sessions_are_deferred_released_and_outputs_preserved(self):
        count = {"created": 0, "active": 0, "peak": 0}

        class Tracked:
            def __init__(self, *args, **kwargs):
                self.inner = ort.InferenceSession(*args, **kwargs)
                count["created"] += 1
                count["active"] += 1
                count["peak"] = max(count["peak"], count["active"])

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def __del__(self):
                count["active"] -= 1

        weights = np.array([[1, 2, 3], [-1, 4, 0]], dtype=np.float32)
        graph = helper.make_graph([helper.make_node("MatMul", ["X", "W"], ["Y"])], "head",
                                  [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
                                  [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])],
                                  [numpy_helper.from_array(weights, "W")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
        owner = stream.StreamedOrt(SimpleNamespace(InferenceSession=Tracked), "cpu")
        heads, originals = [], {}
        for name in ("head_000000", "head_032768"):
            path = self.root / f"{name}.onnx"
            onnx.save(model, path)
            originals[path] = path.read_bytes()
            head = owner.InferenceSession(str(path), sess_options=self.options(name),
                                          providers=["CPUExecutionProvider"], enable_fallback=False)
            head.disable_fallback()
            heads.append(head)
        self.assertEqual(count["created"], 0)
        for value in (np.array([[2, 3]], dtype=np.float32), np.array([[4, -2]], dtype=np.float32)):
            for head in heads:
                result = head.run(None, {"X": value})[0]
                np.testing.assert_array_equal(result, value @ weights)
                self.assertEqual(count["active"], 0)
        self.assertEqual(count["created"], 4)
        self.assertEqual(count["peak"], 1)
        self.assertEqual(owner.max_heads, 1)
        for head in heads:
            events = json.loads(Path(head.end_profiling()).read_text())
            self.assertEqual(sum(e.get("name") == "model_run" for e in events), 2)
            self.assertEqual({e["args"]["provider"] for e in events if e.get("args", {}).get("provider")}, {"CPUExecutionProvider"})
        for path, before in originals.items():
            self.assertEqual(path.read_bytes(), before)

    def test_primary_execution_error_survives_profile_error(self):
        class Broken:
            def disable_fallback(self):
                pass

            def run(self, *args, **kwargs):
                raise RuntimeError("primary graph error")

            def end_profiling(self):
                raise RuntimeError("secondary audit error")

        owner = stream.StreamedOrt(SimpleNamespace(InferenceSession=lambda *a, **k: Broken()), "cpu")
        head = owner.InferenceSession(str(self.root / "head_test.onnx"), sess_options=self.options("head_test"))
        head.disable_fallback()
        with self.assertRaisesRegex(RuntimeError, "primary graph error"):
            head.run(None, {"X": np.ones(1, dtype=np.float32)})
        self.assertEqual(owner.active_heads, 0)
        self.assertIn("secondary audit error", head.profile_errors[0])

    def test_qnn_cache_switch_is_local_and_keeps_fallback_option(self):
        seen = []
        source = self.root / "head_test.onnx"
        source.write_bytes(b"original graph")
        options = self.options("head_test")
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

        class FakeSession:
            def disable_fallback(self):
                pass

            def run(self, output_names, feed):
                return [feed["X"].copy()]

            def end_profiling(self):
                path = Path(options.profile_file_prefix + ".json")
                path.write_text("[]")
                return str(path)

        def construct(path, **kwargs):
            seen.append((path, options.get_session_config_entry("ep.context_enable")))
            self.assertEqual(options.get_session_config_entry("session.disable_cpu_ep_fallback"), "1")
            if seen[-1][1] == "1":
                Path(options.get_session_config_entry("ep.context_file_path")).write_bytes(b"fake compiled context")
            return FakeSession()

        owner = stream.StreamedOrt(SimpleNamespace(InferenceSession=construct), "qnn")
        head = owner.InferenceSession(str(source), sess_options=options)
        head.disable_fallback()
        for _ in range(2):
            head.run(None, {"X": np.ones(1, dtype=np.float32)})
        self.assertEqual(seen[0], (str(source), "1"))
        self.assertEqual(seen[1][1], "0")
        self.assertTrue(seen[1][0].endswith("head_context_cache/head_test_ctx.onnx"))
        self.assertEqual(source.read_bytes(), b"original graph")

    def test_report_keeps_execution_failure_as_primary_status(self):
        owner = stream.StreamedOrt(ort, "cpu")
        owner.report_path = self.root / "run_report.json"
        owner.report_path.write_text(json.dumps({"status": "provider_audit_failed", "error": "original execution error"}))
        stream.annotate_report(owner)
        record = json.loads(owner.report_path.read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], "original execution error")


if __name__ == "__main__":
    unittest.main()
