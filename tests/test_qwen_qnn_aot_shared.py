import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_qwen_qnn_aot_shared.py"
SPEC = importlib.util.spec_from_file_location("qwen_aot_shared", SOURCE)
aot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aot)


class AotSharedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle, self.cache = self.root / "bundle", self.root / "cache"
        (self.bundle / "int8").mkdir(parents=True)
        self.cache.mkdir()

    def graph_cache(self, name):
        weights = np.array([[1, 2, 3], [-1, 4, 0]], dtype=np.float32)
        graph = helper.make_graph([helper.make_node("MatMul", ["X", "W"], ["Y"])], "head",
                                  [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
                                  [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])],
                                  [numpy_helper.from_array(weights, "W")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
        source = self.bundle / "int8" / (name + ".onnx")
        onnx.save(model, source)
        folder = self.cache / name
        folder.mkdir()
        target = folder / "model_ctx.onnx"
        # A regular ONNX graph stands in for the QNN EPContext on this CPU host.
        target.write_bytes(source.read_bytes())
        (folder / "context.bin").write_bytes(b"compiled context placeholder")
        aot.write_json(folder / "ready.json", {"schema": 1, "files": {
            p.name: aot.sha(p) for p in (target, folder / "context.bin")}})
        return source, target, weights

    def test_cached_dispatch_keeps_real_cpu_results_source_and_provider_guard(self):
        source, target, weights = self.graph_cache("layer_00")
        before = source.read_bytes()
        seen = []

        def factory(path, **kwargs):
            seen.append(path)
            self.assertFalse(kwargs["enable_fallback"])
            self.assertEqual(kwargs["sess_options"].get_session_config_entry(
                "session.disable_cpu_ep_fallback"), "1")
            return ort.InferenceSession(path, sess_options=ort.SessionOptions(), providers=["CPUExecutionProvider"])

        owner = aot.AotOrt(SimpleNamespace(InferenceSession=factory), self.cache, self.bundle, ["layer_00"])
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.profile_file_prefix = str(self.root / "runs" / "layer_00")
        session = owner.InferenceSession(str(source), sess_options=options, enable_fallback=False)
        x = np.array([[2, 3]], dtype=np.float32)
        np.testing.assert_array_equal(session.run(None, {"X": x})[0], x @ weights)
        self.assertEqual(seen, [str(target)])
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(owner.loaded, ["layer_00"])

    def test_changed_context_binary_is_rejected_before_session_creation(self):
        self.graph_cache("layer_14")
        (self.cache / "layer_14" / "context.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            aot.cache_record(self.cache, "layer_14")

    def test_changed_source_is_rejected(self):
        source, _, _ = self.graph_cache("layer_00")
        manifest = {"files": {"int8/layer_00.onnx": aot.sha(source)}}
        aot.verify_sources(self.bundle, manifest)
        source.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "Source checksum mismatch"):
            aot.verify_sources(self.bundle, manifest)

    def test_external_context_prepare_options_and_sharing_are_separate(self):
        class Options:
            def __init__(self):
                self.config = {}

            def add_session_config_entry(self, key, value):
                self.config[key] = value

            def add_provider_for_devices(self, devices, values):
                self.devices, self.values = devices, values

        device = SimpleNamespace(ep_name="QNNExecutionProvider")
        runner = SimpleNamespace(options=Options, ort=SimpleNamespace(get_ep_devices=lambda: [device]))
        qnn = SimpleNamespace(get_qnn_htp_path=lambda: "/test/libQnnHtp.so")
        with patch.dict("sys.modules", {"onnxruntime_qnn": qnn}):
            options = aot.qnn_options(runner, self.root / "model_ctx.onnx", "layer_14", prepare=True)
            self.assertEqual(options.config["ep.context_embed_mode"], "0")
            self.assertEqual(options.config["session.disable_cpu_ep_fallback"], "1")
            self.assertEqual(options.config["ep.context_node_name_prefix"], "layer_14_")
            self.assertEqual(options.values["enable_htp_prepare_only"], "1")
            self.assertNotIn("htp_share_resource_optimization", options.values)

    def test_existing_loop_receives_shared_external_context_config_and_failure_is_preserved(self):
        observed = []

        def options():
            return ort.SessionOptions()

        runner = SimpleNamespace(options=options, ort=ort)
        report = self.root / "run_report.json"
        report.write_text(json.dumps({"status": "provider_audit_failed", "error": "original QNN error"}))

        def run(args):
            value = runner.options()
            observed.append(value.get_session_config_entry("ep.qnnexecutionprovider.htp_share_resource_optimization"))
            self.assertEqual(value.get_session_config_entry("ep.context_embed_mode"), "0")
            self.assertEqual(value.get_session_config_entry("ep.context_enable"), "0")
            runner.ort.report_path = report
            raise RuntimeError("primary QNN error")

        runner.run = run
        args = SimpleNamespace(cache=self.cache, bundle=self.bundle, resource_sharing=True)
        with self.assertRaisesRegex(RuntimeError, "primary QNN error"):
            aot.run_cached(args, runner, [{"name": "layer_00"}])
        record = json.loads(report.read_text())
        self.assertEqual(observed, ["1"])
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], "original QNN error")

    def test_plain_aot_does_not_request_unsupported_resource_sharing(self):
        runner = SimpleNamespace(options=ort.SessionOptions, ort=ort)
        seen = []

        def run(args):
            options = runner.options()
            for key in ("htp_share_resource_optimization", "enable_vtcm_backup_buffer_sharing"):
                with self.assertRaises(RuntimeError):
                    options.get_session_config_entry("ep.qnnexecutionprovider." + key)
            self.assertEqual(options.get_session_config_entry("ep.context_enable"), "0")
            self.assertEqual(options.get_session_config_entry("ep.context_embed_mode"), "0")
            seen.append(True)

        runner.run = run
        args = SimpleNamespace(cache=self.cache, bundle=self.bundle, resource_sharing=False)
        aot.run_cached(args, runner, [{"name": "layer_00"}])
        self.assertEqual(seen, [True])

    def test_prepare_creates_guarded_external_cache_without_executing_graph(self):
        source = self.bundle / "int8" / "layer_14.onnx"
        source.write_bytes(b"unchanged source")
        seen = []

        class Prepared:
            def disable_fallback(self):
                seen.append("fallback disabled")

            def run(self, *args, **kwargs):
                raise AssertionError("Preparation must not run inference")

        def construct(path, **kwargs):
            self.assertEqual(path, str(source))
            self.assertFalse(kwargs["enable_fallback"])
            folder = self.cache / "layer_14"
            (folder / "model_ctx.onnx").write_bytes(b"context model")
            (folder / "context.bin").write_bytes(b"compiled binary")
            return Prepared()

        fake_ort = SimpleNamespace(register_execution_provider_library=lambda *args: None,
                                   InferenceSession=construct)
        runner = SimpleNamespace(ort=fake_ort)
        args = SimpleNamespace(cache=self.cache, bundle=self.bundle)
        qnn = SimpleNamespace(get_library_path=lambda: "/test/qnn.so")
        with patch.dict("sys.modules", {"onnxruntime_qnn": qnn}), patch.object(aot, "qnn_options", return_value=object()):
            aot.prepare_graph(args, runner, "layer_14")
        self.assertEqual(aot.cache_record(self.cache, "layer_14"), self.cache / "layer_14/model_ctx.onnx")
        self.assertEqual(source.read_bytes(), b"unchanged source")
        self.assertEqual(seen, ["fallback disabled"])


if __name__ == "__main__":
    unittest.main()
