"""Meaningful precision, original-tensor, and recurrent-state regression checks."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto as T, helper as H, numpy_helper as NH
import onnxruntime as ort

spec = importlib.util.spec_from_file_location("qwen35_w8a16", Path(__file__).resolve().parents[1] / "scripts/qwen35_w8a16_gpu.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def execute(model, feeds):
    opt = ort.SessionOptions()
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opt.intra_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), opt, providers=["CPUExecutionProvider"]).run(None, feeds)


class QuantizationBridgeTests(unittest.TestCase):
    def test_compact_graph_changes_only_qdq_implementation(self):
        original = bridge.fixture()
        original.graph.node.append(H.make_node("Identity", ["Y"], ["unused"]))
        before = original.SerializeToString()
        lowered, report = bridge.compact_qdq(original)
        self.assertEqual(before, original.SerializeToString())
        self.assertEqual(len(lowered.graph.node), len(original.graph.node))
        self.assertEqual(bridge.initializer_digest(original), bridge.initializer_digest(lowered))
        for old, new in zip(original.graph.node, lowered.graph.node):
            restored = onnx.NodeProto(); restored.CopyFrom(new)
            restored.domain, restored.op_type = old.domain, old.op_type
            if not old.HasField("domain"):
                restored.ClearField("domain")
            self.assertEqual(old.SerializeToString(), restored.SerializeToString())
        self.assertEqual(lowered.graph.node[0].op_type, "QuantizeU16")
        self.assertTrue(report["original_io_unchanged"])
        with self.assertRaises(ValueError):
            bridge.compact_qdq(bridge.fixture(T.INT8))

    @unittest.skipUnless(os.environ.get("QWEN35_QDQ_REFERENCE_LIBRARY"), "Set QWEN35_QDQ_REFERENCE_LIBRARY to test compiled reference")
    def test_compiled_reference_matches_original_exhaustive_and_per_axis(self):
        library = os.environ["QWEN35_QDQ_REFERENCE_LIBRARY"]
        def converted(model, feeds):
            lower, _ = bridge.compact_qdq(model)
            opt = ort.SessionOptions()
            opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            opt.intra_op_num_threads = 1
            opt.register_custom_ops_library(library)
            return ort.InferenceSession(lower.SerializeToString(), opt, providers=["CPUExecutionProvider"]).run(None, feeds)
        for dtype, limit, zp in [(T.UINT16, 65535, 32768), (T.UINT8, 255, 0)]:
            for scale in (.125, 5.893147772439988e-06, .0031, .07):
                model = bridge.fixture(dtype, scale)
                codes = np.arange(limit + 1, dtype=np.float32)
                x = np.concatenate([(codes-zp)*np.float32(scale),
                                    (np.arange(-100, 100, dtype=np.float32)+.5)*np.float32(scale),
                                    np.array([-1e20, 1e20], np.float32)])
                for a, b in zip(execute(model, {"X": x}), converted(model, {"X": x})):
                    np.testing.assert_array_equal(a, b)
        for axis in (0, 1, -1):
            model, expected = bridge.weight_fixture(axis)
            np.testing.assert_array_equal(execute(model, {})[0], converted(model, {})[0])
            np.testing.assert_array_equal(expected, converted(model, {})[0])

    def test_real_non_binary_activation_scales_preserve_every_code(self):
        for scale in (5.893147772439988e-06, .0031, .07):
            model = bridge.fixture(T.UINT16, scale)
            lowered, _ = bridge.expand_qdq(model)
            x = (np.arange(65536, dtype=np.float32)-32768)*np.float32(scale)
            for a, b in zip(execute(model, {"X": x}), execute(lowered, {"X": x})):
                np.testing.assert_array_equal(a, b)

    def test_all_integer_codes_and_ties_match_original_qdq(self):
        for dtype, npdtype, zp in [(T.UINT16, np.uint16, 32768), (T.INT16, np.int16, 0),
                                    (T.UINT8, np.uint8, 0), (T.INT8, np.int8, 0)]:
            with self.subTest(dtype=dtype):
                original = bridge.fixture(dtype)
                lowered, report = bridge.expand_qdq(original)
                onnx.checker.check_model(lowered)
                limits = np.iinfo(npdtype)
                codes = np.arange(limits.min, limits.max + 1, dtype=np.float32)
                x = np.concatenate([(codes-zp)*.125,
                                    (np.arange(-100, 100, dtype=np.float32)+.5)*.125,
                                    np.array([-1e20, 1e20], np.float32)])
                a, b = execute(original, {"X": x}), execute(lowered, {"X": x})
                for lhs, rhs in zip(a, b):
                    np.testing.assert_array_equal(lhs, rhs)
                self.assertTrue(report["original_initializers_unchanged"])
                self.assertTrue(report["original_io_unchanged"])
                np.testing.assert_array_equal(b[0][:codes.size], codes.astype(npdtype))

    def test_per_channel_weight_decode_in_a_real_matmul(self):
        rng = np.random.default_rng(13)
        for axis, shape in [(0, (5, 7)), (1, (7, 5)), (-1, (7, 5))]:
            w = rng.integers(0, 256, shape, dtype=np.uint8)
            scales = np.array([.0031, .07, .014, .021, .23], np.float32)
            zps = np.array([120, 17, 128, 250, 0], np.uint8)
            x = rng.normal(size=(2, shape[0])).astype(np.float32)
            g = H.make_graph([H.make_node("DequantizeLinear", ["W", "S", "Z"], ["DW"], domain="com.microsoft", axis=axis),
                              H.make_node("MatMul", ["X", "DW"], ["Y"])], "weight_decode",
                             [H.make_tensor_value_info("X", T.FLOAT, x.shape)],
                             [H.make_tensor_value_info("Y", T.FLOAT, [2, shape[1]])],
                             [NH.from_array(w, "W"), NH.from_array(scales, "S"), NH.from_array(zps, "Z")])
            model = H.make_model(g, opset_imports=[H.make_opsetid("", 17), H.make_opsetid("com.microsoft", 1)])
            model.ir_version = 10
            before = model.SerializeToString()
            lowered, _ = bridge.expand_qdq(model)
            self.assertEqual(before, model.SerializeToString())
            np.testing.assert_array_equal(execute(model, {"X": x})[0], execute(lowered, {"X": x})[0])
            self.assertEqual([t.SerializeToString() for t in lowered.graph.initializer[:3]],
                             [t.SerializeToString() for t in model.graph.initializer])

    def test_unreviewed_quantization_is_rejected(self):
        for change in ["block", "negative_scale", "float_zp", "missing_zp", "foreign_op"]:
            model = bridge.fixture()
            if change == "block":
                model.graph.node[0].attribute.append(H.make_attribute("block_size", 16))
            elif change == "negative_scale":
                model.graph.initializer[0].CopyFrom(NH.from_array(np.array(-.125, np.float32), "scale"))
            elif change == "float_zp":
                model.graph.initializer[1].CopyFrom(NH.from_array(np.array(0., np.float32), "zp"))
            elif change == "missing_zp":
                del model.graph.node[0].input[2:]
            else:
                model.graph.node[0].domain = "unreviewed.vendor"
            with self.subTest(change=change), self.assertRaises(ValueError):
                bridge.expand_qdq(model)
            with self.subTest(compact_change=change), self.assertRaises(ValueError):
                bridge.compact_qdq(model)

    def test_external_weight_descriptors_preserved_without_loading(self):
        model = bridge.fixture()
        w = onnx.TensorProto(name="external_weight", data_type=T.UINT8, dims=[1024], data_location=T.EXTERNAL)
        e = w.external_data.add(); e.key = "location"; e.value = "language_weights.data"
        e = w.external_data.add(); e.key = "offset"; e.value = "16"
        e = w.external_data.add(); e.key = "length"; e.value = "1024"
        model.graph.initializer.append(w)
        lowered, _ = bridge.expand_qdq(model)
        self.assertEqual(w.SerializeToString(), lowered.graph.initializer[2].SerializeToString())

    def test_state_requantization_exhaustive_against_scalar_rounding(self):
        row = {"output_scale": "0.000041078928916249424", "output_zero_point": "32494",
               "input_scale": "0.000040970437112264335", "input_zero_point": "32406"}
        codes = np.arange(65536, dtype=np.uint16)
        actual = bridge.requantize(codes, row)
        def scalar(code):
            # Python round uses nearest-even, like C++ nearbyintf.
            centered = np.float32(int(code)-32494)
            decoded = np.float32(centered * np.float32(row["output_scale"]))
            scaled = np.float32(decoded / np.float32(row["input_scale"]))
            return min(65535, max(0, round(float(scaled))+32406))
        expected = np.fromiter((scalar(c) for c in codes), dtype=np.uint16, count=65536)
        np.testing.assert_array_equal(actual, expected)

    def test_state_zero_encoding_and_strided_update(self):
        contract = [{"name": "cache", "kind": "input", "dtype": "uint16", "shape": "2,4"}]
        row = {"source": "cache_out", "destination": "cache", "source_offset_elements": "1",
               "destination_offset_elements": "0", "rows": "2", "row_elements": "3",
               "source_stride_elements": "4", "destination_stride_elements": "4",
               "operation": "copy_uint16", "input_zero_point": "12345"}
        feeds = bridge.initial_state(contract, [row])
        np.testing.assert_array_equal(feeds["cache"], np.full((2, 4), 12345, np.uint16))
        source = np.arange(8, dtype=np.uint16).reshape(2, 4)
        before = source.copy()
        bridge.advance_state(feeds, {"cache_out": source}, [row])
        np.testing.assert_array_equal(feeds["cache"], [[1, 2, 3, 12345], [5, 6, 7, 12345]])
        np.testing.assert_array_equal(source, before)
        row["destination_offset_elements"] = "100"
        with self.assertRaises(ValueError):
            bridge.advance_state(feeds, {"cache_out": source}, [row])

    def test_cpu_or_empty_activity_is_not_a_gpu_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"
            for events in [[], [{"args": {"provider": "CPUExecutionProvider"}}],
                           [{"args": {"provider": "CUDAExecutionProvider"}}, {"args": {"provider": "CPUExecutionProvider"}}]]:
                path.write_text(json.dumps(events))
                with self.assertRaises(RuntimeError):
                    bridge.audit_profile(path, "CUDAExecutionProvider")
            path.write_text(json.dumps([{"args": {"provider": "CUDAExecutionProvider"}}]))
            self.assertEqual(bridge.audit_profile(path, "CUDAExecutionProvider"), {"CUDAExecutionProvider": 1})


if __name__ == "__main__":
    unittest.main()
