"""Numerical and fail-closed checks for the standalone ONNX graph patch."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
import onnxruntime as ort


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/patch_qwen_qdq_portability.py"
spec = importlib.util.spec_from_file_location("qdq_patch", SCRIPT)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


def sample(scale=.125, output_scale=.07, shape=(256,)):
    tensors = [nh.from_array(np.array(scale, np.float32), "scale"),
               nh.from_array(np.array(output_scale, np.float32), "out_scale"),
               nh.from_array(np.array(0, np.int8), "zp")]
    nodes = [h.make_node("Cast", ["X"], ["cast"], name="identity_cast", to=T.FLOAT),
             h.make_node("QuantizeLinear", ["cast", "scale", "zp"], ["qx"]),
             h.make_node("DequantizeLinear", ["qx", "scale", "zp"], ["dx"]),
             h.make_node("Cast", ["dx"], ["cast2"], name="identity_cast_2", to=T.FLOAT),
             h.make_node("Neg", ["cast2"], ["neg"], name="negation"),
             h.make_node("QuantizeLinear", ["neg", "out_scale", "zp"], ["qy"]),
             h.make_node("DequantizeLinear", ["qy", "out_scale", "zp"], ["Y"])]
    model = h.make_model(h.make_graph(nodes, "test", [h.make_tensor_value_info("X", T.FLOAT, shape)],
                         [h.make_tensor_value_info("Y", T.FLOAT, shape)], tensors),
                         opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 10
    return model


def execute(model, x):
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    o.intra_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), o, providers=["CPUExecutionProvider"])
    return session.run(None, {"X": x})[0]


class PortabilityTests(unittest.TestCase):
    def test_all_int8_codes_and_different_encodings(self):
        for shape in [(256,), (1, 1, 256), (1, 2, 128)]:
            for scale, output_scale in [(.125, .07), (.0001, .00021), (4., 1.)]:
                with self.subTest(shape=shape, scale=scale):
                    original = sample(scale, output_scale, shape)
                    # DQ followed by an identity Cast is the real inspected pattern.
                    original.graph.value_info.append(h.make_tensor_value_info("dx", T.FLOAT, shape))
                    patched, edits = patch.patch_graph(original)
                    x = (np.arange(-128, 128, dtype=np.float32) * scale).reshape(shape)
                    np.testing.assert_array_equal(execute(original, x), execute(patched, x))
                    self.assertEqual(len(edits), 3)
                    self.assertFalse(any(n.op_type in ("Neg", "Cast") for n in patched.graph.node))
                    old = {t.name: t.SerializeToString() for t in original.graph.initializer}
                    new = {t.name: t.SerializeToString() for t in patched.graph.initializer}
                    self.assertTrue(all(new[k] == v for k, v in old.items()))

    def test_dq_type_without_shape_information(self):
        original = sample()
        patched, edits = patch.patch_graph(original)
        self.assertEqual(len(edits), 3)
        onnx.checker.check_model(patched)

    def test_real_integer_conversion_is_retained(self):
        original = sample()
        original.graph.input[0].type.tensor_type.elem_type = T.INT32
        patched, _ = patch.patch_graph(original)
        self.assertTrue(any(n.name == "identity_cast" for n in patched.graph.node))
        x = np.arange(-128, 128, dtype=np.int32)
        np.testing.assert_array_equal(execute(original, x), execute(patched, x))

    def test_asymmetric_zero_point_is_rejected(self):
        original = sample()
        original.graph.initializer[2].CopyFrom(nh.from_array(np.array(3, np.int8), "zp"))
        with self.assertRaisesRegex(ValueError, "signed symmetric"):
            patch.patch_graph(original)

    def test_missing_output_quantization_is_rejected(self):
        original = sample()
        original.graph.node[-2].op_type = "Identity"
        del original.graph.node[-2].input[1:]
        with self.assertRaisesRegex(ValueError, "QDQ pattern"):
            patch.patch_graph(original)

    def test_entire_32_graph_bundle_and_checksums(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "original", Path(temp) / "patched"
            (source / "int8").mkdir(parents=True)
            files = {}
            for i in range(32):
                path = source / "int8" / f"part_{i:02}.onnx"
                onnx.save_model(sample(), str(path))
                files[str(path.relative_to(source))] = patch.sha(path)
            (source / "asset.txt").write_text("unchanged asset")
            files["asset.txt"] = patch.sha(source / "asset.txt")
            (source / "manifest.json").write_text(json.dumps({"files": files}))
            result = subprocess.run([sys.executable, str(SCRIPT), "--source", str(source),
                                     "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(list((output / "int8").glob("*.onnx"))), 32)
            for name, digest in manifest["files"].items():
                self.assertEqual(patch.sha(output / name), digest)
                self.assertEqual(patch.sha(source / name), files[name])
            self.assertEqual((output / "asset.txt").read_text(), "unchanged asset")
            repeat = subprocess.run([sys.executable, str(SCRIPT), "--source", str(source),
                                     "--output", str(output)], capture_output=True, text=True)
            self.assertNotEqual(repeat.returncode, 0)


if __name__ == "__main__":
    unittest.main()
