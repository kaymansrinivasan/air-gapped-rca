"""Exercise failure capture without needing a QNN device or Qwen weights."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "scripts/diagnose_qwen_qnn_execute.py"
SPEC = importlib.util.spec_from_file_location("qnn_diagnostic", SOURCE)
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


class FailingSession:
    def run(self, outputs, feed, *args, **kwargs):
        feed["state"][:] = 999  # Simulate corruption during the failed call.
        raise RuntimeError("simulated QNN failure")

    def disable_fallback(self):
        return "delegated"


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "layer_14.onnx"
        self.model.write_bytes(b"unchanged-model")
        self.runner = self.root / "runner.py"
        self.runner.write_text("# unchanged runner\n")
        self.args = argparse.Namespace(target="layer_14", output=self.root, runner=self.runner)

    def capture(self):
        proxy = diag.SessionCapture(FailingSession(), self.model, self.args)
        state = np.array([1, 2, 3], dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "simulated QNN failure"):
            proxy.run(None, {"state": state})
        return proxy

    def test_capture_is_pre_failure_and_source_files_unchanged(self):
        proxy = self.capture()
        record, feed = diag.verify_fixture(self.args)
        np.testing.assert_array_equal(feed["state"], [1, 2, 3])
        self.assertEqual(record["call_number"], 1)
        self.assertEqual(record["inputs"]["state"]["dtype"], "float32")
        self.assertEqual(self.model.read_bytes(), b"unchanged-model")
        self.assertEqual(self.runner.read_text(), "# unchanged runner\n")
        self.assertEqual(proxy.disable_fallback(), "delegated")

    def test_changed_graph_or_fixture_is_refused_before_replay(self):
        self.capture()
        self.model.write_bytes(b"different-model")
        with self.assertRaisesRegex(RuntimeError, "Changed file"):
            diag.verify_fixture(self.args)
        self.model.write_bytes(b"unchanged-model")
        with (self.root / "failed_inputs.npz").open("wb") as stream:
            np.savez(stream, state=np.array([8], dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "Changed file"):
            diag.verify_fixture(self.args)

    def test_other_layer_failure_is_not_mislabeled_as_target(self):
        other = self.root / "layer_13.onnx"
        other.write_bytes(b"another-model")
        proxy = diag.SessionCapture(FailingSession(), other, self.args)
        with self.assertRaisesRegex(RuntimeError, "simulated QNN failure"):
            proxy.run(None, {"state": np.ones(2, dtype=np.float32)})
        self.assertFalse((self.root / "fault.json").exists())


if __name__ == "__main__":
    unittest.main()
