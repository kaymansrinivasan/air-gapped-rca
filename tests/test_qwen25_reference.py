import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "scripts/prepare_qwen25_reference.py"
spec = importlib.util.spec_from_file_location("qwen25_reference", path)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class CheckpointIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {"model_type": "qwen2", "hidden_size": 896, "num_hidden_layers": 24,
                       "num_attention_heads": 14, "num_key_value_heads": 2,
                       "vocab_size": 151936, "tie_word_embeddings": True}
        for name in q.FILES:
            (self.root / name).write_text(json.dumps(self.config) if name == "config.json" else "fixture")
        self.manifest = {"model_id": q.MODEL, "revision": q.REVISION,
                         "files": {name: {"bytes": (self.root / name).stat().st_size,
                                          "sha256": q.digest(self.root / name)} for name in q.FILES}}
        self.guard = patch.object(q, "WEIGHTS_SHA256", q.digest(self.root / "model.safetensors"))
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def test_modified_same_size_tokenizer_is_rejected_without_overwriting(self):
        q.verify(self.root, self.manifest)
        target = self.root / "tokenizer.json"
        target.write_text("changed")
        with self.assertRaisesRegex(ValueError, "integrity failed: tokenizer.json"):
            q.verify(self.root, self.manifest)
        self.assertEqual(target.read_text(), "changed")

    def test_updated_manifest_cannot_hide_changed_weights(self):
        target = self.root / "model.safetensors"
        target.write_text("changed")
        self.manifest["files"]["model.safetensors"]["sha256"] = q.digest(target)
        with self.assertRaisesRegex(ValueError, "upstream checkpoint"):
            q.verify(self.root, self.manifest)

    def test_wrong_model_revision_and_configuration_are_rejected(self):
        self.manifest["revision"] = "different"
        with self.assertRaisesRegex(ValueError, "different model or revision"):
            q.verify(self.root, self.manifest)
        self.config["model_type"] = "qwen3_5_text"
        with self.assertRaisesRegex(ValueError, "configuration"):
            q.validate_config(self.config)


if __name__ == "__main__":
    unittest.main()
