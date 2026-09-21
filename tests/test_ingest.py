"""Block 2 acceptance and corruption tests; uses only the standard library."""

from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest.mock import patch

from src import ingest


ROOT = Path(__file__).resolve().parents[1]


class IngestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks, cls.audit, cls.question = ingest.build(ROOT)
        cls.config = json.loads((ROOT / ingest.DEFAULT_SPLIT).read_bytes())

    def fixture(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        shutil.copytree(ROOT / "syn_data", root / "syn_data")
        return root

    def save_config(self, root, config):
        (root / ingest.DEFAULT_SPLIT).write_bytes(ingest.json_bytes(config))

    def replace_document(self, root, path, document):
        data = ingest.json_bytes(document)
        (root / path).write_bytes(data)
        config = copy.deepcopy(self.config)
        for role in ("historical", "query"):
            for entry in config[role]["files"]:
                if entry["path"] == path:
                    entry["git_blob_sha"] = ingest.git_blob_sha(data)
        self.save_config(root, config)

    def test_complete_split_and_no_answer_leakage(self):
        historical = self.chunks["historical"]
        current = self.chunks["current"]
        self.assertEqual({c["incident_id"] for c in historical},
                         {f"SYN-A595-{i:03}" for i in range(1, 7)})
        self.assertEqual(len({c["evidence_id"] for c in historical}), 25)
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["incident_id"], "SYN-A595-007")
        self.assertEqual(current[0]["origin"], "SIMULATED")
        self.assertEqual(current[0]["original_citations"], [])
        self.assertIsNone(current[0]["source_anchor"])
        query = json.loads(current[0]["text"])
        self.assertEqual([o["status"] for o in query["observations"]], ["FAIL", "PASS", "FAIL"])
        self.assertIsNone(query["actual_physical_cause"])
        allowed = {e["path"] for role in ("historical", "query") for e in self.config[role]["files"]}
        self.assertEqual({c["source_file"] for c in historical + current}, allowed)
        self.assertEqual(self.question, self.config["query"]["question"])

    def test_every_quote_and_line_is_preserved_independently(self):
        covered = defaultdict(set)
        for chunk in self.chunks["historical"] + self.chunks["current"]:
            for span in [chunk, *chunk["context"], *chunk["original_citations"]]:
                data = (ROOT / span["source_file"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), span["source_sha256"])
                # Independent extraction, including line terminators.
                lines = data.decode("utf-8").splitlines(keepends=True)
                lo, hi = span["source_lines"]["start"], span["source_lines"]["end"]
                self.assertGreaterEqual(lo, 1)
                self.assertLessEqual(hi, len(lines))
                self.assertEqual(span["text"], "".join(lines[lo - 1:hi]))
                covered[span["source_file"]].update(range(lo, hi + 1))
        for entry in self.audit["inputs"]:
            self.assertEqual(covered[entry["source_file"]], set(range(1, entry["lines"] + 1)))

    def test_complete_test_records_and_retest_context(self):
        by_file = defaultdict(list)
        for chunk in self.chunks["historical"]:
            if chunk["stage"] != "retest":
                continue
            by_file[chunk["source_file"]].append(chunk)
            document = json.loads((ROOT / chunk["source_file"]).read_bytes())
            if chunk["kind"] == "retest_group":
                # Parse the actual exact test excerpt, not a metadata projection.
                actual = json.loads("[" + chunk["text"].strip().rstrip(",") + "]")
                self.assertEqual(actual, [document["tests"][i] for i in chunk["test_indices"]])
                context = "".join(c["text"] for c in chunk["context"])
                for value in (document["synthetic_device_key"], document["scope"], document["label"]):
                    self.assertIn(value, context)
                for cited in document["cites"]:
                    self.assertIn(cited, context)
        total = 0
        for path, chunks in by_file.items():
            document = json.loads((ROOT / path).read_bytes())
            indices = [i for c in chunks for i in c["test_indices"]]
            self.assertEqual(indices, list(range(len(document["tests"]))))
            total += len(indices)
        self.assertEqual(total, 373)
        self.assertEqual(len(by_file), 7)

    def test_wrong_device_and_shared_failure_not_erased(self):
        wrong = [c for c in self.chunks["historical"] if c["evidence_id"] == "SYN-A595-006-RETEST-1"]
        self.assertTrue(wrong)
        for chunk in wrong:
            self.assertFalse(chunk["device_matches_incident"])
            self.assertTrue(chunk["device_key"].endswith("-OTHER-DEVICE"))
        failures = {c["incident_id"]: c for c in self.chunks["historical"] if c["stage"] == "failure"}
        self.assertEqual(failures["SYN-A595-001"]["shared_source_failure_key"],
                         failures["SYN-A595-002"]["shared_source_failure_key"])
        self.assertNotEqual(failures["SYN-A595-001"]["chunk_id"], failures["SYN-A595-002"]["chunk_id"])

    def test_deterministic_offline_and_only_allowlisted_reads(self):
        allowed = {e["source_file"] for e in self.audit["inputs"]}
        allowed.update(self.audit["resolved_original_files"])
        allowed.add(ingest.DEFAULT_SPLIT)
        read_bytes = Path.read_bytes
        def guarded_read(path):
            self.assertIn(path.resolve().relative_to(ROOT).as_posix(), allowed)
            return read_bytes(path)
        with patch.object(Path, "read_bytes", guarded_read), \
                patch.object(socket, "socket", side_effect=AssertionError("Network forbidden")):
            chunks, audit, question = ingest.build(ROOT)
        self.assertEqual(chunks, self.chunks)
        self.assertEqual(audit, self.audit)
        self.assertEqual(question, self.question)

    def test_changed_evidence_fails_hash_check(self):
        root = self.fixture()
        path = root / self.config["historical"]["files"][0]["path"]
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ingest.IngestError, "Git blob hash mismatch"):
            ingest.build(root)

    def test_changed_original_fails_hash_check(self):
        root = self.fixture()
        path = root / "syn_data/source/a595_tester_log.txt"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ingest.IngestError, "Original hash mismatch"):
            ingest.build(root)

    def test_bad_citation_fails_even_with_updated_config_hash(self):
        root = self.fixture()
        path = self.config["historical"]["files"][0]["path"]
        document = json.loads((root / path).read_bytes())
        document["source_citation"]["line_start"] += 1
        self.replace_document(root, path, document)
        with self.assertRaisesRegex(ingest.IngestError, "Original excerpt mismatch"):
            ingest.build(root)

    def test_query_cannot_enter_history(self):
        root = self.fixture()
        config = copy.deepcopy(self.config)
        config["historical"]["incident_ids"].append("SYN-A595-007")
        config["historical"]["files"].extend(config["query"]["files"])
        self.save_config(root, config)
        with self.assertRaisesRegex(ingest.IngestError, "Current incident"):
            ingest.build(root)

    def test_answer_key_cannot_be_allowlisted_as_evidence(self):
        root = self.fixture()
        config = copy.deepcopy(self.config)
        config["historical"]["files"].append({"path": "syn_data/reference/answer_key.jsonl"})
        self.save_config(root, config)
        with self.assertRaisesRegex(ingest.IngestError, "Not an allowed incident evidence path"):
            ingest.build(root)

    def test_future_current_evidence_is_rejected(self):
        root = self.fixture()
        config = copy.deepcopy(self.config)
        config["query"]["files"].append({"path": "syn_data/incidents/SYN-A595-007/simulated_action.json"})
        self.save_config(root, config)
        with self.assertRaisesRegex(ingest.IngestError, "future evidence"):
            ingest.build(root)

    def test_missing_duplicate_and_excluded_inputs_fail(self):
        for mode in ("missing", "duplicate", "excluded"):
            with self.subTest(mode=mode):
                root = self.fixture()
                config = copy.deepcopy(self.config)
                entry = config["historical"]["files"][0]
                if mode == "missing":
                    (root / entry["path"]).unlink()
                elif mode == "duplicate":
                    config["historical"]["files"].append(entry)
                else:
                    config["evaluator_only"]["held_out_files"] = [entry["path"]]
                self.save_config(root, config)
                with self.assertRaises((OSError, ingest.IngestError)):
                    ingest.build(root)

    def test_unsafe_paths_and_duplicate_json_keys_rejected(self):
        for path in ("../escape", "C:/escape", "/escape", "syn_data\\escape"):
            with self.assertRaises(ingest.IngestError):
                ingest.source_path(ROOT, path)
        with self.assertRaisesRegex(ingest.IngestError, "Duplicate JSON key"):
            ingest.read_json(b'{"tests": [], "tests": [1]}')
        with self.assertRaises(ingest.IngestError):
            ingest.read_json(b'{"value": NaN}')

    def test_crlf_unicode_braces_and_minified_records(self):
        document = {"incident_id": "X", "evidence_id": "X-R", "synthetic_device_key": "D",
                    "origin": "SIMULATED", "label": "fictional", "tests": [
                        {"test_number": i, "kind": "parametric", "unit": "mA",
                         "test_name": 'quoted "tests": [ and brace } and micro μ',
                         "value": i, "origin": "SIMULATED"} for i in range(20)], "cites": ["X-A"]}
        for compact in (False, True):
            text = json.dumps(document, ensure_ascii=False, indent=None if compact else 2).replace("\n", "\r\n")
            source = ingest.Source("sample.json", text.encode())
            chunks = ingest.chunk_record({"source": source, "document": document,
                                          "stage": "retest", "role": "historical"}, {"X": "D"}, [], 512)
            self.assertEqual([i for c in chunks for i in c["test_indices"]], list(range(20)))
            if compact:
                self.assertEqual(len(chunks), 1)
                self.assertTrue(chunks[0]["over_target"])
            else:
                self.assertGreater(len(chunks), 1)
                for chunk in chunks:
                    self.assertIn("\r\n", chunk["text"])

    def test_bundle_hashes_reproducibility_and_no_overwrite(self):
        root = self.fixture()
        before = {p: p.read_bytes() for p in (root / "syn_data").rglob("*") if p.is_file()}
        for name in ("out1", "out2"):
            chunks, audit, question = ingest.build(root)
            ingest.write_bundle(root / name, chunks, audit, question)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        for file in (root / "out1").iterdir():
            self.assertEqual(file.read_bytes(), (root / "out2" / file.name).read_bytes())
        manifest = json.loads((root / "out1/chunk_manifest.json").read_bytes())
        for name, info in manifest["outputs"].items():
            self.assertEqual(hashlib.sha256((root / "out1" / name).read_bytes()).hexdigest(), info["sha256"])
        with self.assertRaisesRegex(ingest.IngestError, "already exists"):
            ingest.write_bundle(root / "out1", chunks, audit, question)
        self.assertEqual(ingest.main(["--repo-root", str(root), "--output", str(root / "syn_data/new")]), 1)
        self.assertFalse((root / "syn_data/new").exists())


if __name__ == "__main__":
    unittest.main()
