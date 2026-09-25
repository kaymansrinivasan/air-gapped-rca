"""Evidence integrity and retrieval behavior at the Block 2/3 boundary."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

from chunk import build, write_bundle
from src.index import build_index, query_index, source_chunks

ROOT = Path(__file__).resolve().parents[1]


class ProductAPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks, cls.manifest = build(ROOT)
        cls.temp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temp.name)
        write_bundle(cls.out / "chunks", cls.chunks, cls.manifest)
        cls.index_manifest = build_index(cls.out / "chunks", cls.out / "index")
        cls.db = cls.out / "index" / "product_a_index.sqlite3"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_every_chunk_quotes_its_exact_immutable_device_block(self):
        self.assertEqual(len(self.chunks), 3920)
        self.assertEqual(sum(c["failed_test"] is not None for c in self.chunks), 392)
        self.assertEqual(sum(c["failed_test"] is None for c in self.chunks), 3528)
        self.assertEqual({c["cause_status"] for c in self.chunks}, {"UNKNOWN"})
        self.assertEqual(len({c["chunk_id"] for c in self.chunks}), 3920)
        for chunk in self.chunks:
            data = (ROOT / chunk["source_file"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), chunk["source_sha256"])
            lines = data.decode().splitlines(keepends=True)
            bounds = chunk["source_lines"]
            self.assertEqual(chunk["source_text"], "".join(lines[bounds["start"]-1:bounds["end"]]))
            self.assertIn(chunk["source_text"], chunk["retrieval_text"])
        self.assertEqual(self.manifest["summary"]["executed_tests"]["total"], 11466)

    def test_stop_rule_and_category_examples(self):
        by_id = {c["dut_id"]: c for c in self.chunks}
        cases = [
            ("Product_A-L01-W03-D012", [100], 4, "TSTIN", 100),
            ("Product_A-L01-W01-D100", [100, 210], 3, "65.58", 210),
            ("Product_A-L01-W02-D005", [100, 210, 606], 2, "2347", 606),
            ("Product_A-L01-W01-D001", [100, 210, 606], 1, "Failing Pins:  0", None),
        ]
        for dut, tests, bin_no, evidence, failed in cases:
            chunk = by_id[dut]
            self.assertEqual((chunk["executed_tests"], chunk["bin"], chunk["failed_test"]),
                             (tests, bin_no, failed))
            self.assertIn(evidence, chunk["source_text"])

    def test_search_filters_failure_and_excludes_current_dut(self):
        self.assertEqual(self.index_manifest["documents"], 3920)
        self.assertEqual(self.index_manifest["failure_documents"], 392)
        self.assertFalse(self.index_manifest["semantic_embeddings"])
        for phrase, test in [("continuity Open/Short- TSTIN", 100),
                             ("IDD_Static high current", 210),
                             ("SCAN failed vector DTOP Product_A-L01-W02-D005", 606)]:
            for mode in ("keyword", "vector", "hybrid"):
                found = query_index(self.db, phrase, top_k=5, mode=mode)
                self.assertTrue(found, (phrase, mode))
                self.assertEqual({x["failed_test"] for x in found}, {test})
                self.assertNotIn("Product_A-L01-W02-D005", [x["dut_id"] for x in found])
                self.assertTrue(all(x["cause_status"] == "UNKNOWN" for x in found))
        passing = query_index(self.db, "Open/Short- passed", top_k=30, failures_only=False,
                              failed_test=None)
        self.assertTrue(any(x["bin"] == 1 for x in passing))

    def test_mutated_chunk_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            original = self.out / "chunks"
            (folder / "chunk_manifest.json").write_bytes((original / "chunk_manifest.json").read_bytes())
            data = (original / "product_a_dut_chunks.jsonl").read_bytes()
            (folder / "product_a_dut_chunks.jsonl").write_bytes(data.replace(b'CONTINUITY_FAIL', b'FALSE_CAUSE_FAIL', 1))
            with self.assertRaisesRegex(ValueError, "hash"):
                source_chunks(folder)


if __name__ == "__main__":
    unittest.main()
