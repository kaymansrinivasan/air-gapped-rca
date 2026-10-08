import json
from pathlib import Path
import re
import unittest

from benchmark import load_manifest, load_gold, summarize
from src.rca_local.core import load_history, make_catalog
from src.verify import verify_selection

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "eval/product_a_v1_50.json"


class EvaluationTests(unittest.TestCase):
    def test_frozen_questions_have_distinct_log_provenance(self):
        manifest, digest = load_manifest(MANIFEST)
        items = manifest["items"]
        self.assertEqual(len(items), 50)
        self.assertEqual(len({x["current"]["dut_id"] for x in items}), 50)
        self.assertEqual(sum(x["category"] == "no_answer" for x in items), 10)
        self.assertEqual(len(digest), 64)
        self.assertTrue(all(x["gold"]["review_status"] == "pending_engineer_review" for x in items))

    def test_all_no_answer_probes_refuse_even_if_model_selects_history(self):
        manifest, _ = load_manifest(MANIFEST)
        for item in manifest["items"]:
            if item["category"] != "no_answer":
                continue
            current = item["current"]
            history = load_history(ROOT, current)
            matches = [case for case in history.values()
                       if case["records"]["observation"]["original_chunk"]["failed_test"] == 210]
            bundle = {"current": current, "matches": matches}
            catalog = make_catalog(bundle)
            selected = {field: [v for v in catalog.values() if v["kind"] == kind]
                        for field, kind in (("cause_ids", "cause"), ("check_ids", "check"))}
            status, _, _ = verify_selection(ROOT, bundle, "unconfirmed", selected)
            self.assertEqual(status, "refuse", item["id"])

    def test_pending_gold_cannot_start_an_official_run(self):
        manifest, digest = load_manifest(MANIFEST)
        path = ROOT / "eval/product_a_v1_50_review_template.json"
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            load_gold(path, manifest, digest)
        with self.assertRaisesRegex(ValueError, "different manifest"):
            load_gold(path, manifest, "0" * 64)

    def test_ai_proxy_is_explicit_and_never_scored_as_engineer_accuracy(self):
        manifest, digest = load_manifest(MANIFEST)
        path = ROOT / "eval/product_a_v1_50_proxy_gold.json"
        with self.assertRaisesRegex(ValueError, "allow-proxy-gold"):
            load_gold(path, manifest, digest)
        labels, origin = load_gold(path, manifest, digest, allow_proxy=True)
        self.assertEqual(len(labels), 50)
        self.assertEqual(sum(x["expected_status"] == "refuse" for x in labels.values()), 10)
        results = []
        for item in manifest["items"]:
            label = labels[item["id"]]
            results.append({"status": label["expected_status"], "latency_seconds": 1,
                            "cause_ids": label["acceptable_cause_ids"][:1],
                            "check_ids": label["acceptable_check_ids"][:1]})
        summary = summarize(manifest["items"], results, labels, origin)
        self.assertEqual(summary["proxy_exact_answer_agreement"], 1.0)
        self.assertIsNone(summary["exact_answer_accuracy"])
        self.assertIsNone(summary["false_answer_rate_per_answer"])

    def test_offline_review_page_has_only_frozen_unreviewed_inputs(self):
        page = (ROOT / "eval/product_a_v1_50_review.html").read_text()
        payload = json.loads(re.search(
            r'<script type="application/json" id="payload">(.*?)</script>',
            page, re.S)[1])
        manifest, digest = load_manifest(MANIFEST)
        self.assertEqual(payload["manifest_sha256"], digest)
        self.assertEqual(len(payload["items"]), len(manifest["items"]))
        self.assertTrue(all(x["review_status"] == "pending_engineer_review"
                            for x in payload["items"]))


if __name__ == "__main__":
    unittest.main()
