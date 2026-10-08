import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

from benchmark import load_gold, load_manifest
from src.benchmark_suite import audit_answer, decode_measurements, main, report, score
from src.rca_local.core import build_prompt, load_history, make_catalog, validate_selection
from src.rca_local.pipeline import run_pipeline
from src.compare_suites import load_pair, write_report

ROOT = Path(__file__).resolve().parents[1]


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / "syn_data/product_a_scenarios_v1",
                        self.root / "syn_data/product_a_scenarios_v1")
        self.manifest, self.digest = load_manifest(ROOT / "eval/product_a_v1_50.json")
        self.current = next(x["current"] for x in self.manifest["items"] if x["category"] == "no_answer")
        self.bundle = {"current": self.current, "matches": [
            m for m in load_history(self.root, self.current).values()
            if m["records"]["observation"]["original_chunk"]["failed_test"] == 210]}

    def run_case(self, variant=None):
        calls = []
        def worker(mode, source, target):
            if mode == "retrieve":
                result = copy.deepcopy(self.bundle)
            else:
                calls.append(mode)
                self.assertTrue(json.loads(source.read_text())["profile"])
                catalog_data = json.loads((source.parent / "catalog.json").read_text())
                catalog, aliases = catalog_data["catalog"], catalog_data["aliases"]
                result = {"text": json.dumps({"status": "unconfirmed", **{
                    field: [a for a, key in aliases.items() if catalog[key]["kind"] == kind][:1]
                    for field, kind in (("cause_ids", "cause"), ("check_ids", "check"))}})}
            target.write_text(json.dumps(result))
            return result
        with patch("src.rca_local.pipeline.settings", return_value=(self.root, None, None, None)), \
             patch("src.rca_local.pipeline.run_worker", side_effect=worker):
            answer = run_pipeline(self.current, experiment={"variant": variant, "max_causes": 3} if variant else None)
        base = "rca_experiments" if variant else "rca_app"
        return answer, self.root / "artifacts" / base / answer["run_id"], calls

    def test_normal_ui_still_refuses_without_generation(self):
        answer, folder, calls = self.run_case()
        self.assertEqual((answer["status"], calls), ("refuse", []))
        self.assertNotIn("experiment", answer)
        self.assertTrue(folder.is_dir())

    def test_full_experiment_refuses_and_uses_separate_artifact_path(self):
        answer, folder, calls = self.run_case("full")
        self.assertEqual((answer["status"], calls), ("refuse", []))
        self.assertEqual(folder.parent.name, "rca_experiments")

    def test_removing_verifier_exposes_unsupported_model_selection(self):
        answer, folder, calls = self.run_case("no_verifier")
        self.assertEqual((answer["status"], calls), ("unconfirmed", ["generate"]))
        self.assertFalse(json.loads((folder / "verification.json").read_text())["enforced"])
        audit = audit_answer(self.root, folder, answer)
        self.assertEqual(audit["citation_count"], audit["valid_citation_count"])
        self.assertEqual(len(audit["unsupported_ids"]), 2)

    def test_no_refusal_retains_rejection_audit_but_returns_model_selection(self):
        answer, folder, _ = self.run_case("no_refusal")
        verification = json.loads((folder / "verification.json").read_text())
        self.assertEqual(answer["status"], "unconfirmed")
        self.assertTrue(verification["enforced"])
        self.assertFalse(verification["refusal_enforced"])
        self.assertEqual(len(verification["rejected"]), 2)
        prompt = json.loads((folder / "prompt.json").read_text())["prompt"]
        self.assertIn("Do not refuse", prompt)
        self.assertNotIn("unsupported, refuse", prompt)

    def test_three_causes_are_only_accepted_when_explicitly_enabled(self):
        catalog = make_catalog(self.bundle)
        prompt, aliases = build_prompt(self.bundle, catalog, max_causes=3)
        answer = {"status": "unconfirmed", **{
            field: [a for a, key in aliases.items() if catalog[key]["kind"] == kind][:limit]
            for field, kind, limit in (("cause_ids", "cause", 3), ("check_ids", "check", 2))}}
        self.assertEqual(len(answer["cause_ids"]), 3)
        with self.assertRaisesRegex(ValueError, "maximum is 2"):
            validate_selection(json.dumps(answer), aliases, catalog)
        self.assertEqual(len(validate_selection(json.dumps(answer), aliases, catalog, max_causes=3)[1]["cause_ids"]), 3)
        self.assertIn("THREE C IDs", prompt)

    def test_decode_rate_uses_native_measurements_and_counts_retries(self):
        (self.root / "profile.json").write_text(json.dumps({"generation": {"generated_tokens": 20, "tokens_per_second": 10}}))
        retry = self.root / "retry_1"
        retry.mkdir()
        (retry / "profile.json").write_text(json.dumps({"generation": {"generated_tokens": 10, "tokens_per_second": 5}}))
        self.assertEqual(decode_measurements(self.root, "jetson"), {"decode_tokens": 30, "decode_seconds": 4, "decode_samples": 2})

    def test_scoring_distinguishes_refusal_rejection_and_false_answer(self):
        labels, origin = load_gold(ROOT / "eval/product_a_v1_50_proxy_gold.json", self.manifest, self.digest, True)
        rows = []
        for item in self.manifest["items"]:
            label = labels[item["id"]]
            rows.append({"id": item["id"], "status": label["expected_status"], "cause_ids": label["acceptable_cause_ids"][:1],
                         "check_ids": label["acceptable_check_ids"][:1], "latency_seconds": 10,
                         "citation_count": 2, "valid_citation_count": 2, "decode_tokens": 10, "decode_seconds": 1,
                         "decode_samples": 1, "mean_power_watts": 5, "peak_power_watts": 6, "energy_joules": 50})
        summary = score(self.manifest["items"], rows, labels, origin, .50)
        self.assertEqual(summary["proxy_answer_acceptance_rate"], 1)
        self.assertEqual(summary["refusal_rate"], .2)
        self.assertEqual(summary["proxy_top3_cause_hit_rate_on_answered"], 1)
        self.assertIsNone(summary["top3_cause_hit_rate_on_answered"])
        self.assertAlmostEqual(summary["electricity_cost_per_1000_triages"], 50 / 3600 * .50)
        rows[-1].update(status="unconfirmed", cause_ids=["wrong"], check_ids=["wrong"])
        rows[-2].update(status="rejected")
        summary = score(self.manifest["items"], rows, labels, origin)
        self.assertEqual(summary["false_answer_count"], 1)
        self.assertEqual(summary["rejection_rate"], .02)
        self.assertEqual(summary["no_answer_refusal_count"], 8)
        report(self.root, {"backend": "jetson", "variants": {"full": {"summary": summary}}})
        self.assertTrue((self.root / "REPORT.md").is_file())
        self.assertIn("unmeasured", (self.root / "REPORT.md").read_text())

    def test_interrupted_suite_resumes_without_repeating_completed_questions(self):
        out = self.root / "suite"
        argv = ["benchmark_suite", "--backend", "rb3", "--data-root", str(self.root), "--output-dir", str(out), "--resume"]
        calls = []
        fail_once = [True]
        def pipeline(current, *, experiment):
            if len(calls) == 6 and fail_once[0]:
                fail_once[0] = False
                raise RuntimeError("interruption")
            calls.append((experiment["variant"], current["dut_id"]))
            return {"run_id": f"{len(calls):032x}", "status": "refuse", "generation_attempts": 0,
                    "causes": [], "checks": []}
        with patch("sys.argv", argv), patch("src.benchmark_suite.environment", return_value={}), \
             patch.dict("os.environ", {}, clear=False), \
             patch("src.rca_local.pipeline.run_pipeline", side_effect=pipeline), \
             patch("src.benchmark_suite.audit_answer", return_value={"citation_count": 0, "valid_citation_count": 0}), \
             patch("src.benchmark_suite.decode_measurements", return_value={"decode_tokens": 0, "decode_seconds": 0, "decode_samples": 0}), \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                main()
            self.assertEqual(len(json.loads((out / "progress.json").read_text())["variants"]["full"]), 6)
            main()
            self.assertEqual(len(calls), 200)
            self.assertEqual(len(set(calls)), 200)
            self.assertTrue((out / "suite.json").is_file())
            main()
            self.assertEqual(len(calls), 200)

    def test_comparison_refuses_different_protocol_and_renders_missing_power(self):
        from src.benchmark_suite import PROTOCOL, VARIANTS
        base = {"protocol": PROTOCOL, "manifest_sha256": "a", "gold_sha256": "b", "label_origin": "ai_proxy_from_six_synthetic_scenarios",
                "max_causes": 3, "source_sha256": {"observation": "c"}, "code_sha256": {"pipeline": "d"},
                "variants": {v: {"summary": {"count": 50, "median_latency_seconds": 10, "p95_latency_seconds": 11,
                    "proxy_top1_cause_hit_rate_on_answered": 1, "proxy_false_answer_rate_per_answer": 0}} for v in VARIANTS}}
        jetson, rb3 = self.root / "jetson.json", self.root / "rb3.json"
        jetson.write_text(json.dumps({**base, "backend": "jetson"}))
        rb3.write_text(json.dumps({**base, "backend": "rb3"}))
        boards = load_pair(jetson, rb3)
        out = self.root / "comparison"
        write_report(boards, out)
        for path in out.glob("*.svg"):
            ET.parse(path)
        self.assertIn("unmeasured", (out / "comparison.md").read_text())
        rb3.write_text(json.dumps({**base, "backend": "rb3", "max_causes": 2}))
        with self.assertRaisesRegex(ValueError, "max_causes"):
            load_pair(jetson, rb3)


if __name__ == "__main__":
    unittest.main()
