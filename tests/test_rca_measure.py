"""An audit must follow a saved source, not trust the benchmark row alone."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from benchmark import load_manifest
from src.measure import audit, load_power, parse_tegrastats_power
from src.rca_local.core import load_history, make_catalog, render_answer

ROOT = Path(__file__).resolve().parents[1]


class MeasureTests(unittest.TestCase):
    def test_reaudits_saved_citation_and_rejects_tampering(self):
        manifest, digest = load_manifest(ROOT / "eval/product_a_v1_50.json")
        current = manifest["items"][0]["current"]
        matches = [case for case in load_history(ROOT, current).values()
                   if case["records"]["observation"]["original_chunk"]["failed_test"] == 100]
        bundle = {"current": current, "matches": matches}
        catalog = make_catalog(bundle)
        cause = next(x for x in catalog.values() if x["kind"] == "cause"
                     and "TSTIN" not in x["text"])
        check = next(x for x in catalog.values() if x["kind"] == "check")
        selected = {"cause_ids": [cause], "check_ids": [check]}
        from src.verify import verify_selection
        status, _, rejected = verify_selection(ROOT, bundle, "unconfirmed", selected)
        self.assertEqual((status, rejected), ("unconfirmed", []))
        with tempfile.TemporaryDirectory() as scratch:
            data = Path(scratch)
            shutil.copytree(ROOT / "syn_data/product_a_scenarios_v1",
                            data / "syn_data/product_a_scenarios_v1")
            results = data / "result.json"
            run_id = "a" * 32
            folder = data / "artifacts/rca_app" / run_id
            folder.mkdir(parents=True)
            answer = render_answer(bundle, "unconfirmed", selected)
            answer["run_id"] = run_id
            (folder / "answer.json").write_text(json.dumps(answer))
            (folder / "evidence.json").write_text(json.dumps(bundle))
            row = {"id": "PA-EVAL-001", "run_id": run_id, "status": "unconfirmed",
                   "cause_ids": [cause["id"]], "check_ids": [check["id"]],
                   "latency_seconds": 20}
            results.write_text(json.dumps({"backend": "jetson_tensorrt_edge_llm",
                "summary": {"count": 1}, "results": [row], "manifest_sha256": digest,
                "label_origin": "ai_proxy_from_six_synthetic_scenarios"}))
            report = audit(results, data)
            self.assertEqual(report["summary"]["citation_validity"], 1.0)
            self.assertIsNone(report["summary"]["average_power_watts"])
            answer["causes"][0]["text"] = "invented cause"
            (folder / "answer.json").write_text(json.dumps(answer))
            with self.assertRaises(ValueError):
                audit(results, data)

    def test_power_samples_need_real_values_for_every_case(self):
        self.assertEqual(parse_tegrastats_power("RAM 5974/7607MB VDD_IN 6948mW/6948mW"), 6.948)
        self.assertIsNone(parse_tegrastats_power("RAM 5974/7607MB"))
        with tempfile.TemporaryDirectory() as scratch:
            file = Path(scratch) / "power.csv"
            file.write_text("case_id,watts\nPA-EVAL-001,5.5\n")
            self.assertEqual(load_power(file, ["PA-EVAL-001"]), {"PA-EVAL-001": [5.5]})
            with self.assertRaisesRegex(ValueError, "Every"):
                load_power(file, ["PA-EVAL-001", "PA-EVAL-002"])


if __name__ == "__main__":
    unittest.main()
