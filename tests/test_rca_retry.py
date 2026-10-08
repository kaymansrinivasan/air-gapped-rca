"""Exercise selection retries without loading models or modifying real evidence."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.rca_local.core import EvidenceError, build_prompt, load_history, make_catalog, observation_from_form
from src.rca_local.pipeline import run_pipeline


ROOT = Path(__file__).resolve().parents[1]
BAD = '{"status":"unconfirmed","cause_ids":["C1","C2","C5"],"check_ids":["K3","K4","K6","K7"]}'


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / "syn_data", self.root / "syn_data")
        self.current = observation_from_form({
            "product": "Product_A", "failed_test": 100, "dut_id": "Product_A-L06-W02-D009",
            "pins": "TSTIN", "observation": "100 Open/Short-: FAIL", "question": "What should I check?",
        })
        history = load_history(self.root, self.current)
        self.bundle = {"current": self.current, "matches": [
            history["Product_A-L01-W03-D012"], history["Product_A-L01-W03-D005"]]}
        self.catalog = make_catalog(self.bundle)
        self.prompt, self.aliases = build_prompt(self.bundle, self.catalog)
        self.valid = json.dumps({"status": "unconfirmed", **{
            field: [a for a, key in self.aliases.items() if self.catalog[key]["kind"] == kind][:2]
            for field, kind in (("cause_ids", "cause"), ("check_ids", "check"))}})
        self.calls = []

    def run_job(self, outputs, mutate_source=False):
        outputs = iter(outputs)

        def worker(mode, source, target):
            if mode == "retrieve":
                result = copy.deepcopy(self.bundle)
            else:
                self.calls.append(source)
                output = next(outputs)  # An unplanned third call fails the test.
                if isinstance(output, Exception):
                    raise output
                result = {"text": output, "finish_reason": "end-of-sequence"}
                if mutate_source:
                    path = self.root / self.bundle["matches"][0]["sources"]["investigation"]["file"]
                    path.write_text(path.read_text() + "\n")
            target.write_text(json.dumps(result))
            return result

        with patch("src.rca_local.pipeline.settings", return_value=(self.root, None, None, None)), \
                patch("src.rca_local.pipeline.run_worker", side_effect=worker):
            answer = run_pipeline(self.current)
        return answer, self.root / "artifacts/rca_app" / answer["run_id"]

    def test_valid_first_response_needs_no_retry(self):
        answer, folder = self.run_job([self.valid])
        self.assertEqual(answer["status"], "unconfirmed")
        self.assertEqual(answer["generation_attempts"], 1)
        self.assertFalse((folder / "retry_1").exists())

    def test_observed_overselection_retries_and_preserves_both_attempts(self):
        answer, folder = self.run_job([BAD, self.valid])
        self.assertEqual(answer["status"], "unconfirmed")
        self.assertEqual(answer["generation_attempts"], 2)
        self.assertEqual(json.loads((folder / "model.json").read_text())["text"], BAD)
        self.assertEqual(json.loads((folder / "retry_1/model.json").read_text())["text"], self.valid)
        error = json.loads((folder / "validation.json").read_text())["error"]
        self.assertEqual(error, "cause_ids contains 3 IDs; maximum is 2.")
        retry_prompt = json.loads((folder / "retry_1/prompt.json").read_text())["prompt"]
        self.assertTrue(retry_prompt.startswith(self.prompt))
        self.assertNotIn(BAD, retry_prompt)
        for item in answer["causes"] + answer["checks"]:
            self.assertEqual(item["text"], self.catalog[item["id"]]["text"])
            self.assertEqual(item["citation"], self.catalog[item["id"]]["citation"])

    def test_two_invalid_responses_still_reject_without_trimming(self):
        answer, folder = self.run_job([BAD, BAD])
        self.assertEqual(answer["status"], "rejected")
        self.assertEqual(answer["generation_attempts"], 2)
        self.assertEqual(answer["causes"], [])
        self.assertEqual(answer["checks"], [])
        self.assertIn("after two attempts", answer["error"])
        self.assertFalse(json.loads((folder / "retry_1/validation.json").read_text())["accepted"])

    def test_invalid_ids_or_wrong_evidence_type_cannot_pass_retry(self):
        for invalid in ("C999", "K3"):
            with self.subTest(invalid=invalid):
                wrong = json.loads(self.valid)
                wrong["cause_ids"] = [invalid]
                answer, _ = self.run_job([BAD, json.dumps(wrong)])
                self.assertEqual(answer["status"], "rejected")

    def test_retry_may_refuse(self):
        refusal = '{"status":"refuse","cause_ids":[],"check_ids":[]}'
        answer, _ = self.run_job([BAD, refusal])
        self.assertEqual(answer["status"], "refuse")
        self.assertEqual(answer["causes"], [])
        self.assertEqual(answer["checks"], [])

    def test_runtime_failure_does_not_retry(self):
        answer, _ = self.run_job([EvidenceError("TensorRT failed")])
        self.assertEqual(answer["status"], "rejected")
        self.assertEqual(answer["generation_attempts"], 1)

    def test_source_integrity_failure_does_not_retry(self):
        answer, _ = self.run_job([BAD], mutate_source=True)
        self.assertEqual(answer["status"], "rejected")
        self.assertIn("source changed", answer["error"])
        self.assertEqual(answer["generation_attempts"], 1)

    def test_retry_budget_failure_preserves_original_and_rejects(self):
        answer, folder = self.run_job([BAD, EvidenceError("Evidence exceeds the engine budget")])
        self.assertEqual(answer["status"], "rejected")
        self.assertIn("budget", answer["error"])
        self.assertEqual(json.loads((folder / "model.json").read_text())["text"], BAD)
        self.assertEqual(answer["generation_attempts"], 2)


if __name__ == "__main__":
    unittest.main()
