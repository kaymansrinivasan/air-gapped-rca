import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from src.rca_local.core import (
    EvidenceError, adapt_procedure, build_prompt, load_history,
    make_catalog, observation_from_form, render_answer, validate_selection,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    current = json.loads((ROOT / "syn_data/product_a_scenarios_v1/case_07/observation.json").read_text())
    current["question"] = "What could explain the failure and what should I check?"
    history = load_history(ROOT, current)
    matches = [{**case, "rank": i, "vector_distance": d} for i, (case, d) in enumerate(
        [(history["Product_A-L01-W03-D012"], .5303), (history["Product_A-L01-W03-D005"], .6455)], 1)]
    bundle = {"current": current, "matches": matches}
    catalog = make_catalog(bundle)
    _, aliases = build_prompt(bundle, catalog)
    return bundle, catalog, aliases


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.bundle, self.catalog, self.aliases = fixture()

    def selection(self):
        causes = [a for a, key in self.aliases.items() if self.catalog[key]["kind"] == "cause"]
        checks = [a for a, key in self.aliases.items() if self.catalog[key]["kind"] == "check"]
        return {"status": "unconfirmed", "cause_ids": causes[:2], "check_ids": checks[:2]}

    def test_all_case_links_and_eight_sources_are_preserved(self):
        self.assertEqual(sum(len(m["sources"]) for m in self.bundle["matches"]), 8)
        self.assertEqual(len(load_history(ROOT, self.bundle["current"])), 6)

    def test_valid_selection_uses_source_owned_text_and_citations(self):
        status, selected = validate_selection(json.dumps(self.selection()), self.aliases, self.catalog)
        result = render_answer(self.bundle, status, selected)
        self.assertEqual(result["status"], "unconfirmed")
        for item in result["causes"] + result["checks"]:
            self.assertTrue(item["citation"]["record_id"].endswith("_investigation"))
            record = json.loads((ROOT / item["citation"]["file"]).read_text())
            allowed = record["possible_causes"] if item["kind"] == "cause" else [x["check"] for x in record["checks"]]
            self.assertIn(item["text"], allowed)

    def test_observed_bad_draft_is_rejected(self):
        bad = {"status": "unconfirmed", "candidates": [{"cause": "DUT-related electrical connection abnormality on TSTIN", "citations": ["S1"]}],
               "next_checks": [{"check": "Retest D005 after maintenance.", "citations": ["S8"]}],
               "limitations": ["Historical outcomes are synthetic and unreviewed."]}
        with self.assertRaises(EvidenceError):
            validate_selection(json.dumps(bad), self.aliases, self.catalog)

    def test_unrelated_historical_pin_cause_is_not_an_answer_choice(self):
        causes = [x["text"] for x in self.catalog.values() if x["kind"] == "cause"]
        self.assertIn("DUT-related electrical connection abnormality on TSTIN", causes)
        self.assertNotIn("DUT-related abnormality affecting XCKP, DTO2P and CCK", causes)
        self.assertEqual(len(self.bundle["matches"]), 2)

    def test_invented_ids_wrong_types_confirmed_and_duplicates_rejected(self):
        bad = []
        item = self.selection(); item["cause_ids"] = ["S1"]; bad.append(item)
        item = self.selection(); item["cause_ids"] = item["check_ids"][:1]; bad.append(item)
        item = self.selection(); item["status"] = "confirmed"; bad.append(item)
        item = self.selection(); item["check_ids"] = item["check_ids"][:1] * 2; bad.append(item)
        item = self.selection(); item["check_ids"] = "K3"; bad.append(item)
        item = self.selection(); item["extra"] = "invented text"; bad.append(item)
        item = self.selection(); item["cause_ids"] = []; bad.append(item)
        for value in bad:
            with self.subTest(value=value), self.assertRaises(EvidenceError):
                validate_selection(json.dumps(value), self.aliases, self.catalog)

    def test_duplicate_json_keys_and_truncated_json_rejected(self):
        for text in ['{"status":"refuse","status":"unconfirmed","cause_ids":[],"check_ids":[]}', '{"status":']:
            with self.assertRaises(EvidenceError):
                validate_selection(text, self.aliases, self.catalog)

    def test_refusal_has_no_suggestions(self):
        status, items = validate_selection('{"status":"refuse","cause_ids":[],"check_ids":[]}', self.aliases, self.catalog)
        self.assertEqual(status, "refuse")
        bad = self.selection(); bad["status"] = "refuse"
        with self.assertRaises(EvidenceError):
            validate_selection(json.dumps(bad), self.aliases, self.catalog)

    def test_historical_dut_is_adapted_only_in_procedure(self):
        original = "Retest D005 on both setups. Do not replace D0050."
        changed = adapt_procedure(original, "Product_A-L01-W03-D005", self.bundle["current"]["dut_id"])
        self.assertIn("current DUT (Product_A-L06-W01-D001)", changed)
        self.assertIn("D0050", changed)
        self.assertEqual(original, "Retest D005 on both setups. Do not replace D0050.")
        self.assertIn("current DUT", adapt_procedure("Retest Product_A-L01-W03-D005.", "Product_A-L01-W03-D005", "NEW-DUT"))

    def test_history_link_tampering_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(ROOT / "syn_data/product_a_scenarios_v1", root / "syn_data/product_a_scenarios_v1")
            path = root / "syn_data/product_a_scenarios_v1/case_01/action.json"
            data = json.loads(path.read_text()); data["dut_id"] = "WRONG"
            path.write_text(json.dumps(data))
            with self.assertRaises(EvidenceError):
                load_history(root, self.bundle["current"])

    def test_current_dut_cannot_leak_into_history(self):
        current = copy.deepcopy(self.bundle["current"])
        current["dut_id"] = "Product_A-L01-W03-D012"
        with self.assertRaises(EvidenceError):
            load_history(ROOT, current)

    def test_form_needs_valid_scope_and_supports_new_dut(self):
        data = {"dut_id": "NEW-DUT-123", "product": "Product_A", "failed_test": "100",
                "observation": "Continuity failed on TSTIN", "pins": "TSTIN", "question": "What should I check?"}
        current = observation_from_form(data)
        self.assertFalse(current["index_as_history"])
        self.assertEqual(current["test_result"]["failing_pins"], [{"name": "TSTIN"}])
        for key, value in [("product", "Unknown"), ("failed_test", "999"), ("observation", ""), ("question", ""), ("dut_id", "../path")]:
            with self.subTest(key=key), self.assertRaises(EvidenceError):
                observation_from_form({**data, key: value})


if __name__ == "__main__":
    unittest.main()
