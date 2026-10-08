"""Behavioral gates for source-bound suggestions; no model is needed."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from src.rca_local.core import EvidenceError, build_prompt, load_history, make_catalog, observation_from_form
from src.verify import compatible_matches, verify_selection

ROOT = Path(__file__).resolve().parents[1]


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / "syn_data", self.root / "syn_data")

    def selection(self, test, text, pins, case):
        current = observation_from_form({
            "product": "Product_A", "failed_test": test,
            "dut_id": "NEW-DUT", "observation": text,
            "question": "Which cause and check?", "pins": pins,
        })
        history = load_history(self.root, current)
        match = next(x for x in history.values() if x["records"]["observation"]["case_id"] == case)
        bundle = {"current": current, "matches": [match]}
        catalog = make_catalog(bundle)
        chosen = {
            "cause_ids": [x for x in catalog.values() if x["kind"] == "cause"][:1],
            "check_ids": [x for x in catalog.values() if x["kind"] == "check"][:1],
        }
        return bundle, chosen

    def test_exact_continuity_pin_and_source_pass(self):
        bundle, selected = self.selection(100, "100 Open/Short-: FAIL on TSTIN", "TSTIN", "product_a_case_01")
        status, kept, rejected = verify_selection(self.root, bundle, "unconfirmed", selected)
        self.assertEqual(status, "unconfirmed")
        self.assertEqual(len(kept["cause_ids"]), 1)
        self.assertEqual(rejected, [])

    def test_pin_specific_cause_with_unknown_current_pins_refuses(self):
        bundle, selected = self.selection(100, "100 Open/Short-: FAIL", "", "product_a_case_01")
        status, kept, rejected = verify_selection(self.root, bundle, "unconfirmed", selected)
        self.assertEqual(status, "refuse")
        self.assertEqual(kept["cause_ids"], [])
        self.assertEqual(rejected, [])  # The catalog excludes the pin cause earlier.

        # The verifier must also withstand a stale catalog or changed incident.
        bundle, selected = self.selection(100, "100 Open/Short-: FAIL on TSTIN", "TSTIN", "product_a_case_01")
        bundle["current"]["test_result"]["failing_pins"] = []
        status, _, rejected = verify_selection(self.root, bundle, "unconfirmed", selected)
        self.assertEqual(status, "refuse")
        self.assertIn("pins", rejected[0]["reason"])

    def test_idd_high_and_low_are_not_interchangeable(self):
        high_text = ("210 IDD_Static: FAIL\nMeasured static current: 68.20 mA\n"
                     "Allowed range: 35.00–55.00 mA")
        low_text = ("210 IDD_Static: FAIL\nMeasured static current: 30.83 mA\n"
                    "Allowed range: 35.00–55.00 mA")
        for text, case, expected in (
            (high_text, "product_a_case_03", "unconfirmed"),
            (high_text, "product_a_case_04", "refuse"),
            (low_text, "product_a_case_03", "refuse"),
            (low_text, "product_a_case_04", "unconfirmed"),
            ("210 IDD_Static curr 35.00 ma < 65.58 ma (F) < 55.00 ma", "product_a_case_03", "unconfirmed"),
            ("210 IDD_Static curr 35.00 ma < 30.83 ma (F) < 55.00 ma", "product_a_case_03", "refuse"),
        ):
            with self.subTest(case=case, text=text):
                bundle, selected = self.selection(210, text, "", case)
                status, _, _ = verify_selection(self.root, bundle, "unconfirmed", selected)
                self.assertEqual(status, expected)

    def test_question_cannot_supply_missing_current_measurement(self):
        bundle, selected = self.selection(210, "210 IDD_Static: FAIL", "", "product_a_case_03")
        bundle["current"]["question"] = "The current is above the upper limit: what caused it?"
        self.assertEqual(verify_selection(self.root, bundle, "unconfirmed", selected)[0], "refuse")

    def test_idd_catalog_excludes_opposite_direction_before_generation(self):
        high = ("210 IDD_Static curr 35.00 ma < 68.33 ma (F) < 55.00 ma")
        bundle, _ = self.selection(210, high, "", "product_a_case_03")
        history = load_history(self.root, bundle["current"])
        bundle["matches"] = [m for m in history.values()
                             if m["records"]["observation"]["case_id"]
                             in ("product_a_case_03", "product_a_case_04")]
        eligible = compatible_matches(bundle)
        self.assertEqual([m["records"]["observation"]["case_id"] for m in eligible],
                         ["product_a_case_03"])
        catalog = make_catalog({**bundle, "matches": eligible})
        self.assertTrue(catalog)
        self.assertTrue(all(x["case_id"] == "product_a_case_03" for x in catalog.values()))
        prompt, aliases = build_prompt({**bundle, "matches": eligible}, catalog)
        self.assertNotIn("product_a_case_04", prompt)
        self.assertTrue(all(catalog[key]["case_id"] == "product_a_case_03"
                            for key in aliases.values()))
        bundle["current"]["observation_text"] = "210 IDD_Static: FAIL"
        bundle["current"]["retrieval_text"] = "210 IDD_Static: FAIL"
        self.assertEqual(compatible_matches(bundle), [])

    def test_changed_or_forged_citation_is_an_error(self):
        bundle, selected = self.selection(100, "100 Open/Short-: FAIL", "TSTIN", "product_a_case_01")
        forged = copy.deepcopy(selected)
        forged["cause_ids"][0]["citation"]["lines"]["end"] = 999999
        with self.assertRaises(EvidenceError):
            verify_selection(self.root, bundle, "unconfirmed", forged)
        source = self.root / selected["cause_ids"][0]["citation"]["file"]
        source.write_text(source.read_text() + " ")
        with self.assertRaisesRegex(EvidenceError, "changed"):
            verify_selection(self.root, bundle, "unconfirmed", selected)

    def test_refusal_is_empty_even_when_model_selected_ids(self):
        bundle, selected = self.selection(100, "100 Open/Short-: FAIL", "TSTIN", "product_a_case_01")
        status, kept, _ = verify_selection(self.root, bundle, "refuse", selected)
        self.assertEqual(status, "refuse")
        self.assertEqual(kept, {"cause_ids": [], "check_ids": []})


if __name__ == "__main__":
    unittest.main()
