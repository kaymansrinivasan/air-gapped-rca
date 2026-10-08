import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.rca_local.chat import chat_input, chat_prompt, fact_catalog, run_chat, validate_chat, validate_case_language
from src.rca_local.core import EvidenceError, load_history, observation_from_form

ROOT = Path(__file__).resolve().parents[1]


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / "syn_data", self.root / "syn_data")
        self.current = observation_from_form({"product": "Product_A", "failed_test": 100,
                                             "dut_id": "NEW-DUT", "pins": "TSTIN",
                                             "question": "Which checks distinguish the causes?"})
        history = load_history(self.root, self.current)
        self.bundle = {"current": self.current, "matches": [history["Product_A-L01-W03-D012"]]}
        self.facts = fact_catalog(self.bundle)
        self.run_id = "a" * 32
        self.folder = self.root / "artifacts/rca_app" / self.run_id
        self.folder.mkdir(parents=True)
        (self.folder / "evidence.json").write_text(json.dumps(self.bundle))
        (self.folder / "answer.json").write_text(json.dumps({"causes": [], "checks": []}))
        self.calls = []

    def run_question(self, outputs, question="What happened to the reference die?"):
        outputs = iter(outputs)

        def worker(mode, source, target):
            self.assertEqual(mode, "generate")
            self.calls.append(json.loads(source.read_text())["prompt"])
            value = {"text": next(outputs)}
            target.write_text(json.dumps(value))
            return value

        with patch("src.rca_local.chat.settings", return_value=(self.root, None, None, None)), \
                patch("src.rca_local.chat.run_worker", side_effect=worker):
            return run_chat(self.run_id, question)

    def test_catalog_wording_is_owned_by_historical_sources(self):
        for item in self.facts.values():
            record = json.loads((self.root / item["citation"]["file"]).read_text())
            self.assertEqual(item["historical_dut"], record["dut_id"])
            if item["kind"] == "historical check and simulated outcome":
                self.assertIn(item["text"], ["Check: " + x["check"] + "\nSimulated outcome: " + x["result"] for x in record["checks"]])
            else:
                texts = record.get("possible_causes", []) + [record.get(k) for k in ("assessment", "action_taken", "conclusion", "limitation")]
                self.assertIn(item["text"], texts)
            self.assertNotIn("NEW-DUT", item["text"])

    def test_plain_text_is_accepted_and_invalid_references_or_json_are_rejected(self):
        reply = "Compare a known-good reference under comparable conditions; the current cause is unconfirmed. [E3]"
        text, items = validate_chat(reply, self.facts)
        self.assertEqual(text, reply)
        self.assertEqual(items, [self.facts["E3"]])
        invalid = [
            "The cause is this. [E999]", "", None,
            '{"scope":"case","evidence_ids":["E1"]}',
            '```json\n{"scope":"case"}\n```',
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(EvidenceError):
                validate_chat(text, self.facts)

    def test_prompt_preserves_case_and_only_two_recent_turns(self):
        turns = [{"question": q, "scope": "conversation", "text": "Previous explanation " + q, "items": []} for q in ("OLD_TURN", "RECENT_ONE", "RECENT_TWO")]
        prompt = chat_prompt(self.bundle, {}, self.facts, turns, "Explain that check.")
        self.assertIn(self.current["retrieval_text"], prompt)
        self.assertNotIn("OLD_TURN", prompt)
        self.assertIn("RECENT_ONE", prompt)
        self.assertIn("RECENT_TWO", prompt)
        self.assertIn("Explain that check.", prompt)
        self.assertIn("Previous explanation RECENT_TWO", prompt)
        self.assertIn("ONE sentence", prompt)

    def test_followups_are_saved_and_reused_with_citations(self):
        output = "The reference also failed on the suspect setup in the historical case. [E3]"
        first = self.run_question([output], "REFERENCE_QUESTION")
        second = self.run_question([output], "Could you give me one line?")
        self.assertEqual(first["items"], [self.facts["E3"]])
        self.assertEqual(second["run_id"], self.run_id)
        self.assertIn("REFERENCE_QUESTION", self.calls[-1])
        self.assertIn(output, self.calls[-1])
        self.assertEqual(second["text"], output)
        self.assertEqual(len(list((self.folder / "chat").glob("*/answer.json"))), 2)

    def test_second_invalid_response_is_rejected_and_both_saved(self):
        invalid = "Compare the reference. [E999]"
        answer = self.run_question([invalid, invalid])
        self.assertEqual(answer["scope"], "rejected")
        self.assertEqual(answer["items"], [])
        turn = self.folder / "chat" / answer["turn_id"]
        self.assertTrue((turn / "model.json").is_file())
        self.assertTrue((turn / "retry_1/model.json").is_file())

    def test_plain_refusal_needs_no_json_or_invented_citations(self):
        reply = "I can help with ATE testing and this failure investigation. Please ask an ATE question."
        answer = self.run_question([reply], "Recommend a holiday destination.")
        self.assertEqual(answer["text"], reply)
        self.assertEqual(answer["items"], [])
        self.assertEqual(answer["citation_status"], "no_explicit_references")
        self.assertIn("Stay within ATE testing", self.calls[-1])

    def test_shortening_without_citation_is_not_a_format_failure(self):
        reply = "Compare a known-good reference with the failing DUT to help distinguish a setup issue from a DUT issue."
        answer = self.run_question([reply], "Could you give me one line?")
        self.assertEqual(answer["scope"], "conversation")
        self.assertEqual(answer["generation_attempts"], 1)
        self.assertEqual(answer["text"], reply)
        self.assertEqual(answer["items"], [])
        self.assertTrue(answer["context_sources"])

    def test_changed_source_blocks_generation(self):
        path = self.root / self.facts["E1"]["citation"]["file"]
        path.write_text(path.read_text() + "\n")
        answer = self.run_question([])
        self.assertEqual(answer["scope"], "rejected")
        self.assertEqual(answer["generation_attempts"], 0)
        self.assertEqual(self.calls, [])

    def test_invalid_run_paths_and_questions_rejected(self):
        for run_id, question in (("../secret", "check?"), (self.run_id, ""), (self.run_id, "x" * 701), (self.run_id, [])):
            with self.assertRaises(EvidenceError):
                chat_input({"run_id": run_id, "question": question})

    def test_ten_turn_limit_is_enforced(self):
        for i in range(10):
            target = self.folder / "chat" / f"{i:02d}"
            target.mkdir(parents=True)
            (target / "answer.json").write_text('{}')
        with self.assertRaisesRegex(EvidenceError, "ten turns"):
            self.run_question([])

    def test_observed_idd_wrong_dut_and_overclaim_are_rejected(self):
        history = load_history(self.root, self.current)
        bundle = {"current": self.current, "matches": [history["Product_A-L01-W01-D100"]]}
        for text in (
            "The engineer should check contact integrity, repeat the measurement, and hold D100 for further electrical analysis.",
            "This indicates a DUT-related excessive static current condition.",
        ):
            with self.subTest(text=text), self.assertRaises(EvidenceError):
                validate_case_language(text, bundle)

    def test_historical_description_and_cautious_current_advice_remain_valid(self):
        validate_case_language(
            "In the historical case, D012 passed continuity on the comparison setup. "
            "For the current DUT, compare a known-good reference. "
            "This may indicate a DUT-related issue, but the cause is unconfirmed.", self.bundle)
        validate_case_language("The measurement does not confirm a DUT-related cause.", self.bundle)

    def test_wrong_dut_advice_retries_without_rewriting_historical_results(self):
        bad = "The engineer should retest D012 on a verified setup."
        good = "Compare the current DUT with a known-good reference under comparable conditions. [E3]"
        answer = self.run_question([bad, good])
        self.assertEqual(answer["text"], good)
        self.assertEqual(answer["generation_attempts"], 2)
        turn = self.folder / "chat" / answer["turn_id"]
        self.assertEqual(json.loads((turn / "model.json").read_text())["text"], bad)

    def test_repeated_historical_dut_advice_still_rejects(self):
        bad = "Retest D012 on the original setup."
        answer = self.run_question([bad, bad])
        self.assertEqual(answer["scope"], "rejected")
        self.assertIn("historical DUT", answer["error"])

    def test_previously_accepted_wrong_dut_reply_is_not_reused_as_memory(self):
        turn = self.folder / "chat" / "01_old"
        turn.mkdir(parents=True)
        old = {"question": "OLD_BAD_QUESTION", "scope": "conversation", "text": "Retest D012 on the original setup."}
        (turn / "answer.json").write_text(json.dumps(old))
        self.run_question(["The current DUT's cause remains unconfirmed."])
        self.assertNotIn("OLD_BAD_QUESTION", self.calls[-1])
        self.assertTrue((turn / "answer.json").is_file())


if __name__ == "__main__":
    unittest.main()
