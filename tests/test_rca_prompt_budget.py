from types import SimpleNamespace
import unittest

from src.rca_local.core import EvidenceError
from src.rca_local.workers import select_prompt


class FakeTokenizer:
    """One character per token makes the boundary exact for these unit tests."""
    def encode(self, text, add_special_tokens):
        return SimpleNamespace(ids=list(text))


class PromptBudgetTests(unittest.TestCase):
    def test_full_history_selected_when_it_fits(self):
        variants = [{"prompt": "case + last two turns", "history_turns": 2},
                    {"prompt": "case + latest turn", "history_turns": 1}]
        result = select_prompt({"prompt": variants[0]["prompt"], "prompt_options": variants},
                               "<Q>", "Q", FakeTokenizer(),
                               {"max_input_len": 100, "max_kv_cache_capacity": 100}, 10, 10)
        self.assertEqual(result[0], variants[0]["prompt"])
        self.assertEqual(result[3], 2)

    def test_only_older_turn_can_be_omitted(self):
        variants = [{"prompt": "x" * 90, "history_turns": 2},
                    {"prompt": "case + latest actual answer", "history_turns": 1}]
        result = select_prompt({"prompt": variants[0]["prompt"], "prompt_options": variants},
                               "<Q>", "Q", FakeTokenizer(),
                               {"max_input_len": 100, "max_kv_cache_capacity": 100}, 10, 10)
        self.assertEqual(result[0], variants[1]["prompt"])
        self.assertEqual(result[3], 1)

    def test_no_clipping_if_even_required_context_is_too_large(self):
        job = {"prompt": "x" * 90}
        with self.assertRaisesRegex(EvidenceError, "No evidence was truncated"):
            select_prompt(job, "<Q>", "Q", FakeTokenizer(),
                          {"max_input_len": 100, "max_kv_cache_capacity": 100}, 10, 10)

    def test_existing_investigation_prompt_is_unchanged(self):
        result = select_prompt({"prompt": "original"}, "<Q>", "Q", FakeTokenizer(),
                               {"max_input_len": 100, "max_kv_cache_capacity": 100}, 10, 10)
        self.assertEqual(result, ("original", "<original>", 10, None))


if __name__ == "__main__":
    unittest.main()
