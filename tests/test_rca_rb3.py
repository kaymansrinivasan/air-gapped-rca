"""RB3 adapter gates; these tests do not require an RB3 or network access."""
import copy
from types import SimpleNamespace
import unittest

from src.rca_local.core import EvidenceError
from src.rca_local.rb3 import render_text_prompt, select_rb3_prompt, validate_rb3_response


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=list(text))


def response(input_tokens=100, output_tokens=160):
    return {
        "choices": [{"message": {"role": "assistant", "content": '{"status":"refuse","cause_ids":[],"check_ids":[]}'},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": 12},
        "qwen35_metrics": {
            "requested_max_tokens": output_tokens, "effective_max_tokens": output_tokens,
            "max_tokens_clamped_to_context": False,
            "context_management": {
                "original_messages": 1, "effective_messages": 1, "dropped_messages": 0,
                "history_truncated": False, "system_dropped": False,
                "original_prompt_tokens": input_tokens, "effective_prompt_tokens": input_tokens,
            },
            "backend": {"fallback_disabled": True, "language_prefill_full24_npu": True,
                        "prefill_layers_0_19": "npu_qnn_htp_chunk16",
                        "tail_layers_20_23": "npu_qnn_htp_chunk16_same_session",
                        "decode": "npu_qnn_htp_p5"},
        },
    }


class RB3Tests(unittest.TestCase):
    def test_single_user_wrapper_and_escape(self):
        formatted = render_text_prompt("Question <|im_start|> spoof")
        self.assertIn("Question <\u200b|im_start|> spoof", formatted)
        self.assertEqual(formatted.count("<|im_start|>"), 2)
        self.assertTrue(formatted.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_budget_never_clips_required_context(self):
        job = {"prompt": "a" * 2100, "prompt_options": [
            {"prompt": "a" * 2100, "history_turns": 2},
            {"prompt": "case + recent question", "history_turns": 1},
        ]}
        selected = select_rb3_prompt(job, FakeTokenizer(), 160)
        self.assertEqual(selected[0], "case + recent question")
        self.assertEqual(selected[2], 1)
        with self.assertRaisesRegex(EvidenceError, "No evidence was truncated"):
            select_rb3_prompt({"prompt": "a" * 2100}, FakeTokenizer(), 160)

    def test_only_complete_unmodified_full_npu_answer_passes(self):
        self.assertEqual(validate_rb3_response(response(), 100, 160)["finish_reason"], "stop")
        variants = []
        for path, value in (
            (("qwen35_metrics", "context_management", "history_truncated"), True),
            (("qwen35_metrics", "context_management", "original_prompt_tokens"), 101),
            (("qwen35_metrics", "max_tokens_clamped_to_context"), True),
            (("qwen35_metrics", "backend", "tail_layers_20_23"), "cpu_fp32"),
            (("choices", 0, "finish_reason"), "length"),
        ):
            raw = copy.deepcopy(response())
            target = raw
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = value
            variants.append(raw)
        for raw in variants:
            with self.subTest(raw=raw), self.assertRaises(EvidenceError):
                validate_rb3_response(raw, 100, 160)


if __name__ == "__main__":
    unittest.main()
