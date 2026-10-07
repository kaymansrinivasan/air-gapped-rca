"""Offline RB3 Qwen server adapter for the shared RCA pipeline.

The server binds to loopback. Its per-request metrics are checked before any
model text can enter the structured answer validator or the chat UI.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .core import EvidenceError, require

CONTEXT = 2048
MARGIN = 64
MODEL = "qwen3.5-0.8b-q6a-2k"


def render_text_prompt(prompt: str) -> str:
    # Match runtime/2k/request_frontend.py:render_messages for one user turn.
    # User text is escaped so it cannot impersonate chat control tokens.
    safe = prompt.replace("<|", "<\u200b|")
    return f"<|im_start|>user\n{safe}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def select_rb3_prompt(job: dict, tokenizer, output_tokens: int):
    variants = job.get("prompt_options", [{"prompt": job["prompt"], "history_turns": None}])
    require(isinstance(variants, list) and 1 <= len(variants) <= 3, "Invalid prompt options.")
    last_count = None
    for variant in variants:
        prompt = variant["prompt"]
        require(isinstance(prompt, str), "Prompt must be text.")
        formatted = render_text_prompt(prompt)
        count = len(tokenizer.encode(formatted, add_special_tokens=False).ids)
        last_count = count
        if count + output_tokens + MARGIN <= CONTEXT:
            return prompt, count, variant.get("history_turns")
    raise EvidenceError(
        f"Conversation exceeds the RB3 budget: {last_count} input + "
        f"{output_tokens} output + {MARGIN} margin > {CONTEXT}. "
        "Start a new investigation with a shorter log or question. No evidence was truncated."
    )


def validate_rb3_response(raw: dict, expected_input: int, output_tokens: int) -> dict:
    require(isinstance(raw, dict) and isinstance(raw.get("choices"), list)
            and len(raw["choices"]) == 1, "Unexpected RB3 response structure.")
    choice = raw["choices"][0]
    require(choice.get("finish_reason") == "stop", "RB3 output was incomplete; no answer accepted.")
    text = choice.get("message", {}).get("content")
    require(isinstance(text, str) and text.strip(), "RB3 returned no text.")
    usage = raw.get("usage") or {}
    metrics = raw.get("qwen35_metrics") or {}
    context = metrics.get("context_management") or {}
    require(context.get("original_messages") == 1 and context.get("effective_messages") == 1
            and context.get("dropped_messages") == 0 and context.get("history_truncated") is False
            and context.get("system_dropped") is False,
            "RB3 modified the supplied messages; no answer accepted.")
    require(context.get("original_prompt_tokens") == expected_input
            and type(context.get("effective_prompt_tokens")) is int
            and context["effective_prompt_tokens"] >= expected_input
            and usage.get("prompt_tokens") == context["effective_prompt_tokens"],
            "RB3 tokenization differs from the counted prompt; no answer accepted.")
    require(metrics.get("requested_max_tokens") == output_tokens
            and metrics.get("effective_max_tokens") == output_tokens
            and metrics.get("max_tokens_clamped_to_context") is False
            and context["effective_prompt_tokens"] + output_tokens + MARGIN <= CONTEXT,
            "RB3 changed the output budget; no answer accepted.")
    backend = metrics.get("backend") or {}
    require(backend.get("fallback_disabled") is True
            and backend.get("language_prefill_full24_npu") is True
            and str(backend.get("prefill_layers_0_19", "")).startswith("npu_qnn_htp")
            and str(backend.get("tail_layers_20_23", "")).startswith("npu_qnn_htp")
            and str(backend.get("decode", "")).startswith("npu_qnn_htp"),
            "RB3 response did not report full QNN HTP language inference.")
    return {"text": text.strip(), "input_tokens": expected_input,
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice["finish_reason"]}


def _endpoint() -> str:
    url = os.environ.get("RCA_RB3_URL", "http://127.0.0.1:8000/v1/chat/completions")
    parsed = urlsplit(url)
    require(parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")
            and parsed.path == "/v1/chat/completions" and not parsed.username
            and not parsed.password and not parsed.query and not parsed.fragment,
            "RB3 inference endpoint must be the local Qwen chat route.")
    return url


def rb3_payload(prompt: str, output_tokens: int) -> dict:
    # Override the server's nonzero text presence penalty. In the on-board
    # pilot, that default produced an added reasoning field and exhausted the
    # output allowance; greedy zero-penalty decoding returned the requested
    # three-key JSON. Keep the 160-token selection budget matched to Jetson.
    return {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "repetition_penalty": 1.0, "seed": 0,
            "max_tokens": output_tokens, "stream": False}


def generate_rb3(job: dict, run_dir: Path) -> dict:
    from tokenizers import Tokenizer
    from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
    from llama_index.core.llms.callbacks import llm_completion_callback

    runtime = Path(os.environ.get("RCA_RB3_RUNTIME", Path.home() /
        "radxa-dragon-q6a-qwen3.5-0.8b-qcs6490-qnn-npu/install-2k"))
    config = json.loads((runtime / "runtime.json").read_text())
    require(config.get("model_id") == MODEL and config.get("host") == "127.0.0.1"
            and config.get("port") == 8000 and config.get("runtime", {}).get("full_npu_prefill") is True,
            "RB3 runtime config does not match the tested 2K NPU profile.")
    tokenizer = Tokenizer.from_file(config["paths"]["tokenizer"])
    tokenizer.no_truncation()
    tokenizer.no_padding()
    output_tokens = job.get("output_tokens", 160)
    require(type(output_tokens) is int and 1 <= output_tokens <= 384, "Invalid output token budget.")
    prompt, count, history_turns = select_rb3_prompt(job, tokenizer, output_tokens)
    (run_dir / "budget.json").write_text(json.dumps({
        "input_tokens": count, "output_tokens": output_tokens, "margin": MARGIN,
        "context_limit": CONTEXT, "history_turns_used": history_turns,
    }, indent=2))
    endpoint = _endpoint()

    class RB3Qwen(CustomLLM):
        @property
        def metadata(self):
            return LLMMetadata(model_name=MODEL, context_window=CONTEXT, num_output=output_tokens)

        @llm_completion_callback()
        def complete(self, prompt: str, formatted: bool = False, **kwargs):
            payload = rb3_payload(prompt, output_tokens)
            (run_dir / "request.json").write_text(json.dumps(payload, indent=2))
            headers = {"Content-Type": "application/json"}
            key = os.environ.get("QWEN35_API_KEY")
            if key:
                headers["Authorization"] = "Bearer " + key
            request = Request(endpoint, data=json.dumps(payload).encode(), headers=headers)
            try:
                with urlopen(request, timeout=600) as response:
                    response_bytes = response.read()
            except (HTTPError, URLError, TimeoutError) as exc:
                (run_dir / "runtime.log").write_text(f"RB3 server request failed: {type(exc).__name__}: {exc}\n")
                raise EvidenceError("RB3 server request failed; inspect runtime.log.") from exc
            (run_dir / "response.json").write_bytes(response_bytes)
            raw = json.loads(response_bytes)
            checked = validate_rb3_response(raw, count, output_tokens)
            return CompletionResponse(text=checked["text"], raw=raw)

        def stream_complete(self, prompt, **kwargs):
            raise NotImplementedError("RCA uses complete responses for audit.")

    result = RB3Qwen().complete(prompt)
    return {**validate_rb3_response(result.raw, count, output_tokens),
            "history_turns_used": history_turns}
