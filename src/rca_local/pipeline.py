"""Run the local RAG stages and retain a complete audit per request."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from .core import EvidenceError, LIMITATIONS, build_prompt, make_catalog, render_answer, validate_selection
from .workers import settings
from src.verify import verify_selection


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_worker(mode, source, target):
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "ANONYMIZED_TELEMETRY": "False"})
    package_root = Path(__file__).resolve().parents[2]
    with target.with_suffix(".log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "src.rca_local.workers", mode, str(source), str(target)],
                                   cwd=str(package_root), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=660)
        except subprocess.TimeoutExpired:
            # Kill the worker and its TensorRT child, not unrelated Jetson processes.
            import signal
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise EvidenceError("Local inference timed out; this job was stopped.") from None
    if code != 0:
        error = target.with_suffix(".error.json")
        message = json.loads(error.read_text())["error"] if error.exists() else "Worker failed; inspect its saved log."
        raise EvidenceError(message)
    return json.loads(target.read_text())


def run_pipeline(current):
    root, _, _, _ = settings()
    run_id = uuid4().hex
    folder = root / "artifacts/rca_app" / run_id
    folder.mkdir(parents=True)
    dump(folder / "incident.json", current)
    bundle = None
    generation_attempts = 0
    try:
        bundle = run_worker("retrieve", folder / "incident.json", folder / "evidence.json")
        from src.verify import compatible_matches
        eligible = compatible_matches(bundle)
        dump(folder / "eligibility.json", {
            "retrieved_cases": [m["records"]["observation"]["case_id"] for m in bundle["matches"]],
            "eligible_cases": [m["records"]["observation"]["case_id"] for m in eligible],
            "rule": "IDD direction must be observed and match the historical case (test 210 only).",
        })
        if not eligible:
            answer = {"status": "refuse", "summary": "No symptom-compatible historical case supports an answer.", "causes": [], "checks": [], "limitations": LIMITATIONS}
        else:
            selection_bundle = {**bundle, "matches": eligible}
            catalog = make_catalog(selection_bundle)
            prompt, aliases = build_prompt(selection_bundle, catalog)
            dump(folder / "catalog.json", {"catalog": catalog, "aliases": aliases})
            correction = ""
            for attempt in range(2):
                # Keep the first attempt in its original location; never overwrite it.
                attempt_dir = folder if attempt == 0 else folder / "retry_1"
                attempt_dir.mkdir(exist_ok=True)
                dump(attempt_dir / "prompt.json", {"prompt": prompt + correction})
                generation_attempts += 1
                # Each worker rechecks the full token budget, without truncation.
                generated = run_worker("generate", attempt_dir / "prompt.json", attempt_dir / "model.json")
                # Integrity and runtime failures do not trigger a selection retry.
                for match in bundle["matches"]:
                    for source in match["sources"].values():
                        actual = hashlib.sha256((root / source["file"]).read_bytes()).hexdigest()
                        if actual != source["sha256"]:
                            raise EvidenceError("A historical source changed during this request.")
                try:
                    status, selected = validate_selection(generated["text"], aliases, catalog)
                except EvidenceError as exc:
                    dump(attempt_dir / "validation.json", {"accepted": False, "error": str(exc)})
                    if attempt == 1:
                        raise EvidenceError(f"Model selection failed after two attempts: {exc}") from exc
                    # Do not copy untrusted model text or error text into instructions.
                    correction = (
                        "\nYour previous response failed selection validation. Select again from "
                        "the same evidence. Use exactly the three required keys. Use only listed "
                        "C IDs for cause_ids and listed K IDs for check_ids, no duplicates, "
                        "one or two per array for unconfirmed. If unsupported, use refuse with "
                        "both arrays empty. Return JSON only."
                    )
                else:
                    dump(attempt_dir / "validation.json", {"accepted": True})
                    break
            status, selected, rejected = verify_selection(root, bundle, status, selected)
            dump(folder / "verification.json", {
                "status": status,
                "retained_ids": {field: [item["id"] for item in selected[field]]
                                 for field in ("cause_ids", "check_ids")},
                "rejected": rejected,
                "scope": "Source integrity, failed test, pins and IDD direction only. "
                         "Diagnostic correctness requires engineer review.",
            })
            answer = render_answer(bundle, status, selected)
    except Exception as exc:
        answer = {"status": "rejected", "summary": "No validated model answer is available.", "error": str(exc),
                  "causes": [], "checks": [], "limitations": LIMITATIONS}
    answer.update({"run_id": run_id, "generation_attempts": generation_attempts,
                   "dut_id": current["dut_id"], "question": current["question"],
                   "observation": current["retrieval_text"], "dut_id_provided": current.get("dut_id_provided", True),
                   "input_notice": current.get("input_notice", ""), "current_is_synthetic": current.get("synthetic", False)})
    # Evidence stays readable on rejection, explicitly separate from an answer.
    answer["historical_evidence"] = bundle["matches"] if bundle else []
    dump(folder / "answer.json", answer)
    return answer
