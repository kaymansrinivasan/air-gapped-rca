"""Run the frozen evaluation and three explicitly isolated benchmark ablations.

No experiment option is exposed by the web app. Results and raw model outputs
live outside artifacts/rca_app. A checkpoint is saved after every question.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

from benchmark import load_gold, load_manifest, summarize
from src.measure import JetsonPowerSampler
from src.rca_local.core import EvidenceError
from src.verify import _source_record, verify_selection

VARIANTS = ("full", "no_keyword", "no_verifier", "no_refusal")
PROTOCOL = "ate-proxy-ablation-suite-v1"
REPO = Path(__file__).resolve().parents[1]
DEFINITIONS = {
    "full": "Hybrid retrieval, applicability prefilter, source verification and refusal.",
    "no_keyword": "Same pipeline; vector ranking only, with no keyword contribution.",
    "no_verifier": "Remove IDD prefilter, pin catalog filter and post-selection applicability verification. Schema, ID ownership and source hash checks remain.",
    "no_refusal": "Force a model selection from retrieved history, bypass the IDD abstention prefilter, and return raw selections if verification would remove all support. Verification still records rejected claims. Invalid JSON is still rejected.",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def command(args):
    try:
        result = subprocess.run(args, cwd=REPO, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=20)
        return {"exit_code": result.returncode, "output": result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def environment(backend):
    info = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "python": platform.python_version(),
            "git_commit": command(["git", "rev-parse", "HEAD"]),
            "git_status": command(["git", "status", "--short"]),
            "packages": {}}
    for name in ("chromadb", "llama-index-core", "llama-index-vector-stores-chroma", "tokenizers", "onnxruntime"):
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][name] = None
    for name in ("/proc/device-tree/model", "/etc/os-release", "/proc/meminfo"):
        path = Path(name)
        if path.exists():
            info[name] = path.read_text(errors="replace").replace("\u0000", "").strip()
    if backend == "jetson":
        from src.rca_local.workers import settings
        _, build, engine, _ = settings()
        info["power_mode"] = command(["nvpmodel", "-q"])
        info["clock_settings"] = command(["jetson_clocks", "--show"])
        info["runtime_commit"] = command(["git", "-C", str(build.parent), "rev-parse", "HEAD"])
        paths = [engine / "config.json", engine / "tokenizer.json"]
    else:
        runtime = Path.home() / "radxa-dragon-q6a-qwen3.5-0.8b-qcs6490-qnn-npu"
        info["runtime_commit"] = command(["git", "-C", str(runtime), "rev-parse", "HEAD"])
        paths = [runtime / "install-2k/runtime.json", runtime / "downloads/common/tokenizer.json"]
    info["runtime_file_sha256"] = {str(p): sha(p) for p in paths if p.is_file()}
    info["comparison_limitations"] = [
        "Different runtime precision and engine lifecycle; this is a system comparison, not an isolated GPU versus NPU speed test.",
        "A metadata command failure means that field must be documented separately; no default hardware value is assumed."]
    return info


def decode_measurements(folder, backend):
    tokens, seconds, samples = 0, 0.0, 0
    for attempt in (folder, folder / "retry_1"):
        if backend == "jetson" and (attempt / "profile.json").exists():
            data = read(attempt / "profile.json").get("generation", {})
            n, rate = data.get("generated_tokens"), data.get("tokens_per_second")
        elif backend == "rb3" and (attempt / "response.json").exists():
            data = read(attempt / "response.json")
            n = data.get("usage", {}).get("completion_tokens")
            rate = data.get("qwen35_metrics", {}).get("decode_tokens_per_second")
        else:
            continue
        if type(n) is int and n > 0 and type(rate) in (int, float) and math.isfinite(rate) and rate > 0:
            tokens += n
            seconds += n / rate
            samples += 1
    return {"decode_tokens": tokens, "decode_seconds": seconds, "decode_samples": samples}


def audit_answer(root, folder, answer):
    """Observe ablation errors without suppressing or repairing their outputs."""
    items = answer["causes"] + answer["checks"]
    if not items:
        return {"citation_count": 0, "valid_citation_count": 0, "unsupported_ids": []}
    bundle = read(folder / "evidence.json")
    matches = {m["records"]["observation"]["case_id"]: m for m in bundle["matches"]}
    valid, errors = 0, []
    for item in items:
        try:
            _source_record(root, item, matches[item["case_id"]])
            valid += 1
        except (EvidenceError, KeyError, OSError) as exc:
            errors.append({"id": item["id"], "error": str(exc)})
    try:
        _, _, rejected = verify_selection(root, bundle, answer["status"],
                                         {"cause_ids": answer["causes"], "check_ids": answer["checks"]})
    except (EvidenceError, KeyError, OSError) as exc:
        rejected = [{"id": x["id"], "reason": str(exc)} for x in items]
    return {"citation_count": len(items), "valid_citation_count": valid,
            "citation_errors": errors, "unsupported_ids": [x["id"] for x in rejected]}


def score(items, rows, labels, origin, tariff=None):
    if [x["id"] for x in rows] != [x["id"] for x in items]:
        raise ValueError("Completed results must match all frozen questions in order.")
    data = summarize(items, rows, labels, origin)
    proxy = origin == "ai_proxy_from_six_synthetic_scenarios"
    prefix = "proxy_" if proxy else ""
    acceptance = data.pop("proxy_exact_answer_agreement" if proxy else "exact_answer_accuracy")
    data[prefix + "answer_acceptance_rate"] = acceptance
    data["metric_definition"] = "Expected status plus nonempty subsets of acceptable causes/checks, not exact set equality."
    answered = [r for r in rows if r["status"] == "unconfirmed"]
    hits3 = sum(bool(set(r["cause_ids"][:3]) & set(labels[r["id"]]["acceptable_cause_ids"])) for r in answered)
    data[prefix + "top3_cause_hit_rate_on_answered"] = hits3 / len(answered) if answered else None
    data["refusal_rate"] = sum(r["status"] == "refuse" for r in rows) / len(rows)
    data["rejection_rate"] = sum(r["status"] == "rejected" for r in rows) / len(rows)
    data["answer_coverage"] = len(answered) / len(rows)
    data["no_answer_refusal_rate"] = data["no_answer_refusal_count"] / data["no_answer_total"]
    citations = sum(r["citation_count"] for r in rows)
    data["citation_validity"] = sum(r["valid_citation_count"] for r in rows) / citations if citations else None
    data["citation_count"] = citations
    secs = sum(r["decode_seconds"] for r in rows)
    data["decode_tokens_per_second"] = sum(r["decode_tokens"] for r in rows) / secs if secs else None
    data["decode_sample_count"] = sum(r["decode_samples"] for r in rows)
    powered = [r for r in rows if r["mean_power_watts"] is not None]
    data["power_case_coverage"] = len(powered)
    complete_power = len(powered) == len(rows)
    data["average_power_watts"] = statistics.mean(r["mean_power_watts"] for r in powered) if complete_power else None
    data["peak_power_watts"] = max(r["peak_power_watts"] for r in powered) if complete_power else None
    data["mean_energy_joules_per_triage"] = statistics.mean(r["energy_joules"] for r in powered) if complete_power else None
    energy = data["mean_energy_joules_per_triage"]
    data["energy_kwh_per_1000_triages"] = energy / 3600 if energy is not None else None
    data["electricity_cost_per_1000_triages"] = energy / 3600 * tariff if energy is not None and tariff is not None else None
    data["false_answer_scope"] = "Proxy rubric violations among answered questions; not independently verified physical diagnoses."
    return data


def value(v):
    return "unmeasured" if v is None else f"{v:.4g}" if isinstance(v, float) else str(v)


def report(output, data):
    rows = data["variants"]
    fields = ("answer_acceptance_rate", "top1_cause_hit_rate_on_answered", "top3_cause_hit_rate_on_answered",
              "false_answer_rate_per_answer", "refusal_rate", "rejection_rate", "citation_validity",
              "median_latency_seconds", "p95_latency_seconds", "decode_tokens_per_second",
              "average_power_watts", "peak_power_watts", "mean_energy_joules_per_triage", "electricity_cost_per_1000_triages")
    def metric(summary, key):
        return summary.get("proxy_" + key, summary.get(key))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", *fields])
        writer.writerows([v, *[metric(rows[v]["summary"], k) for k in fields]] for v in rows)
    lines = ["# Frozen ATE proxy benchmark", "", f"Backend: {data['backend']}. Protocol: {PROTOCOL}.", "",
             "All labels are provisional when label_origin is AI proxy. Source citation checks do not establish physical causes.", "",
             "| Metric | " + " | ".join(rows) + " |", "| --- | " + " | ".join("---:" for _ in rows) + " |"]
    for k in fields:
        lines.append("| " + k + " | " + " | ".join(value(metric(rows[v]["summary"], k)) for v in rows) + " |")
    lines += ["", "## Experimental conditions", "", *[f"- {v}: {DEFINITIONS[v]}" for v in rows], "",
              "## Remaining evidence", "", "- Independent engineer labels are still required for diagnostic accuracy.",
              "- RB3 power requires contemporaneous board input measurements; absent samples stay unmeasured.",
              "- Physical network disconnect, reboot and a new request require a witnessed device test.",
              "- Free-form chat is experimental and outside this structured-answer evaluation.",
              "- The current log library has only six curated histories. Similar question templates limit generalisation.",
              "- Profiling and engine lifecycle differ between runtimes; report metadata and sample counts with rates."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Dependency-free SVG: measured values only; missing metrics have no bar.
    for field in ("false_answer_rate_per_answer", "median_latency_seconds", "mean_energy_joules_per_triage"):
        vals = [(v, metric(rows[v]["summary"], field)) for v in rows]
        scale = max([x for _, x in vals if x is not None] or [1]) or 1
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="880" height="300" viewBox="0 0 880 300">',
               '<rect width="880" height="300" fill="white"/>',
               f'<text x="20" y="30" font-family="sans-serif" font-size="18">{html.escape(field)} — {html.escape(data["backend"])}</text>']
        for i, (variant, n) in enumerate(vals):
            y = 62 + i * 48
            svg.append(f'<text x="20" y="{y+20}" font-family="sans-serif" font-size="14">{variant}</text>')
            if n is not None:
                svg.append(f'<rect x="150" y="{y}" width="{560*n/scale:.2f}" height="28" fill="#19776b"/>')
            svg.append(f'<text x="730" y="{y+20}" font-family="sans-serif" font-size="14">{value(n)}</text>')
        svg.append('<text x="20" y="282" font-family="sans-serif" font-size="12">Provisional proxy evaluation; unmeasured values are not zero.</text></svg>')
        (output / (field + ".svg")).write_text("\n".join(svg), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jetson", "rb3"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--jetson-power", action="store_true")
    parser.add_argument("--electricity-per-kwh", type=float)
    parser.add_argument("--currency")
    args = parser.parse_args()
    if args.jetson_power and args.backend != "jetson":
        parser.error("--jetson-power requires Jetson.")
    if args.electricity_per_kwh is not None and (not math.isfinite(args.electricity_per_kwh) or args.electricity_per_kwh <= 0 or not args.currency):
        parser.error("A positive electricity price and currency are required together.")
    root, out = args.data_root.resolve(), args.output_dir.resolve()
    manifest_path = REPO / "eval/product_a_v1_50.json"
    gold_path = REPO / "eval/product_a_v1_50_proxy_gold.json"
    manifest, digest = load_manifest(manifest_path)
    labels, origin = load_gold(gold_path, manifest, digest, allow_proxy=True)
    os.environ.update(RCA_DATA_ROOT=str(root), RCA_BACKEND=args.backend,
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", ANONYMIZED_TELEMETRY="False")
    config = {"protocol": PROTOCOL, "backend": args.backend, "manifest_sha256": digest,
              "gold_sha256": sha(gold_path), "label_origin": origin, "max_causes": 3,
              "jetson_power": args.jetson_power, "data_root": str(root),
              "electricity_per_kwh": args.electricity_per_kwh, "currency": args.currency,
              "source_sha256": {str(p.relative_to(root)): sha(p) for p in sorted((root / "syn_data/product_a_scenarios_v1").glob("case_*/*.json"))},
              "code_sha256": {str(p.relative_to(REPO)): sha(p) for p in sorted((REPO / "src").rglob("*.py"))}}
    config["code_sha256"]["benchmark.py"] = sha(REPO / "benchmark.py")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This suite is already running.") from None
        checkpoint = out / "progress.json"
        if checkpoint.exists():
            if not args.resume:
                raise SystemExit("Suite exists. Use --resume to continue its saved checkpoint.")
            state = read(checkpoint)
            if state["configuration"] != config:
                raise SystemExit("Code, sources or settings changed. Choose a new output directory.")
        else:
            state = {"configuration": config, "environment": environment(args.backend), "variants": {}}
            save(checkpoint, state)
        from src.rca_local.pipeline import run_pipeline
        for variant in VARIANTS:
            rows = state["variants"].setdefault(variant, [])
            expected = [x["id"] for x in manifest["items"][:len(rows)]]
            if [x["id"] for x in rows] != expected or len(rows) > len(manifest["items"]):
                raise SystemExit("Checkpoint question order is invalid.")
            os.environ["RCA_RETRIEVAL_MODE"] = "vector_only" if variant == "no_keyword" else "hybrid"
            for item in manifest["items"][len(rows):]:
                with JetsonPowerSampler() if args.jetson_power else nullcontext() as sampler:
                    start = time.perf_counter()
                    answer = run_pipeline(item["current"], experiment={"variant": variant, "max_causes": 3})
                    elapsed = time.perf_counter() - start
                readings = list(sampler.values) if sampler else []
                mean = statistics.mean(readings) if readings else None
                folder = root / "artifacts/rca_experiments" / answer["run_id"]
                row = {"id": item["id"], "status": answer["status"], "run_id": answer["run_id"],
                       "latency_seconds": elapsed, "generation_attempts": answer["generation_attempts"],
                       "cause_ids": [x["id"] for x in answer["causes"]], "check_ids": [x["id"] for x in answer["checks"]],
                       "error": answer.get("error"), "power_samples_watts": readings,
                       "mean_power_watts": mean, "peak_power_watts": max(readings) if readings else None,
                       "energy_joules": mean * elapsed if mean is not None else None,
                       **decode_measurements(folder, args.backend), **audit_answer(root, folder, answer)}
                rows.append(row)
                save(checkpoint, state)
                print(f"{variant} {len(rows):02d}/50 {item['id']} {row['status']} {elapsed:.2f}s", flush=True)
                if len(rows) == 1 and row["status"] == "rejected" and not (folder / "model.json").exists():
                    raise SystemExit("First request failed before a usable generation: " + str(row["error"]) + ". Check its run artifacts before resuming.")
            result = {**config, "variant": variant, "definition": DEFINITIONS[variant], "results": rows,
                      "summary": score(manifest["items"], rows, labels, origin, args.electricity_per_kwh)}
            save(out / (variant + ".json"), result)
        data = {**config, "environment": state["environment"],
                "variants": {v: read(out / (v + ".json")) for v in VARIANTS}}
        save(out / "suite.json", data)
        report(out, data)
        print("COMPLETE:", out / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
