"""Generate cross-board tables and SVG figures from two completed suite files."""
import argparse
import csv
import html
import json
from pathlib import Path

from src.benchmark_suite import PROTOCOL, VARIANTS, value


def load_pair(jetson_path, rb3_path):
    boards = {"Jetson": json.loads(Path(jetson_path).read_text()),
              "RB3": json.loads(Path(rb3_path).read_text())}
    j, r = boards.values()
    for field in ("protocol", "manifest_sha256", "gold_sha256", "label_origin", "max_causes", "source_sha256", "code_sha256"):
        if j.get(field) != r.get(field):
            raise ValueError("Different comparison protocol: " + field)
    if j.get("protocol") != PROTOCOL or j["backend"] != "jetson" or r["backend"] != "rb3":
        raise ValueError("Expected one Jetson and one RB3 suite of the current protocol.")
    for data in boards.values():
        if set(data["variants"]) != set(VARIANTS) or any(v["summary"]["count"] != 50 for v in data["variants"].values()):
            raise ValueError("Every condition must have all 50 frozen questions.")
    return boards


def metric(data, key, variant="full"):
    s = data["variants"][variant]["summary"]
    return s.get("proxy_" + key, s.get(key))


def svg_frame(title, footer):
    return ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="380" viewBox="0 0 800 380">',
            '<rect width="800" height="380" fill="white"/>',
            f'<text x="24" y="32" font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
            f'<text x="24" y="360" font-family="sans-serif" font-size="12">{html.escape(footer)}</text>']


def write_report(boards, output):
    output.mkdir(parents=True, exist_ok=True)
    fields = {
        "Quality (proxy labels)": ("top1_cause_hit_rate_on_answered", "top3_cause_hit_rate_on_answered", "answer_acceptance_rate", "citation_validity", "refusal_rate", "rejection_rate", "false_answer_rate_per_answer"),
        "Performance (full condition)": ("median_latency_seconds", "p95_latency_seconds", "decode_tokens_per_second", "average_power_watts", "peak_power_watts", "mean_energy_joules_per_triage", "electricity_cost_per_1000_triages"),
    }
    text = ["# Cross-board experimental results", "", "Generated directly from completed suite.json files. Quality uses the provisional proxy key, not independent engineer labels.", ""]
    for board, data in boards.items():
        text.append(f"{board} electricity scenario: {value(data.get('electricity_per_kwh'))} {data.get('currency') or 'currency unspecified'}/kWh (illustrative assumption).")
    text.append("")
    for title, keys in fields.items():
        text += ["## " + title, "", "| Metric | Jetson | RB3 |", "| --- | ---: | ---: |"]
        text += ["| " + key + " | " + " | ".join(value(metric(data, key)) for data in boards.values()) + " |" for key in keys]
        text += [""]
    text += ["## Ablations", "", "| Board | Condition | Proxy acceptance | Proxy false answer rate | Refusal rate | Rejection rate | Median seconds |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    ablation_fields = ("answer_acceptance_rate", "false_answer_rate_per_answer", "refusal_rate", "rejection_rate", "median_latency_seconds")
    for board, data in boards.items():
        for variant in VARIANTS:
            text.append("| " + board + " | " + variant + " | " + " | ".join(value(metric(data, k, variant)) for k in ablation_fields) + " |")
    text += ["", "## Interpretation limits", "", "- Source-owned citations can still support an inapplicable hypothesis.",
             "- Three-cause benchmark protocol differs from the earlier two-cause UI runs.",
             "- Different precision and engine lifecycles prevent attributing latency differences solely to the accelerator.",
             "- Missing energy is unmeasured, not zero. Cost is electricity only at the tariff/currency saved in each suite.",
             "- Offline cold boot, independent engineering review and free-form chat validation need separate evidence."]
    (output / "comparison.md").write_text("\n".join(text) + "\n")
    with (output / "comparison.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        all_fields = [*fields["Quality (proxy labels)"], *fields["Performance (full condition)"]]
        writer.writerow(["board", "condition", *all_fields])
        for board, data in boards.items():
            for variant in VARIANTS:
                writer.writerow([board, variant, *[metric(data, k, variant) for k in all_fields]])
    for name, title, points in (
        ("energy", "Mean energy per triage (J)", [(b, metric(d, "mean_energy_joules_per_triage")) for b, d in boards.items()]),
        ("false_answers", "Proxy false answers: full versus no verifier", [(b + " " + v, metric(d, "false_answer_rate_per_answer", v)) for b, d in boards.items() for v in ("full", "no_verifier")]),
    ):
        svg = svg_frame(title, "Provisional proxy evaluation. Unmeasured values have no bar.")
        maximum = max([n for _, n in points if n is not None] or [1]) or 1
        for i, (label, n) in enumerate(points):
            y = 65 + i * 62
            svg.append(f'<text x="24" y="{y+22}" font-family="sans-serif" font-size="14">{html.escape(label)}</text>')
            if n is not None:
                svg.append(f'<rect x="210" y="{y}" width="{440*n/maximum:.2f}" height="32" fill="#19776b"/>')
            svg.append(f'<text x="665" y="{y+22}" font-family="sans-serif" font-size="14">{value(n)}</text>')
        (output / (name + ".svg")).write_text("\n".join(svg + ["</svg>"]))
    svg = svg_frame("Proxy top-1 hit rate versus median latency", "Hit rate is conditional on answered questions; report refusal and rejection rates alongside it.")
    max_time = max(metric(d, "median_latency_seconds") for d in boards.values()) * 1.2 or 1
    svg += ['<path d="M80 65V310H730" fill="none" stroke="#333"/>',
            '<text x="320" y="340" font-family="sans-serif" font-size="14">Median end-to-end latency (s)</text>',
            '<text x="24" y="78" font-family="sans-serif" font-size="13">1.0</text>',
            '<text x="24" y="310" font-family="sans-serif" font-size="13">0.0</text>']
    for i in range(5):
        t = max_time * i / 4
        svg.append(f'<text x="{80+650*i/4:.1f}" y="325" font-family="sans-serif" font-size="12">{t:.1f}</text>')
    for board, data in boards.items():
        accuracy, latency = metric(data, "top1_cause_hit_rate_on_answered"), metric(data, "median_latency_seconds")
        if accuracy is not None:
            x, y = 80 + 650 * latency / max_time, 310 - 235 * accuracy
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="#19776b"/>')
            svg.append(f'<text x="{x+10:.2f}" y="{y+18:.2f}" font-family="sans-serif" font-size="13">{board}: {accuracy:.3f}, {latency:.2f}s</text>')
    (output / "quality_latency.svg").write_text("\n".join(svg + ["</svg>"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jetson", type=Path, required=True)
    parser.add_argument("--rb3", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    boards = load_pair(args.jetson, args.rb3)
    write_report(boards, args.output_dir)
    print("Saved:", args.output_dir / "comparison.md")


if __name__ == "__main__":
    main()
