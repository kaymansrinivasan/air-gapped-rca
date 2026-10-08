"""Build an offline, single-file engineer review page for the frozen inputs."""
import json
from pathlib import Path

from benchmark import load_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "eval/product_a_v1_50.json"
TEMPLATE = ROOT / "eval/product_a_v1_50_review_template.json"
OUTPUT = ROOT / "eval/product_a_v1_50_review.html"

HTML = r"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Product_A RCA | Engineer review</title>
<style>
:root { font-family: system-ui, sans-serif; color: #183435; background: #f4f7f6 }
body { max-width: 1000px; margin: auto; padding: 24px }
header, nav, .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap }
header { justify-content: space-between }
article { background: white; border: 1px solid #d7e3e0; border-radius: 12px; padding: 24px; margin: 16px 0 }
h1 { font-size: 1.6rem } h2 { margin-bottom: 8px } h3 { margin-top: 20px }
button, input, select, textarea { font: inherit }
button { border: 1px solid #b9d5ce; border-radius: 7px; padding: 9px 13px; background: white; cursor: pointer }
button.primary { background: #176c5e; color: white }
input[type=text], select, textarea { border: 1px solid #adbfbb; border-radius: 6px; padding: 9px; width: 100%; box-sizing: border-box }
label { display: block; margin: 8px 0 }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f4f7f6; padding: 14px; max-height: 260px; overflow: auto }
.option { display: flex; gap: 8px; align-items: start; padding: 8px; border-top: 1px solid #e2ebe8 }
.option input { margin-top: 5px }
.muted { color: #566d6b } .notice { background: #fff3da; padding: 14px; border-radius: 7px }
.row > * { flex: 1 }
</style>
<header><div><h1>RCA engineer review</h1><p class="muted">50 frozen synthetic inputs · offline worksheet</p></div>
<div><label>Reviewer name <input id="reviewer" type="text" placeholder="Your name"></label></div></header>
<p class="notice">Read the current observation and select acceptable historical hypotheses and checks.
These are unconfirmed suggestions. Do not mark a physical cause as proven. Review every refusal, including
the ten missing-measurement probes. No model answers are shown here.</p>
<nav><button id="prev">Previous</button><strong id="progress"></strong><button id="next">Next</button>
<button id="draft">Save draft</button><button id="export" class="primary">Export reviewed labels</button>
<label>Resume draft <input id="import" type="file" accept=".json,application/json"></label></nav>
<p id="message" role="status"></p><article id="case"></article>
<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
const source = JSON.parse(document.getElementById("payload").textContent);
const rows = source.items;
const observations = source.observations;
let position = 0;
const el = (tag, text) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; return node; };
const message = text => { document.getElementById("message").textContent = text; };
const clone = value => JSON.parse(JSON.stringify(value));
function download(name, value) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2) + "\n"], {type:"application/json"}));
  const a = el("a"); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function render() {
  const item = rows[position], data = observations[item.id];
  const approved = rows.filter(row => row.review_status === "approved").length;
  document.getElementById("progress").textContent = `${position + 1}/${rows.length} · ${approved} marked reviewed`;
  const root = document.getElementById("case"); root.replaceChildren();
  root.append(el("h2", `${item.id} · ${item.dut_id}`),
              el("p", `Failed test ${item.failed_test} · ${data.category} · ${item.source.source}`),
              el("h3", "Current observation"), el("pre", data.observation),
              el("h3", "Question"), el("p", item.question));
  const status = el("select");
  for (const [value, text] of [["", "Choose expected response"], ["unconfirmed", "Unconfirmed suggestions"], ["refuse", "Refuse: insufficient support"]]) {
    const option = el("option", text); option.value = value; status.append(option);
  }
  status.value = item.expected_status || "";
  status.onchange = () => { item.expected_status = status.value || null; item.review_status = "pending_engineer_review"; render(); };
  root.append(el("h3", "Expected response"), status);
  for (const [kind, field] of [["cause", "acceptable_cause_ids"], ["check", "acceptable_check_ids"]]) {
    root.append(el("h3", kind === "cause" ? "Acceptable possible causes" : "Acceptable next checks"));
    for (const option of item.candidate_options.filter(x => x.kind === kind)) {
      const line = el("label"); line.className = "option";
      const box = el("input"); box.type = "checkbox"; box.checked = item[field].includes(option.id);
      box.onchange = () => {
        item[field] = box.checked ? [...item[field], option.id] : item[field].filter(x => x !== option.id);
        item.review_status = "pending_engineer_review"; render();
      };
      line.append(box, el("span", `${option.text} · ${option.historical_case} · ${option.id}`));
      root.append(line);
    }
  }
  root.append(el("h3", "Review note"));
  const note = el("textarea"); note.rows = 3; note.value = item.review_note || "";
  note.oninput = () => { item.review_note = note.value; item.review_status = "pending_engineer_review"; };
  root.append(note);
  const reviewed = el("label"); reviewed.className = "option";
  const tick = el("input"); tick.type = "checkbox"; tick.checked = item.review_status === "approved";
  tick.onchange = () => {
    item.reviewer = document.getElementById("reviewer").value.trim();
    if (tick.checked && (!item.reviewer || !item.review_note.trim() || !item.expected_status ||
      item.expected_status === "unconfirmed" && (!item.acceptable_cause_ids.length || !item.acceptable_check_ids.length) ||
      item.expected_status === "refuse" && (item.acceptable_cause_ids.length || item.acceptable_check_ids.length))) {
      tick.checked = false; message("Add reviewer, note, status and consistent cause/check choices before marking reviewed."); return;
    }
    item.review_status = tick.checked ? "approved" : "pending_engineer_review";
    message(""); render();
  };
  reviewed.append(tick, el("span", "I reviewed this question and its acceptable answers"));
  root.append(reviewed);
}
document.getElementById("prev").onclick = () => { position = Math.max(0, position - 1); render(); };
document.getElementById("next").onclick = () => { position = Math.min(rows.length - 1, position + 1); render(); };
document.getElementById("draft").onclick = () => download("product_a_gold_draft.json",
  {schema_version: source.schema_version, manifest_sha256: source.manifest_sha256, items: rows});
document.getElementById("export").onclick = () => {
  if (rows.some(row => row.review_status !== "approved" || !row.reviewer)) {
    message("All 50 questions need an engineer's review before export. Save a draft to continue later."); return;
  }
  download("product_a_v1_50_gold_reviewed.json",
    {schema_version: source.schema_version, manifest_sha256: source.manifest_sha256, items: rows});
};
document.getElementById("import").onchange = async event => {
  try {
    const draft = JSON.parse(await event.target.files[0].text());
    if (draft.manifest_sha256 !== source.manifest_sha256 ||
        draft.items.length !== rows.length || draft.items.some((x,i) => x.id !== rows[i].id)) throw Error("Wrong evaluation set.");
    for (let i = 0; i < rows.length; i++) {
      for (const key of ["expected_status", "acceptable_cause_ids", "acceptable_check_ids", "review_note", "reviewer", "review_status"])
        rows[i][key] = clone(draft.items[i][key]);
    }
    document.getElementById("reviewer").value = rows.find(x => x.reviewer)?.reviewer || "";
    message("Draft loaded."); render();
  } catch (error) { message("Could not load this draft: " + error.message); }
};
render();
</script></html>"""


def main():
    manifest, digest = load_manifest(MANIFEST)
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    if template["manifest_sha256"] != digest:
        raise SystemExit("Review template is for a different manifest.")
    by_id = {item["id"]: item for item in manifest["items"]}
    payload = {
        **template,
        "observations": {
            item_id: {"category": row["category"], "observation": row["current"]["observation_text"]}
            for item_id, row in by_id.items()
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    OUTPUT.write_text(HTML.replace("__PAYLOAD__", encoded), encoding="utf-8")
    print("Offline review page:", OUTPUT)


if __name__ == "__main__":
    main()
