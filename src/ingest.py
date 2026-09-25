"""Block 2: source-linked Product_A DUT observation chunks. Standard library only."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSION = "product-a-dut-chunks-v1"
LOT = re.compile(r"^Lot:\s+(L\d+)\s+Tester:\s+\S+\s+Program:\s+\S+")
DEVICE = re.compile(r"^Device:\s+(\d+)\s+Station:")
BIN = re.compile(r"^Bin:\s+(\d+)\s+Wafer Coordinates:")

class ChunkError(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise ChunkError(message)

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def stable_json(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")

def source_blocks(source, raw, lot, wafer):
    """Exact inclusive one-based line spans for complete DUT blocks."""
    lines = raw.decode("utf-8").splitlines(keepends=True)
    require(lines and f"Wafer: {wafer} " in "".join(lines[:4]), f"{source}: missing wafer header")
    require(any(x.startswith(f"End of Wafer: {wafer} ") for x in lines[-4:]),
            f"{source}: missing wafer footer")
    starts = [i for i, line in enumerate(lines) if LOT.match(line)]
    require(len(starts) == 112, f"{source}: expected 112 DUT starts")
    blocks = []
    for i, start in enumerate(starts):
        boundary = starts[i + 1] if i + 1 < len(starts) else len(lines)
        require(LOT.match(lines[start]).group(1) == lot, f"{source}:{start + 1}: wrong lot")
        require(start + 1 < boundary, f"{source}:{start + 1}: missing device row")
        device = DEVICE.match(lines[start + 1])
        require(device is not None, f"{source}:{start + 2}: missing device header")
        bins = [(j, BIN.match(lines[j])) for j in range(start + 2, boundary) if BIN.match(lines[j])]
        require(len(bins) == 1, f"{source}:{start + 1}: expected one bin")
        end, match = bins[0]
        require(not any(x.strip() for x in lines[end + 1:boundary]
                        if not x.startswith("End of Wafer:")),
                f"{source}:{end + 1}: unexpected content after bin")
        blocks.append({"device": f"D{int(device.group(1)):03d}", "bin": int(match.group(1)),
                       "source_lines": {"start": start + 1, "end": end + 1},
                       "text": "".join(lines[start:end + 1])})
    return blocks

def build(repo_root=ROOT):
    root = Path(repo_root).resolve()
    dataset = root / "syn_data" / "product_a_v1"
    sys.path.insert(0, str(dataset))
    try:
        from validate import validate_dataset
        report = validate_dataset(dataset)  # independently reparses logs, metadata and manifest
    finally:
        sys.path.pop(0)
    metadata_bytes = (dataset / "device_metadata.jsonl").read_bytes()
    metadata = [json.loads(line) for line in metadata_bytes.splitlines()]
    by_id = {row["dut_id"]: row for row in metadata}
    require(len(metadata) == len(by_id) == 3920, "Expected 3,920 unique DUT metadata records")
    chunks, sources, outcomes = [], [], Counter()
    for lot_number in range(1, 6):
        lot = f"L{lot_number:02d}"
        for wafer_number in range(1, 8):
            wafer = f"W{wafer_number:02d}"
            rel = f"syn_data/product_a_v1/logs/{lot}/{wafer}.txt"
            raw = (root / rel).read_bytes()
            digest = sha256(raw)
            blocks = source_blocks(rel, raw, lot, wafer)
            sources.append({"path": rel, "sha256": digest, "device_blocks": len(blocks)})
            for block in blocks:
                dut_id = f"Product_A-{lot}-{wafer}-{block['device']}"
                require(dut_id in by_id, f"{rel}: missing metadata for {dut_id}")
                row = by_id[dut_id]
                require(row["origin"] == "GENERATED" and row["bin"] == block["bin"],
                        f"{rel}: origin/bin differs from validated log")
                tests = row["tests"]
                require(tests and tests[-1]["status"] == ("PASS" if row["bin"] == 1 else "FAIL"),
                        f"{dut_id}: terminal test contradicts bin")
                test_info = [f"{t['number']} {t['name']} {t['status']}" for t in tests]
                retrieval_text = (f"{dut_id}; lot {lot}; wafer {wafer}; die coordinates "
                                  f"{row['coordinate'][0]}, {row['coordinate'][1]}; "
                                  f"outcome {row['outcome']}; bin {block['bin']}; "
                                  f"executed tests: {', '.join(test_info)}\n{block['text']}")
                chunk = {
                    "schema_version": "1.0", "chunker_version": VERSION,
                    "chunk_id": "pa1-" + hashlib.sha256(
                        (dut_id + "|" + digest + "|" + str(block["source_lines"]["start"])).encode()
                    ).hexdigest()[:24],
                    "dut_id": dut_id, "product": "Product_A", "lot": lot, "wafer": wafer,
                    "device": block["device"], "coordinate": row["coordinate"],
                    "origin": "GENERATED", "stage": "observation", "cause_status": "UNKNOWN",
                    "outcome": row["outcome"], "bin": block["bin"],
                    "executed_tests": [t["number"] for t in tests],
                    "failed_test": next((t["number"] for t in tests if t["status"] == "FAIL"), None),
                    "source_file": rel, "source_sha256": digest,
                    "source_lines": block["source_lines"], "source_text": block["text"],
                    "retrieval_text": retrieval_text,
                }
                require(len(retrieval_text) <= 2000, f"{dut_id}: oversized DUT chunk")
                chunks.append(chunk)
                outcomes[row["outcome"]] += 1
    require(len(chunks) == 3920 and len({x["dut_id"] for x in chunks}) == 3920,
            "Chunk count or identity mismatch")
    require(len({x["chunk_id"] for x in chunks}) == 3920, "Chunk ID collision")
    require(dict(outcomes) == report["overall"]["outcome_counts"], "Outcome count mismatch")
    audit = {"schema_version": "1.0", "chunker_version": VERSION,
             "input_manifest_sha256": sha256((dataset / "manifest.json").read_bytes()),
             "metadata_sha256": sha256(metadata_bytes), "source_files": sources,
             "summary": {"dut_chunks": len(chunks), "wafers": len(sources),
                         "failed_chunks": sum(x["bin"] != 1 for x in chunks),
                         "outcomes": dict(outcomes),
                         "executed_tests": report["overall"]["executed_tests"]},
             "scope": "Generated DUT observations only. No diagnosis, case history, or current query."}
    return chunks, audit

def write_bundle(output, chunks, audit):
    require(not output.exists(), f"Output already exists: {output}")
    output.mkdir(parents=True)
    data = b"".join(stable_json(x) for x in chunks)
    (output / "product_a_dut_chunks.jsonl").write_bytes(data)
    audit["chunks_sha256"] = sha256(data)
    (output / "chunk_manifest.json").write_bytes(stable_json(audit))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        root, output = args.repo_root.resolve(), args.output.resolve()
        require(not output.is_relative_to(root / "syn_data"), "Output must be outside syn_data")
        chunks, audit = build(root)
        write_bundle(output, chunks, audit)
    except (ChunkError, OSError, UnicodeError, ValueError, KeyError) as exc:
        parser.exit(1, f"Ingestion failed: {exc}\n")
    print(json.dumps(audit["summary"], sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
