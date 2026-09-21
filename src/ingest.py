"""Offline Block 2 ingestion for A595. See docs/block2_chunking.md.

Only allowlisted evidence becomes chunks. Source bytes are never rewritten.
No model, index, third-party dependency or network is used.
"""

import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

VERSION = "a595-structural-v1"
DEFAULT_SPLIT = "syn_data/splits/first_demo.json"
STAGES = {
    "observed_failure.json": "failure",
    "simulated_investigation.json": "investigation",
    "simulated_action.json": "action",
    "simulated_retest_1.json": "retest",
    "simulated_retest_2.json": "retest",
}


class IngestError(ValueError):
    """An input or citation failed validation."""


def require(condition, message):
    if not condition:
        raise IngestError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise IngestError(f"Non-JSON numeric constant: {value}")


DECODER = json.JSONDecoder(object_pairs_hook=unique_object, parse_constant=reject_constant)


def read_json(data):
    return DECODER.decode(data.decode("utf-8"))


def source_path(root, relative):
    require(isinstance(relative, str) and "\\" not in relative, "Use relative POSIX paths")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and ".." not in path.parts and ":" not in relative,
            f"Unsafe source path: {relative}")
    resolved = (root / relative).resolve()
    require(resolved.is_relative_to(root.resolve()), f"Source escapes repository: {relative}")
    return resolved


class Source:
    """Exact UTF-8 text and inclusive one-based source line addresses."""

    def __init__(self, path, data):
        self.path, self.data = path, data
        self.text = data.decode("utf-8")
        # Only LF terminates a line; preserve CRLF and all original whitespace.
        self.lines = self.text.split("\n")
        self.lines = [line + "\n" for line in self.lines[:-1]] + ([self.lines[-1]] if self.lines[-1] else [])
        self.starts = []
        offset = 0
        for line in self.lines:
            self.starts.append(offset)
            offset += len(line)
        self.sha = sha256(data)

    def line_at(self, offset):
        return bisect_right(self.starts, offset)

    def citation(self, start, end):
        require(type(start) is int and type(end) is int and 1 <= start <= end <= len(self.lines),
                f"Invalid source lines for {self.path}: {start}-{end}")
        return {"source_id": self.path, "source_file": self.path, "source_sha256": self.sha,
                "source_lines": {"start": start, "end": end},
                "text": "".join(self.lines[start - 1:end])}


def array_item_spans(text, field):
    """Locate top-level array objects without reserializing the source JSON."""
    def skip(pos):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        return pos

    pos = skip(0) + 1
    while True:
        pos = skip(pos)
        if text[pos] == "}":
            break
        key, pos = DECODER.raw_decode(text, pos)
        pos = skip(pos)
        require(text[pos] == ":", "Expected JSON field separator")
        pos = skip(pos + 1)
        if key == field:
            require(text[pos] == "[", f"{field} must be an array")
            spans = []
            pos = skip(pos + 1)
            while text[pos] != "]":
                start = pos
                _, pos = DECODER.raw_decode(text, pos)
                spans.append((start, pos))
                pos = skip(pos)
                if text[pos] == ",":
                    pos = skip(pos + 1)
            return spans
        _, pos = DECODER.raw_decode(text, pos)
        pos = skip(pos)
        if text[pos] == ",":
            pos += 1
    raise IngestError(f"Missing array {field}")


def load_inputs(root, split_path):
    config_bytes = source_path(root, split_path).read_bytes()
    config = read_json(config_bytes)
    require(config.get("schema_version") == "1.1", "Expected split schema 1.1")
    require(config.get("path_base") == "repository_root", "Unsupported path base")
    policy = config["input_policy"]
    require(policy.get("default") == "EXCLUDE" and
            policy.get("recursive_directory_loading_allowed") is False and
            policy.get("exclude_metadata_from_model") is True, "Unsafe input policy")
    historical_ids = config["historical"]["incident_ids"]
    query_id = config["query"]["incident_id"]
    require(historical_ids and len(historical_ids) == len(set(historical_ids)),
            "Historical incident IDs must be nonempty and unique")
    require(query_id not in historical_ids, "Current incident appears in historical inputs")
    require(config["query"]["stage"] == "failure", "Only an incoming failure is supported")
    require(isinstance(config["query"]["question"], str) and config["query"]["question"].strip(),
            "Missing engineer question")
    excluded = set()
    for section, key in (("evaluator_only", "held_out_files"), ("reserved", "files")):
        for entry in config.get(section, {}).get(key, []):
            excluded.add(entry["path"] if isinstance(entry, dict) else entry)
    answer = config.get("evaluator_only", {}).get("reference_answers")
    if answer:
        excluded.add(answer["path"] if isinstance(answer, dict) else answer)
    reserved_ids = set(config.get("reserved", {}).get("incident_ids", []))
    require(not reserved_ids.intersection([*historical_ids, query_id]), "Reserved incident overlap")
    records, paths, evidence_ids = [], set(), set()
    for role, section, allowed_ids in (("historical", "historical", historical_ids),
                                     ("current", "query", [query_id])):
        entries = config[section]["files"]
        require(entries, f"No {role} inputs")
        for entry in entries:
            path = entry["path"]
            require(path not in paths and path not in excluded, f"Duplicate/excluded input: {path}")
            parts = PurePosixPath(path).parts
            require(len(parts) == 4 and parts[:2] == ("syn_data", "incidents") and
                    parts[2] in allowed_ids and parts[3] in STAGES,
                    f"Not an allowed incident evidence path: {path}")
            stage = STAGES[parts[3]]
            require(role != "current" or stage == "failure", "Current input contains future evidence")
            data = source_path(root, path).read_bytes()
            require(git_blob_sha(data) == entry["git_blob_sha"], f"Git blob hash mismatch: {path}")
            document = read_json(data)
            require(document["incident_id"] == parts[2], f"Incident/path mismatch: {path}")
            evidence_id = document["evidence_id"]
            require(evidence_id == entry["evidence_id"] and evidence_id not in evidence_ids,
                    f"Duplicate or mismatched evidence ID: {path}")
            require(document.get("origin") in {"SIMULATED", "REAL_OBSERVED"}, f"Missing origin: {path}")
            require(isinstance(document.get("synthetic_device_key"), str) and
                    document["synthetic_device_key"], f"Missing device identity: {path}")
            require(stage == "failure" or document["origin"] == "SIMULATED",
                    f"This demo supports only simulated follow-up records: {path}")
            if stage == "retest":
                tests = document.get("tests")
                require(isinstance(tests, list) and tests and
                        document.get("test_count") == len(tests), f"Retest count mismatch: {path}")
                for test in tests:
                    require(isinstance(test, dict) and type(test.get("test_number")) is int and
                            "kind" in test and test.get("origin") == "SIMULATED",
                            f"Invalid simulated test record: {path}")
            records.append({"role": role, "stage": stage, "document": document,
                            "source": Source(path, data)})
            paths.add(path)
            evidence_ids.add(evidence_id)
    found = {r["document"]["incident_id"] for r in records if r["stage"] == "failure"}
    require(found == set([*historical_ids, query_id]), "Every incident needs its failure record")
    by_id = {r["document"]["evidence_id"]: r["document"] for r in records}
    for record in records:
        document = record["document"]
        for cited in document.get("cites", []):
            require(cited in by_id and by_id[cited]["incident_id"] == document["incident_id"],
                    f"Unresolved/cross-incident evidence link: {cited}")
    return config, config_bytes, records


def resolve_original(root, document, cache):
    """Resolve measured failure excerpts only; donor templates are not retests."""
    citation = document.get("source_citation")
    if not citation:
        require(document["origin"] != "REAL_OBSERVED", "Measured failure lacks citation")
        return []
    require(document["origin"] == "REAL_OBSERVED", "Simulated evidence cannot claim measured citation")
    require(citation["file"].startswith("source/"), "Unexpected original citation path")
    relative = "syn_data/" + citation["file"]
    if relative not in cache:
        cache[relative] = source_path(root, relative).read_bytes()
    require(sha256(cache[relative]) == citation["sha256"], f"Original hash mismatch: {relative}")
    source = Source(relative, cache[relative])
    resolved = source.citation(citation["line_start"], citation["line_end"])
    # Authored excerpts omit the final terminator, retaining internal whitespace.
    quote = resolved["text"]
    quote = quote[:-2] if quote.endswith("\r\n") else quote[:-1] if quote.endswith("\n") else quote
    require(quote == document["excerpt"], f"Original excerpt mismatch: {document['evidence_id']}")
    binary = "syn_data/" + citation["binary_file"]
    require(citation["binary_file"].startswith("source/"), "Unexpected binary citation path")
    if binary not in cache:
        cache[binary] = source_path(root, binary).read_bytes()
    require(sha256(cache[binary]) == citation["binary_sha256"], f"STDF hash mismatch: {binary}")
    require(type(citation["byte_offset"]) is int and
            0 <= citation["byte_offset"] < len(cache[binary]), "Invalid STDF offset")
    resolved["relationship"] = "original_measured_failure"
    resolved["binary_reference"] = {"source_file": binary, "source_sha256": citation["binary_sha256"],
                                    "byte_offset": citation["byte_offset"],
                                    "record_index": citation["record_index"]}
    return [resolved]


def chunk_record(record, failure_devices, original, target_chars):
    source, document = record["source"], record["document"]
    total_lines = len(source.lines)
    groups = [(1, total_lines, [], [], "whole_record")]
    if record["stage"] == "retest" and len(source.text) > target_chars and len(document["tests"]) > 1:
        spans = array_item_spans(source.text, "tests")
        require(len(spans) == len(document["tests"]), "Test span count mismatch")
        ranges = [(source.line_at(start), source.line_at(end - 1)) for start, end in spans]
        # Same-line objects have no disjoint line citations: keep these atomic.
        if all(a[1] < b[0] for a, b in zip(ranges, ranges[1:])):
            # Include blank lines/separators between adjacent objects as well.
            ranges = [(start, ranges[i + 1][0] - 1 if i + 1 < len(ranges) else end)
                      for i, (start, end) in enumerate(ranges)]
            context = []
            if ranges[0][0] > 1:
                context.append(source.citation(1, ranges[0][0] - 1))
            if ranges[-1][1] < total_lines:
                context.append(source.citation(ranges[-1][1] + 1, total_lines))
            context_size = sum(len(c["text"]) for c in context)
            groups, first, previous_family = [], 0, None
            for index, test in enumerate(document["tests"]):
                # Mechanical grouping keys; not inferred diagnoses/physical families.
                family = (test["test_number"] // 100, test["kind"], test.get("unit"))
                size = sum(len(s) for s in source.lines[ranges[first][0] - 1:ranges[index][1]])
                if index > first and (family != previous_family or size + context_size > target_chars):
                    groups.append((ranges[first][0], ranges[index - 1][1],
                                   list(range(first, index)), context, "retest_group"))
                    first = index
                previous_family = family
            groups.append((ranges[first][0], ranges[-1][1], list(range(first, len(spans))),
                           context, "retest_group"))
    result = []
    for start, end, indices, context, kind in groups:
        primary = source.citation(start, end)
        if kind == "whole_record" and record["stage"] == "retest":
            indices = list(range(len(document["tests"])))
        anchor = document.get("source_anchor")
        device = document["synthetic_device_key"]
        expected_device = failure_devices[document["incident_id"]]
        identity = f"{VERSION}|{record['role']}|{source.path}|{source.sha}|{start}:{end}"
        # Each verbatim piece has its own address; concatenation is not one quote.
        ordered_spans = sorted([*context, primary], key=lambda c: c["source_lines"]["start"])
        retrieval_text = "\n".join(c["text"] for c in ordered_spans)
        result.append({"schema_version": "1.0", "chunk_id": "chunk-" + sha256(identity.encode())[:24],
                       "chunker_version": VERSION, "role": record["role"], "kind": kind,
                       "incident_id": document["incident_id"], "evidence_id": document["evidence_id"],
                       "device_key": device, "incident_device_key": expected_device,
                       "device_matches_incident": device == expected_device,
                       "source_anchor": anchor,
                       "shared_source_failure_key": anchor.get("source_device_key") if anchor else None,
                       "stage": record["stage"], "origin": document["origin"],
                       "linked_evidence_ids": document.get("cites", []),
                       **primary, "context": context, "original_citations": original,
                       "test_indices": indices,
                       "test_numbers": [document["tests"][i]["test_number"] for i in indices],
                       "json_pointers": [f"/tests/{i}" for i in indices] if kind == "retest_group" else [""],
                       "retrieval_text": retrieval_text, "retrieval_chars": len(retrieval_text),
                       "over_target": len(retrieval_text) > target_chars})
    return result


def build(root, split_path=DEFAULT_SPLIT, target_chars=6000):
    require(type(target_chars) is int and target_chars >= 512, "Target must be at least 512 characters")
    root = Path(root).resolve()
    config, config_bytes, records = load_inputs(root, split_path)
    devices = {r["document"]["incident_id"]: r["document"]["synthetic_device_key"]
               for r in records if r["stage"] == "failure"}
    chunks, originals, inventory = {"historical": [], "current": []}, {}, []
    for record in records:
        document, source = record["document"], record["source"]
        citations = resolve_original(root, document, originals) if record["stage"] == "failure" else []
        created = chunk_record(record, devices, citations, target_chars)
        chunks[record["role"]].extend(created)
        covered = set()
        for chunk in created:
            for span in [chunk, *chunk["context"]]:
                bounds = span["source_lines"]
                require(span["text"] == source.citation(bounds["start"], bounds["end"])["text"],
                        f"Chunk quote mismatch: {source.path}")
                covered.update(range(bounds["start"], bounds["end"] + 1))
        require(covered == set(range(1, len(source.lines) + 1)), f"Source lines lost: {source.path}")
        if record["stage"] == "retest":
            indices = [i for chunk in created for i in chunk["test_indices"]]
            require(indices == list(range(document["test_count"])), f"Tests lost/duplicated: {source.path}")
        inventory.append({"source_file": source.path, "source_sha256": source.sha,
                          "git_blob_sha": git_blob_sha(source.data), "role": record["role"],
                          "evidence_id": document["evidence_id"], "lines": len(source.lines),
                          "chunks": len(created)})
    all_chunks = chunks["historical"] + chunks["current"]
    require(len({c["chunk_id"] for c in all_chunks}) == len(all_chunks), "Duplicate chunk IDs")
    audit = {"schema_version": "1.0", "chunker_version": VERSION, "split_id": config["split_id"],
             "split_file": split_path, "split_sha256": sha256(config_bytes), "target_chars": target_chars,
             "intended_use": config["intended_use"], "inputs": inventory,
             "resolved_original_files": {p: sha256(b) for p, b in sorted(originals.items())},
             "summary": {"evidence_files": len(records),
                         "historical_chunks": len(chunks["historical"]),
                         "current_chunks": len(chunks["current"]),
                         "historical_incidents": len(config["historical"]["incident_ids"]),
                         "retest_results": sum(len(c["test_indices"]) for c in all_chunks),
                         "chunks_by_stage": dict(Counter(c["stage"] for c in all_chunks)),
                         "over_target_chunks": sum(c["over_target"] for c in all_chunks)},
             "checks": {"allowlist_and_blob_hashes": "passed", "source_line_coverage": "passed",
                        "exact_quotes": "passed", "retest_coverage": "passed",
                        "original_failure_excerpts": "passed", "historical_current_separation": "passed"},
             "limitations": ["Synthetic demonstration; no cause accuracy or retrieval performance measured.",
                             "Character target is not a tokenizer/context-window guarantee.",
                             "STDF hashes and offset bounds checked; record indices preserved, not decoded.",
                             "Retest donor references describe templates, not actual follow-up measurements."]}
    return chunks, audit, config["query"]["question"]


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_bundle(output, chunks, audit, question):
    """Use a new directory; write the completion manifest last."""
    output = Path(output)
    require(not output.exists(), f"Output already exists; choose a new directory: {output}")
    files = {}
    for role, name in (("historical", "historical_chunks.jsonl"), ("current", "current_case_chunks.jsonl")):
        files[name] = b"".join((json.dumps(c, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                               for c in chunks[role])
    files["question.txt"] = (question + "\n").encode("utf-8")
    audit["outputs"] = {name: {"sha256": sha256(data), "bytes": len(data)} for name, data in files.items()}
    output.mkdir(parents=True)
    for name, data in files.items():
        (output / name).write_bytes(data)
    (output / "chunk_manifest.json").write_bytes(json_bytes(audit))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Repository-relative split config")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--target-chars", type=int, default=6000)
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        evidence_root = (args.repo_root / "syn_data").resolve()
        require(not output.is_relative_to(evidence_root), "Output must be outside immutable syn_data")
        chunks, audit, question = build(args.repo_root, args.split, args.target_chars)
        write_bundle(output, chunks, audit, question)
    except (IngestError, OSError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(audit["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
