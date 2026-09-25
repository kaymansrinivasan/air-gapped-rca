"""Block 3: offline SQLite FTS5 plus exact-search lexical vectors for Product_A.

Vectors are hashed TF-IDF, not learned/semantic embeddings. No network or model
weights are used. See docs/product_a_retrieval.md before interpreting results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import struct
import sys

DIM = 256
VERSION = "product-a-local-index-v1"
TOKENS = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*|[0-9]+")
DUT = re.compile(r"Product_A-L\d{2}-W\d{2}-D\d{3}", re.IGNORECASE)

def sha256(raw):
    return hashlib.sha256(raw).hexdigest()

def terms(text):
    return TOKENS.findall(text.lower())

def slot(term):
    return int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest(), "big") % DIM

def vector(text, idf):
    weights = [0.0] * DIM
    for term, count in Counter(terms(text)).items():
        if term in idf:
            weights[slot(term)] += (1.0 + math.log(count)) * idf[term]
    norm = math.sqrt(sum(x*x for x in weights))
    return [x/norm for x in weights] if norm else weights

def source_chunks(chunk_dir):
    path = Path(chunk_dir) / "product_a_dut_chunks.jsonl"
    data = path.read_bytes()
    manifest = json.loads((Path(chunk_dir) / "chunk_manifest.json").read_text())
    if sha256(data) != manifest["chunks_sha256"] or manifest["summary"]["dut_chunks"] != 3920:
        raise ValueError("Chunk bundle hash/count mismatch")
    rows = [json.loads(x) for x in data.splitlines()]
    if len(rows) != 3920 or len({r["chunk_id"] for r in rows}) != 3920:
        raise ValueError("Expected 3,920 unique chunks")
    for r in rows:
        if r["origin"] != "GENERATED" or r["stage"] != "observation" or r["cause_status"] != "UNKNOWN":
            raise ValueError("Only cause-unknown Product_A observation chunks can be indexed")
        if r["source_text"] not in r["retrieval_text"]:
            raise ValueError(f"Source text missing from retrieval text: {r['chunk_id']}")
    return rows, data, manifest

def build_index(chunk_dir, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Index output exists: {output}")
    rows, data, upstream = source_chunks(chunk_dir)
    output.mkdir(parents=True)
    db_path = output / "product_a_index.sqlite3"
    df = Counter()
    for row in rows:
        df.update(set(terms(row["retrieval_text"])))
    idf = {term: 1.0 + math.log((len(rows) + 1) / (freq + 1)) for term, freq in df.items()}
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE docs (chunk_id TEXT PRIMARY KEY, dut_id TEXT UNIQUE NOT NULL, "
                   "outcome TEXT NOT NULL, failed_test INTEGER, bin INTEGER NOT NULL, "
                   "source_file TEXT NOT NULL, source_sha256 TEXT NOT NULL, "
                   "line_start INTEGER NOT NULL, line_end INTEGER NOT NULL, "
                   "is_failure INTEGER NOT NULL, vector BLOB NOT NULL)")
        try:
            db.execute("CREATE VIRTUAL TABLE fts USING fts5(chunk_id UNINDEXED, content)")
        except sqlite3.OperationalError as exc:
            raise RuntimeError("The local SQLite build needs FTS5 enabled") from exc
        db.execute("CREATE TABLE vocab (term TEXT PRIMARY KEY, idf REAL NOT NULL)")
        db.executemany("INSERT INTO vocab VALUES (?,?)", sorted(idf.items()))
        for row in rows:
            blob = struct.pack(f"<{DIM}f", *vector(row["retrieval_text"], idf))
            source_lines = row["source_lines"]
            db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (row["chunk_id"], row["dut_id"], row["outcome"], row["failed_test"],
                        row["bin"], row["source_file"], row["source_sha256"],
                        source_lines["start"], source_lines["end"], int(row["bin"] != 1), blob))
            db.execute("INSERT INTO fts(chunk_id, content) VALUES (?,?)",
                       (row["chunk_id"], row["retrieval_text"]))
        db.execute("CREATE INDEX by_outcome ON docs(outcome, failed_test)")
        db.execute("CREATE INDEX by_failure ON docs(is_failure)")
    audit = {"schema_version": "1.0", "version": VERSION, "documents": len(rows),
             "failure_documents": sum(r["bin"] != 1 for r in rows),
             "keyword": "SQLite FTS5/BM25", "vector": "256-dimensional hashed TF-IDF; flat cosine scan",
             "semantic_embeddings": False, "local_only": True,
             "source_chunks_sha256": sha256(data), "source_manifest_sha256": upstream["input_manifest_sha256"],
             "index_sha256": sha256(db_path.read_bytes())}
    (output / "index_manifest.json").write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    return audit

def inferred_failed_test(query):
    """Filter only when the query unambiguously names one failing test."""
    labels = set()
    if re.search(r"\b(scan|606)\b", query, re.IGNORECASE):
        labels.add(606)
    if re.search(r"\b(idd[_ ]?static|210)\b", query, re.IGNORECASE):
        labels.add(210)
    if re.search(r"\b(continuity|open/short|100)\b", query, re.IGNORECASE):
        labels.add(100)
    return next(iter(labels)) if len(labels) == 1 else None

def query_index(db_path, query, top_k=5, mode="hybrid", failures_only=True,
                exclude_dut=None, failed_test=None):
    if not query.strip() or not 1 <= top_k <= 100 or mode not in {"keyword", "vector", "hybrid"}:
        raise ValueError("Need nonempty query, 1–100 results, valid mode")
    found = DUT.search(query)
    if found and exclude_dut is None:
        exclude_dut = found.group(0)  # do not retrieve the query DUT as its own history
    search_text = DUT.sub(" ", query)  # identity tokens should not dominate similarity
    if failed_test is None and failures_only:
        failed_test = inferred_failed_test(search_text)
    if failed_test is not None and failed_test not in {100, 210, 606}:
        raise ValueError("Unknown failed test filter")
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        token_list = list(dict.fromkeys(terms(search_text)))
        candidates = {}
        rank_limit = max(top_k * 10, 50)
        where = (" AND docs.is_failure=1" if failures_only else "")
        if failed_test is not None:
            where += f" AND docs.failed_test={failed_test}"
        if mode in {"keyword", "hybrid"} and token_list:
            expression = " OR ".join('"' + token.replace('"', '') + '"' for token in token_list)
            sql = ("SELECT docs.chunk_id FROM fts JOIN docs ON fts.chunk_id=docs.chunk_id "
                   f"WHERE fts MATCH ?{where} ORDER BY bm25(fts), docs.chunk_id LIMIT ?")
            for position, row in enumerate(db.execute(sql, (expression, rank_limit)), 1):
                candidates.setdefault(row["chunk_id"], {})["keyword_rank"] = position
        if mode in {"vector", "hybrid"}:
            idf = dict(db.execute("SELECT term,idf FROM vocab"))
            q = vector(search_text, idf)
            if any(q):
                scored = []
                conditions = []
                if failures_only:
                    conditions.append("is_failure=1")
                if failed_test is not None:
                    conditions.append(f"failed_test={failed_test}")
                sql = "SELECT chunk_id, vector FROM docs" + (" WHERE " + " AND ".join(conditions) if conditions else "")
                for row in db.execute(sql):
                    stored = struct.unpack(f"<{DIM}f", row["vector"])
                    score = sum(a*b for a,b in zip(q, stored))
                    if score > 0:
                        scored.append((score, row["chunk_id"]))
                scored.sort(key=lambda item: (-item[0], item[1]))
                for position, (_, chunk_id) in enumerate(scored[:rank_limit], 1):
                    candidates.setdefault(chunk_id, {})["vector_rank"] = position
        ranked = []
        for chunk_id, ranks in candidates.items():
            score = sum(1.0/(60 + position) for position in ranks.values())
            ranked.append((score, chunk_id, ranks))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        output = []
        for score, chunk_id, ranks in ranked:
            row = db.execute("SELECT * FROM docs WHERE chunk_id=?", (chunk_id,)).fetchone()
            if exclude_dut and row["dut_id"].lower() == exclude_dut.lower():
                continue
            output.append({"chunk_id": chunk_id, "dut_id": row["dut_id"],
                           "outcome": row["outcome"], "failed_test": row["failed_test"],
                           "bin": row["bin"], "score": round(score, 8), **ranks,
                           "source_file": row["source_file"], "source_sha256": row["source_sha256"],
                           "source_lines": {"start": row["line_start"], "end": row["line_end"]},
                           "cause_status": "UNKNOWN"})
            if len(output) >= top_k:
                break
        return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build")
    build.add_argument("--chunks", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    search = sub.add_parser("search")
    search.add_argument("--db", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--mode", choices=("keyword", "vector", "hybrid"), default="hybrid")
    search.add_argument("--include-passing", action="store_true")
    search.add_argument("--exclude-dut")
    search.add_argument("--failed-test", type=int, choices=(100,210,606))
    args = parser.parse_args()
    try:
        if args.action == "build":
            result = build_index(args.chunks, args.output)
        else:
            result = query_index(args.db, args.query, args.top_k, args.mode,
                                 not args.include_passing, args.exclude_dut, args.failed_test)
        print(json.dumps(result, sort_keys=True, indent=2))
    except (OSError, ValueError, RuntimeError, sqlite3.Error, KeyError) as exc:
        parser.exit(1, f"Index failed: {exc}\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
