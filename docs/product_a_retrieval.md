# Product_A Blocks 2 and 3: observation chunks and local search

This prototype reads the immutable generated Product_A wafer logs under
`syn_data/product_a_v1/logs/`. It does not infer root causes from a failed bin.
There are no reviewed Product_A investigations, actions, or retests yet.

## Block 2: one source-linked chunk per DUT

From the repository root, using Python 3.10 or later:

```sh
python src/ingest.py --output artifacts/product_a_chunks
```

The script runs the existing Product_A read-back validator against all 35 raw
wafer files, metadata, and hashes. It then writes
`product_a_dut_chunks.jsonl` (3,920 observations) and `chunk_manifest.json`
into a new output directory. `artifacts/` is gitignored. For every DUT, a
chunk contains the complete contiguous `Lot:` through `Bin:` device block,
full identity, observed test numbers and bin, its exact UTF-8 source file SHA-256,
and inclusive 1-based line range. The raw log bytes never change.

All 3,528 passes and 392 failures are represented. Failure counts: 98 at
Open/Short- (Bin 4), 98 at IDD_Static (Bin 3), 196 at Scan (Bin 2). The
executed test count is 11,466 because later tests do not run after a failure.
`cause_status=UNKNOWN` on every chunk. Generated origin is recorded in the
chunk metadata, outside the raw tester log. Wafer headers and footers remain
in the source files and are checked by the upstream validator; DUT chunks cite
only contiguous DUT blocks.

## Block 3: local indexes

```sh
python src/index.py build --chunks artifacts/product_a_chunks --output artifacts/product_a_index
python src/index.py search --db artifacts/product_a_index/product_a_index.sqlite3 --query "continuity Open/Short- TSTIN" --top-k 5
python src/index.py search --db artifacts/product_a_index/product_a_index.sqlite3 --query "scan failed DTOP Product_A-L01-W02-D005" --top-k 5
python -m unittest discover -s tests -v
```

The output is a single local SQLite file plus an index manifest. FTS5 supplies
keyword search and BM25 ranking. A 256-dimensional hashed TF-IDF vector is
stored per chunk; cosine similarity scans the local vectors exactly. Reciprocal
rank fusion combines the two rankings. There is no server or network call. This
is an inexpensive **lexical vector baseline**: it has no pretrained semantic
embedding model and should not be reported as semantic retrieval. SQLite must
be compiled with FTS5 on each target board; verify this during offline setup.
No separate vector database service is needed for this corpus size.

Search returns failures by default, infers an unambiguous requested failing
test (100/210/606), and excludes an explicitly named current DUT from its
own historical results. `--include-passing` enables passing controls;
`--failed-test` explicitly filters 100, 210 or 606. Returned citations retain
original file hash and line range. These observations support *which test
failed*, not an explanation for why it failed. A formal historical/current
case split, engineer-reviewed causes, a local semantic embedding model,
retrieval evaluation, LLM ranking, citation verification and refusal behavior
remain separate future work. When a reviewed case history is available, add
it as a distinct stage linked by full DUT/incident identity; never retrofit
generated causes into raw observations.

## Reproducibility and checks

The chunker refuses a preexisting output directory, verifies source hashes,
exact spans, origin, stop rule and counts. The index verifies the chunk bundle
hash and does not accept case histories disguised as observations. Tests verify
all 3,920 exact quotes, representative failure paths, query-DUT exclusion,
failure-type filtering, and rejection of a tampered chunk bundle. Search
quality, energy, latency and false-answer rate are not yet measured on the
Jetson AGX Orin or RB3 Gen 2.
