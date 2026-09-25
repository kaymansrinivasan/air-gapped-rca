import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts/product_a_chunks/product_a_dut_chunks.jsonl"

# Read and validate before writing to Chroma.
with SOURCE.open(encoding="utf-8") as file:
    rows = [
        (line_number, json.loads(line))
        for line_number, line in enumerate(file, start=1)
        if line.strip()
    ]

if not rows:
    raise ValueError("The chunk file is empty.")

ids = [chunk["dut_id"] for _, chunk in rows]

if len(ids) != len(set(ids)):
    raise ValueError(
        "Duplicate DUT IDs found. Check chunk IDs before ingestion."
    )

for line_number, chunk in rows:
    text = chunk.get("retrieval_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Missing retrieval_text on line {line_number}")

client = chromadb.PersistentClient(
    path=str(ROOT / "artifacts/chroma_product_a"),
    settings=Settings(anonymized_telemetry=False),
)

collection = client.get_collection(
    name="product_a_observations",
    embedding_function=None,
)

embedder = DefaultEmbeddingFunction()
batch_size = 64

print("Chunks to load:", len(rows))
print("Records before:", collection.count())

for start in range(0, len(rows), batch_size):
    batch = rows[start:start + batch_size]
    documents = [chunk["retrieval_text"] for _, chunk in batch]

    vectors = embedder(documents)

    collection.upsert(
        ids=[chunk["dut_id"] for _, chunk in batch],
        documents=documents,
        embeddings=[
            [float(value) for value in vector]
            for vector in vectors
        ],
        metadatas=[
            {
                "dut_id": chunk["dut_id"],
                "source_file": SOURCE.relative_to(ROOT).as_posix(),
                "source_line": line_number,
            }
            for line_number, chunk in batch
        ],
    )

    print(f"Loaded {start + len(batch)}/{len(rows)}")

print("Records stored:", collection.count())
