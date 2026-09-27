import json
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Read our existing DUT chunks.
with open(
    ROOT / "artifacts/product_a_chunks/product_a_dut_chunks.jsonl",
    encoding="utf-8",
) as file:
    chunks = [json.loads(line) for line in file if line.strip()]

# Pick one DUT that failed continuity.
chunk = next(c for c in chunks if c["failed_test"] == 100)

# Convert its text into a vector.
embedder = DefaultEmbeddingFunction()
vector = embedder([chunk["retrieval_text"]])[0]

print("DUT:", chunk["dut_id"])
print("Text being embedded:\n", chunk["retrieval_text"])
print("Numbers in this vector:", len(vector))
print("First 8 numbers:", [round(float(x), 4) for x in vector[:8]])
