import json
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Read our existing DUT chunks.
with open(
    "artifacts/product_a_chunks/product_a_dut_chunks.jsonl",
    encoding="utf-8",
) as file:
    chunks = [json.loads(line) for line in file if line.strip()]

# Pick one DUT that failed continuity.
chunk = next(c for c in chunks if c["failed_test"] == 100)

# Convert its text into a vector.
embedder = DefaultEmbeddingFunction()
vector = embedder([chunk["retrieval_text"]])[0]

print("DUT:", chunk["dut_id"])
print("Numbers in this vector:", len(vector))
print("First 8 numbers:", [round(float(x), 4) for x in vector[:8]])
