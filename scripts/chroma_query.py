from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

ROOT = Path(__file__).resolve().parent

client = chromadb.PersistentClient(
    path=str(ROOT / "artifacts/chroma_product_a"),
    settings=Settings(anonymized_telemetry=False),
)

collection = client.get_collection(
    name="product_a_observations",
    embedding_function=None,
)

# Describe the failure without a DUT ID or an assumed cause.
query = (
    "Continuity Open/Short- test 100 failed. "
    "Failing pins: XCKP, DTO2P, CCK."
)

embedder = DefaultEmbeddingFunction()
vector = embedder([query])[0]

results = collection.query(
    query_embeddings=[[float(value) for value in vector]],
    n_results=5,
    include=["documents", "metadatas", "distances"],
)

print("QUERY:", query)

for i, record_id in enumerate(results["ids"][0]):
    print(f"\n--- Match {i + 1} ---")
    print("DUT:", record_id)
    print("Distance:", round(results["distances"][0][i], 4))
    print("Source:", results["metadatas"][0][i])
    print(results["documents"][0][i])

    