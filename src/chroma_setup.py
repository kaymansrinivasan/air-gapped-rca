import chromadb
from pathlib import Path
from chromadb.config import Settings

ROOT = Path(__file__).resolve().parents[1]

client = chromadb.PersistentClient(
    path=str(ROOT / "artifacts/chroma_product_a"),
    settings=Settings(anonymized_telemetry=False),
)

collection = client.get_or_create_collection(
    name="product_a_observations",
    embedding_function=None,
)

print("Collection:", collection.name)
print("Records stored:", collection.count())
