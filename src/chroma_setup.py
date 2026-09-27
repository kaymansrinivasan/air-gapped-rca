import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="artifacts/chroma_product_a",
    settings=Settings(anonymized_telemetry=False),
)

collection = client.get_or_create_collection(
    name="product_a_observations",
    embedding_function=None,
)

print("Collection:", collection.name)
print("Records stored:", collection.count())
