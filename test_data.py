from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(path="./qdrant_data")
embedder = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")

query = "Co grozi za kradzież z włamaniem?"
query_vector = embedder.encode(f"query: {query}", normalize_embeddings=True)

try:
    hits = client.query_points(
        collection_name="kodeks_karny",
        with_payload=True,
        query=query_vector,
        limit=3
    )
finally:
    client.close()

print(f"Pytanie: {query}\n")
print(hits.model_dump_json(indent=2))