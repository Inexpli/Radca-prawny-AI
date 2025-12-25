import sys
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


def print_system_info():
    """Drukuje informacje o systemie i konfiguracji CUDA."""
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch built with CUDA: {torch.version.cuda}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"Device count: {torch.cuda.device_count()}")
        current_device = torch.cuda.current_device()
        print(f"Current device ID: {current_device}")
        print(f"Device name: {torch.cuda.get_device_name(current_device)}")
        props = torch.cuda.get_device_properties(current_device)
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("Error: CUDA is not available.")


def test_qdrant_query(query: str):
    """Testuje zapytanie do Qdrant i drukuje wyniki."""
    client = QdrantClient(path="./qdrant_data")
    embedder = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")

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

from typing import List, Dict

from qdrant_client import QdrantClient, models

from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding


MODEL_ID = "speakleash/Bielik-11B-v2.6-Instruct"
QDRANT_PATH = "./qdrant_data"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
SPARSE_MODEL = "Qdrant/bm25"
SEARCH_COLLECTION = "polskie_prawo"

client = QdrantClient(path=QDRANT_PATH)
dense_embedder = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL)

def test_new_query(query: str, top_k: int = 5) -> None:
    """
    Szuka w każdej kolekcji, łączy wyniki i zwraca X najlepszych globalnie.
    """
    dense_vector = dense_embedder.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()
    sparse_result = list(sparse_embedder.embed([query]))[0]

    qdrant_sparse_vector = models.SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist()
    )
    
    all_hits = []

    if client.collection_exists(SEARCH_COLLECTION):
        hits = client.query_points(
            collection_name=SEARCH_COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=20,
                ),
                models.Prefetch(
                    query=qdrant_sparse_vector,
                    using="sparse",
                    limit=20,
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k
        ).points
        all_hits.extend(hits)

    print(all_hits)

test_new_query("Co grozi za morderstwo?", top_k=5)