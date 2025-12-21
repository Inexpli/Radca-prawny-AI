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