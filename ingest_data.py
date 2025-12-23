import re
import os
import uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter
from fastembed import SparseTextEmbedding


MAIN_COLLECTION = "polskie_prawo"

DATA_SOURCES = [
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19970880553/U/D19970553Lj.pdf",
        "file_path": "data/rag/kodeks_karny.md",
        "collection_name": MAIN_COLLECTION,
        "source_label": "Kodeks Karny"
    },
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19640160093/U/D19640093Lj.pdf",
        "file_path": "data/rag/kodeks_cywilny.md",
        "collection_name": MAIN_COLLECTION,
        "source_label": "Kodeks Cywilny"
    },
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19740240141/U/D19740141Lj.pdf",
        "file_path": "data/rag/kodeks_pracy.md",
        "collection_name": MAIN_COLLECTION,
        "source_label": "Kodeks Pracy"
    },
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19640090059/U/D19640059Lj.pdf",
        "file_path": "data/rag/kodeks_rodzinny_i_opiekunczy.md",
        "collection_name": MAIN_COLLECTION,
        "source_label": "Kodeks Rodzinny i Opiekuńczy"
    },
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19710120114/U/D19710114Lj.pdf",
        "file_path": "data/rag/kodeks_wykroczen.md",
        "collection_name": MAIN_COLLECTION,
        "source_label": "Kodeks Wykroczeń"
    },
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19970780483/U/D19970483Lj.pdf",
        "file_path": "data/rag/konstytucja_rp.md",
        "collection_name": MAIN_COLLECTION,
        "source_label": "Konstytucja RP"
    }
]


QDRANT_PATH = "./qdrant_data"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
SPARSE_MODEL_NAME = "Qdrant/bm25"


def ensure_directories(path: str) -> None:
    """Tworzy katalogi, jeśli nie istnieją."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def convert_pdf_to_markdown(source_url: str, output_path: str) -> None:
    """Konwertuje plik PDF z podanego URL do formatu Markdown i zapisuje go lokalnie."""
    ensure_directories(output_path)
    
    # if os.path.exists(output_path):
    #     print(f"LOG: Plik {output_path} już istnieje. Pomijam konwersję.")
    #     return

    print(f"LOG: Konwertowanie {source_url} -> {output_path} ...")
    converter = DocumentConverter()
    doc = converter.convert(source_url).document

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc.export_to_markdown())

    print(f"LOG: Zapisano dokument do {output_path}")


def clean_noise(text: str) -> str:
    """Usuwa niepotrzebne bloki tekstu z dokumentu."""
    lines = text.split('\n')
    cleaned_lines = []
    skip_block = False
    
    for line in lines:
        s_line = line.strip()
        if s_line.startswith("1) Niniejsza ustawa") or s_line.startswith("Opracowano na podstawie") or s_line == "©Kancelaria Sejmu":
            skip_block = True
            continue
        
        if re.match(r'^-?\s*Art\.', s_line):
            skip_block = False
            
        if not skip_block:
            cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)


def parse_legal_act(text: str, source_label: str) -> List[Dict]:
    """
    Parsuje polskie akty prawne (Kodeksy, Konstytucję) dzieląc je na artykuły.
    """
    art_pattern = re.compile(r'^(?:- )?Art\.\s+(\d+[a-z]?)\.', re.MULTILINE)
    chapter_pattern = re.compile(r'^##\s+(Rozdział\s+[IVXLCDM]+|[IVXLCDM]+\.)', re.MULTILINE | re.IGNORECASE)
    
    chunks = []
    lines = text.split('\n')
    
    current_chapter = "Brak rozdziału / Część ogólna"
    current_art_num = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line: continue

        chap_match = chapter_pattern.match(line)
        if chap_match:
            current_chapter = line.replace("##", "").strip()
            continue

        art_match = art_pattern.match(line)
        if art_match:
            if current_art_num:
                full_text = "\n".join(current_content).strip()
                if full_text:
                    chunks.append({
                        "text": full_text,
                        "metadata": {
                            "source": source_label,
                            "chapter": current_chapter,
                            "article": f"Art. {current_art_num}"
                        }
                    })
            
            current_art_num = art_match.group(1)
            current_content = [line]
        else:
            if current_art_num:
                current_content.append(line)
    
    if current_art_num and current_content:
        chunks.append({
            "text": "\n".join(current_content).strip(),
            "metadata": {
                "source": source_label,
                "chapter": current_chapter,
                "article": f"Art. {current_art_num}"
            }
        })
        
    print(f"LOG: Sparsowano {len(chunks)} artykułów dla źródła: {source_label}")
    return chunks


def process_and_index(client: QdrantClient, embedder: SentenceTransformer, config: Dict) -> None:
    """
    Profesjonalna logika przetwarzania:
    - Bezpieczne tworzenie kolekcji
    - Dynamiczne wykrywanie wymiarowości modelu
    - Zoptymalizowany upload
    """
    collection_name = config["collection_name"]
    file_path = config["file_path"]
    source_url = config.get("url")
    source_label = config["source_label"]
    
    embedding_dim = embedder.get_sentence_embedding_dimension()
    sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    print(f"\n--- Przetwarzanie: {source_label} ({collection_name}) ---")

    try:
        convert_pdf_to_markdown(source_url, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        clean_text = clean_noise(raw_text)
        articles = parse_legal_act(clean_text, source_label)

        if not articles:
            raise ValueError("Nie znaleziono artykułów. Sprawdź format pliku lub regex.")

        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=embedding_dim,
                        distance=models.Distance.DOT
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )
            print(f"LOG: Utworzono nową kolekcję: {collection_name}")
        else:
            print(f"LOG: Kolekcja {collection_name} istnieje. Dopisuję/Aktualizuję dane.")

        print("LOG: Generowanie embeddingów...")
        
        documents_to_embed = [
            f"passage: {art['metadata']['source']} {art['metadata']['chapter']} {art['metadata']['article']} {art['text']}"
            for art in articles
        ]

        dense_embeddings = embedder.encode(documents_to_embed, normalize_embeddings=True)
        sparse_embeddings = list(sparse_embedder.embed(documents_to_embed))

        print(f"LOG: Przygotowywanie {len(articles)} punktów...")
        points = []
        for idx, art in enumerate(articles):
            unique_string = f"{art['metadata']['source']}_{art['metadata']['article']}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))
            
            dense_vec = dense_embeddings[idx].tolist()
            sparse_result = sparse_embeddings[idx]
            
            qdrant_sparse_vec = models.SparseVector(
                indices=sparse_result.indices.tolist(),
                values=sparse_result.values.tolist()
            )
            
            payload_data = {
                "text": art['text'],
                "chapter": art['metadata']['chapter'],
                "article": art['metadata']['article'],
                "source": art['metadata']['source'],
                "full_markdown": f"## {art['metadata']['article']}\n{art['text']}" 
            }
            
            points.append(models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "sparse": qdrant_sparse_vec
                },
                payload=payload_data
            ))

        client.upload_points(
            collection_name=collection_name,
            points=points,
            batch_size=100,
            wait=True
        )

        print(f"SUKCES: Zakończono dla {collection_name}. Zaktualizowano {len(points)} rekordów.")

    except Exception as e:
        print(f"BŁĄD: Wystąpił błąd przy przetwarzaniu {source_label}: {str(e)}")
        raise e
    

def main():

    print("LOG: Inicjalizacja modelu embeddingów (to może chwilę potrwać)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")
    
    print("LOG: Łączenie z Qdrant...")
    client = QdrantClient(path=QDRANT_PATH)

    if client.collection_exists(MAIN_COLLECTION):
        print(f"LOG: Kolekcja '{MAIN_COLLECTION}' już istnieje w Qdrant.")
        client.delete_collection(MAIN_COLLECTION)
        print(f"LOG: Usunięto istniejącą kolekcję '{MAIN_COLLECTION}' dla czystej instalacji.")

    for config in DATA_SOURCES:
        process_and_index(client, embedder, config)
    
    print("\nLOG: Wszystkie operacje zakończone pomyślnie.")

if __name__ == "__main__":
    main()