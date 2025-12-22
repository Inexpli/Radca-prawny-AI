import re
import os
import uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter

DATA_SOURCES = [
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19970880553/U/D19970553Lj.pdf",
        "file_path": "data/rag/kodeks_karny.md",
        "collection_name": "kodeks_karny",
        "source_label": "Kodeks Karny"
    },
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19640160093/U/D19640093Lj.pdf",
        "file_path": "data/rag/kodeks_cywilny.md",
        "collection_name": "kodeks_cywilny",
        "source_label": "Kodeks Cywilny"
    },
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19740240141/U/D19740141Lj.pdf",
        "file_path": "data/rag/kodeks_pracy.md",
        "collection_name": "kodeks_pracy",
        "source_label": "Kodeks Pracy"
    },
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19640090059/U/D19640059Lj.pdf",
        "file_path": "data/rag/kodeks_rodzinny_i_opiekunczy.md",
        "collection_name": "kodeks_rodzinny_i_opiekunczy",
        "source_label": "Kodeks Rodzinny i Opiekuńczy"
    },
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19710120114/U/D19710114Lj.pdf",
        "file_path": "data/rag/kodeks_wykroczen.md",
        "collection_name": "kodeks_wykroczen",
        "source_label": "Kodeks Wykroczeń"
    },
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19970780483/U/D19970483Lj.pdf",
        "file_path": "data/rag/konstytucja_rp.md",
        "collection_name": "konstytucja_rp",
        "source_label": "Konstytucja RP"
    }
]

QDRANT_PATH = "./qdrant_data"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
VECTOR_SIZE = 1024


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
    Główna logika przetwarzania pojedynczego źródła:
    Konwersja -> Parsowanie -> Embeddings -> Qdrant
    """
    collection_name = config["collection_name"]
    file_path = config["file_path"]
    source_url = config["url"]
    source_label = config["source_label"]

    print(f"\n--- Przetwarzanie: {source_label} ({collection_name}) ---")

    convert_pdf_to_markdown(source_url, file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    clean_text = clean_noise(raw_text)
    articles = parse_legal_act(clean_text, source_label)

    if not articles:
        print("WARN: Nie znaleziono artykułów. Sprawdź format pliku lub regex.")
        return

    if client.collection_exists(collection_name):
        print(f"LOG: Usuwanie starej kolekcji {collection_name}...")
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
    )

    print("LOG: Generowanie embeddingów...")
    documents_to_embed = []
    
    for art in articles:
        semantic_text = f"passage: {art['metadata']['source']} {art['metadata']['chapter']} {art['metadata']['article']} {art['text']}"
        documents_to_embed.append(semantic_text)

    batch_size = 64
    all_embeddings = []
    
    for i in range(0, len(documents_to_embed), batch_size):
        batch = documents_to_embed[i : i + batch_size]
        batch_embeddings = embedder.encode(batch, normalize_embeddings=True)
        all_embeddings.extend(batch_embeddings)

    print(f"LOG: Wysyłanie {len(articles)} punktów do Qdrant...")
    points = []
    for idx, art in enumerate(articles):
        unique_string = f"{art['metadata']['source']}_{art['metadata']['article']}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))
        points.append(models.PointStruct(
            id=point_id,
            vector=all_embeddings[idx].tolist(),
            payload={
                "text": art['text'],
                "chapter": art['metadata']['chapter'],
                "article": art['metadata']['article'],
                "source": art['metadata']['source'],
                "full_markdown": f"## {art['metadata']['article']}\n{art['text']}" 
            }
        ))
        
        if len(points) >= 100:
            client.upsert(collection_name=collection_name, points=points)
            points = []

    if points:
        client.upsert(collection_name=collection_name, points=points)

    print(f"SUKCES: Zakończono dla {collection_name}.")


def main():

    print("LOG: Inicjalizacja modelu embeddingów (to może chwilę potrwać)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")
    
    print("LOG: Łączenie z Qdrant...")
    client = QdrantClient(path=QDRANT_PATH)

    for config in DATA_SOURCES:
        process_and_index(client, embedder, config)
    
    print("\nLOG: Wszystkie operacje zakończone pomyślnie.")

if __name__ == "__main__":
    main()