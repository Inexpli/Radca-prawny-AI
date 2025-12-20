import re
from typing import List, Dict
from qdrant_client.http import models
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter

def convert_pdf_to_markdown(source: str) -> None:
    converter = DocumentConverter()
    doc = converter.convert(source).document

    output_path = "data/rag/kodeks_karny.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc.export_to_markdown())

    print(f"LOG: Document saved to {output_path}")

def clean_noise(text: str) -> str:
    lines = text.split('\n')
    cleaned_lines = []
    skip_block = False
    
    for line in lines:
        s_line = line.strip()
        if s_line.startswith("1) Niniejsza ustawa") or s_line.startswith("Opracowano na podstawie"):
            skip_block = True
            continue
        
        if re.match(r'^-?\s*Art\.', s_line):
            skip_block = False
            
        if not skip_block:
            cleaned_lines.append(line)
    
    print("LOG: Noise cleaned from text.")
    return "\n".join(cleaned_lines)

def parse_kodeks(text: str) -> List[Dict]:
    art_pattern = re.compile(r'^(?:- )?Art\.\s+(\d+[a-z]?)\.', re.MULTILINE)
    chapter_pattern = re.compile(r'^##\s+(Rozdział\s+[IVXLCDM]+)', re.MULTILINE)
    
    chunks = []
    lines = text.split('\n')
    
    current_chapter = "Część Ogólna"
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
                        "article_number": current_art_num,
                        "text": full_text,
                        "metadata": {
                            "source": "Kodeks Karny",
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
            "article_number": current_art_num,
            "text": "\n".join(current_content).strip(),
            "metadata": {
                "source": "Kodeks Karny",
                "chapter": current_chapter,
                "article": f"Art. {current_art_num}"
            }
        })
        
    print(f"LOG: Parsed {len(chunks)} articles from the document.")
    return chunks


def main():

    SOURCE = "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19970880553/U/D19970553Lj.pdf"
    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large" 
    COLLECTION_NAME = "kodeks_karny"
    QDRANT_PATH = "./qdrant_data"

    print("1. Conversion of PDF document to Markdown...")

    convert_pdf_to_markdown(SOURCE)

    print("2. Processing document...")

    with open("data/rag/kodeks_karny.md", "r", encoding="utf-8") as f:
        raw_text = f.read()
    articles = parse_kodeks(clean_noise(raw_text))

    print(f"LOG: Number of articles: {len(articles)}")

    print("3. Initialization Qdrant and Model...")

    client = QdrantClient(path=QDRANT_PATH) 
    
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME) 
        
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1024,
            distance=models.Distance.COSINE
        )
    )

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")

    print("4. Preparing vectors...")
    
    documents_to_embed = []
    points = []

    for idx, art in enumerate(articles):
        semantic_text = f"passage: {art['metadata']['chapter']} {art['metadata']['article']} {art['text']}"
        documents_to_embed.append(semantic_text)

    batch_size = 64 
    embeddings = []
    
    for i in range(0, len(documents_to_embed), batch_size):
        batch = documents_to_embed[i : i + batch_size]
        batch_embeddings = embedder.encode(batch, normalize_embeddings=True)
        embeddings.extend(batch_embeddings)

    print("5. Upload to Qdrant")
    
    for idx, art in enumerate(articles):
        points.append(models.PointStruct(
            id=idx,
            vector=embeddings[idx].tolist(),
            payload={
                "text": art['text'], 
                "chapter": art['metadata']['chapter'],
                "article": art['metadata']['article'],
                "full_markdown": f"## {art['metadata']['article']}\n{art['text']}"
            }
        ))

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"LOG: Indexed {len(points)} articles in Qdrant.")

if __name__ == "__main__":
    main()

