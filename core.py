import torch
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from typing import List, Dict
from config import CONFIG, PROMPTS


client = QdrantClient(path=CONFIG["QDRANT_PATH"])

dense_embedder = SentenceTransformer(CONFIG["DENSE_MODEL"], device="cuda")
sparse_embedder = SparseTextEmbedding(CONFIG["SPARSE_MODEL"], device="cuda")
reranker = CrossEncoder(CONFIG["RERANKER_MODEL"], device="cuda")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = CONFIG["MODEL_ID"],
    max_seq_length = CONFIG["GENERATING_CONFIG"]["max_seq_length"],
    dtype = None,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

def search_law(query: str, top_k: int = CONFIG["RAG"]["TOP_K"], fetch_k: int = CONFIG["RAG"]["FETCH_K"]) -> List[Dict]:
    """
    Wyszukuje przepisy prawne w bazie Qdrant na podstawie zapytania użytkownika.
    Używa hybrydowego podejścia z wektorami gęstymi i rzadkimi oraz rerankingu.
    """

    dense_vec = dense_embedder.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()
    sparse_res = list(sparse_embedder.embed([query]))[0]
    
    qdrant_sparse = models.SparseVector(
        indices=sparse_res.indices.tolist(), 
        values=sparse_res.values.tolist()
    )

    collection = CONFIG["SEARCHING_COLLECTION"]
    initial_hits = []

    if client.collection_exists(collection):
        hits = client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vec, 
                    using="dense", 
                    limit=fetch_k
                ),
                models.Prefetch(
                    query=qdrant_sparse, 
                    using="sparse", 
                    limit=fetch_k
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=fetch_k
        ).points
        initial_hits = hits
        
        cross_inp = [[query, hit.payload.get('text', '')] for hit in initial_hits]
        cross_scores = reranker.predict(cross_inp)

        for idx, hit in enumerate(initial_hits):
            hit.score = cross_scores[idx]
        
        reranked_hits = sorted(initial_hits, key=lambda x: x.score, reverse=True)
        final_hits = reranked_hits[:top_k]
        
        return final_hits
        
    return []


def rewrite_query(user_query: str, chat_history: List[Dict]) -> str:
    """
    Inteligentnie przepisuje zapytania.
    1. Sprawdza rerankerem czy temat jest kontynuowany.
    2. Jeśli nie -> zwraca oryginalne pytanie.
    3. Jeśli tak -> używa LLM do przepisania.
    """

    if not chat_history:
        return user_query

    last_user_msg = next((m['content'] for m in reversed(chat_history) if m['role'] == 'user'), None)
    
    if last_user_msg:
        
        scores = reranker.predict([[last_user_msg, user_query]])
        score = scores[0]
        
        if score <= CONFIG["RAG"]["RERANKING_THRESHOLD"]: 
            return user_query

    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-4:]])

    rewrite_prompt = PROMPTS["REWRITING_PROMPT"].format(
        short_history=history_text,
        user_query=user_query
    )

    messages = [{"role": "user", "content": rewrite_prompt}]
    
    
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs_tensor = tokenizer(inputs, return_tensors="pt").to("cuda")
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs_tensor, 
            max_new_tokens=CONFIG["REWRITING_CONFIG"]["max_new_tokens"],
            temperature=CONFIG["REWRITING_CONFIG"]["temperature"],  
            do_sample=True,  
            use_cache=True
        )
    
    rewritten = tokenizer.decode(outputs[0][inputs_tensor.input_ids.shape[1]:], skip_special_tokens=True).strip()

    cleaned = rewritten.replace('"', '').replace("PRECYZYJNE ZAPYTANIE:", "").strip()
    
    if len(cleaned) < 3: 
        return user_query
        
    return cleaned


def client_close():
    """Zamknięcie klienta Qdrant."""
    client.close()