import torch
from rich.rule import Rule
from rich.text import Text
from rich.panel import Panel
from typing import List, Dict
from rich.console import Console
from rich.markdown import Markdown
from unsloth import FastLanguageModel
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import CONFIG, PROMPTS

console = Console()

print(Text("\n\n"))
console.print(Rule("Uruchamianie radcy prawnego AI na bazie Bielik-11B-v2.6-Instruct", style="bold blue"))

print("1. Ładowanie Qdrant...")

client = QdrantClient(path=CONFIG["QDRANT_PATH"])

print(f"2. Ładowanie modeli embeddingowych {CONFIG['MODEL_ID']}...")

dense_embedder = SentenceTransformer(CONFIG["DENSE_MODEL"], device="cuda")
sparse_embedder = SparseTextEmbedding(CONFIG["SPARSE_MODEL"], device="cuda")
reranker = CrossEncoder(CONFIG["RERANKER_MODEL"], device="cuda")

print(f"3. Inicjalizacja tokenizera...")

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
                models.Prefetch(query=dense_vec, using="dense", limit=fetch_k),
                models.Prefetch(query=qdrant_sparse, using="sparse", limit=fetch_k)
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
        
        if score <= -4.0: 
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
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id
        )
    
    rewritten = tokenizer.decode(outputs[0][inputs_tensor.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    del inputs_tensor
    del outputs

    cleaned = rewritten.replace('"', '').replace("PRECYZYJNE ZAPYTANIE:", "").strip()
    
    if len(cleaned) < 3: 
        return user_query
        
    return cleaned

def generate_advice(user_query: str, chat_history: List[Dict]) -> tuple[str, List[str]]:
    """
    Generuje opinię prawną na podstawie zapytania użytkownika i historii rozmowy.
    """
    
    if chat_history:
        console.print("[dim]--- Kontekstualizacja pytania... ---[/dim]")
        try:
            search_query = rewrite_query(user_query, chat_history)
            console.print(f"[dim]   Oryginał: [red]{user_query}[/red][/dim]")
            console.print(f"[dim]   Szukam:   [green]{search_query}[/green][/dim]")
        except Exception:
            console.print("[red]   Błąd przepisywania pytania. Używam oryginału.[/red]")
            search_query = user_query
    else:
        search_query = user_query

    console.print(f"\n[dim]--- Wyszukiwanie przepisów... ---[/dim]")
    hits = search_law(search_query)
    
    context_text = ""
    for hit in hits:
        meta = hit.payload
        source_label = meta.get('source', 'Akt Prawny')
        article_label = meta.get('article', 'Art. ?')
        text_content = meta.get('full_markdown', meta.get('text', ''))
        
        context_text += f"=== {source_label} | {article_label} ===\n{text_content}\n\n"

    if not context_text:
        context_text = "Brak bezpośrednich przepisów w bazie dla tego zapytania."

    system_prompt = PROMPTS["SYSTEM_PROMPT"]

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    
    current_input = f"KONTEKST PRAWNY:\n{context_text}\n\nPYTANIE UŻYTKOWNIKA:\n{user_query}"
    messages.append({"role": "user", "content": current_input})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs_tensor = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    console.print("[dim]--- Generowanie opinii... ---[/dim]")
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs_tensor, 
            max_new_tokens=CONFIG["GENERATING_CONFIG"]["max_new_tokens"],
            temperature=CONFIG["GENERATING_CONFIG"]["temperature"],
            repetition_penalty=CONFIG["GENERATING_CONFIG"]["repetition_penalty"],
            do_sample=True,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs_tensor.input_ids.shape[1]:], skip_special_tokens=True)
    
    return response


if __name__ == "__main__":
    
    history = [] 
    MAX_HISTORY = 4

    while True:
        try:
            q = console.input("\n[bold green]Podaj pytanie[/] (lub 'exit' / 'clear'): ")
            
            if q.lower() in ['exit', 'wyjdz']: 
                break
            
            if q.lower() in ['clear', 'reset']:
                history = []
                console.print("[yellow]Historia konwersacji została wyczyszczona.[/yellow]")
                continue

            if not q.strip(): 
                continue
            
            advice = generate_advice(q, history)
            
            md_content = Markdown(advice)
            console.print(Panel(md_content, title="Opinia Prawna", border_style="cyan", expand=False))
            
            history.append({"role": "user", "content": q}) 
            history.append({"role": "assistant", "content": advice})
            
            if len(history) > MAX_HISTORY:
                history = history[-MAX_HISTORY:]

        except KeyboardInterrupt:
            console.print("\n[red]Przerwano przez użytkownika.[/red]")
            break
        except Exception:
            console.print_exception(show_locals=False)

    client.close()