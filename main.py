import torch
from rich.rule import Rule
from rich.text import Text
from rich.panel import Panel
from typing import List, Dict
from rich.console import Console
from rich.markdown import Markdown
from unsloth import FastLanguageModel
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

console = Console()

MODEL_ID = "speakleash/Bielik-11B-v2.6-Instruct"
QDRANT_PATH = "./qdrant_data"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
SPARSE_MODEL = "Qdrant/bm25"
SEARCH_COLLECTION = "polskie_prawo"

print(Text("\n\n"))
console.print(Rule("Uruchamianie radcy prawnego AI na bazie Bielik-11B-v2.6-Instruct", style="bold blue"))

print("1. Ładowanie Qdrant i modeli embeddingowych...")
client = QdrantClient(path=QDRANT_PATH)
dense_embedder = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL)

print(f"2. Ładowanie modelu {MODEL_ID}...")

print(f"3. Inicjalizacja tokenizera...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = 32768,
    dtype = None,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

def search_law(query: str, top_k: int = 10, score_threshold: float = 0.6) -> List[Dict]:
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
                    limit=30,
                ),
                models.Prefetch(
                    query=qdrant_sparse_vector,
                    using="sparse",
                    limit=30,
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k
        ).points

        valid_hits = [hit for hit in hits if hit.score > score_threshold]
        all_hits.extend(valid_hits)

    return all_hits[:top_k]

def rewrite_query(user_query: str, chat_history: List[Dict]) -> str:
    """
    Inteligentnie przepisuje krótkie pytania na pełne 
    zapytania do bazy, wykorzystując historię rozmowy.
    """
    if not chat_history:
        return user_query

    short_history = chat_history[-4:] 
    
    rewrite_prompt = f"""
    Jesteś prawnikiem-lingwistą. Twoim zadaniem jest przetłumaczenie potocznego pytania klienta na profesjonalne zapytanie do wyszukiwarki prawniczej.
    ZASADY:
    1. Zamień słowa potoczne na ustawowe (np. "morderstwo" -> "zabójstwo", "ukradł auto" -> "zabór pojazdu mechanicznego").
    2. Uwzględnij kontekst z historii rozmowy (jeśli jest).
    3. Wynik ma być jednym, precyzyjnym zdaniem pytającym.

    HISTORIA: {short_history}
    OSTATNIE PYTANIE: "{user_query}"

    PROFESJONALNE ZAPYTANIE:
    """

    messages = [{"role": "user", "content": rewrite_prompt}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs_tensor = tokenizer(inputs, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs_tensor, 
            max_new_tokens=128,
            temperature=0.1,  
            do_sample=True,  
            use_cache=True
        )
    
    rewritten = tokenizer.decode(outputs[0][inputs_tensor.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    cleaned = rewritten.replace('"', '').replace("PROFESJONALNE ZAPYTANIE:", "").strip()
    
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
    hits = search_law(search_query, top_k=5)

    context_text = ""
    for hit in hits:
        meta = hit.payload
        source_label = meta.get('source', 'Akt Prawny')
        article_label = meta.get('article', 'Art. ?')
        text_content = meta.get('full_markdown', meta.get('text', ''))
        
        context_text += f"=== {source_label} | {article_label} ===\n{text_content}\n\n"
        
    if not context_text:
        context_text = "Brak bezpośrednich przepisów w bazie dla tego zapytania."

    system_prompt = """
    Jesteś ekspertem od polskiego prawa. Twoim zadaniem jest interpretacja przepisów i udzielenie profesjonalnej porady.
    Działasz w oparciu o dostarczony KONTEKST PRAWNY, który może zawierać różne kodeksy (Karny, Cywilny, Pracy, Wykroczeń) oraz Konstytucję.

    ZASADY:
    1. Hierarchia: Konstytucja > Ustawy (Kodeksy). Jeśli problem dotyczy praw podstawowych, zacznij od Konstytucji.
    2. Kontekst: Używaj tylko przepisów dostarczonych w sekcji KONTEKST.
    3. Precyzja: Odpowiedź musi być konkretna. Jeśli pytanie dotyczy pracy, skup się na Kodeksie Pracy. Jeśli przestępstwa - na Karnym.
    4. Struktura odpowiedzi:
    - Podstawa Prawna (wymień artykuły i nazwy aktów)
    - Analiza (interpretacja sytuacji w świetle przepisów)
    - Konkluzja (jasne wnioski dla klienta)
    5. Najważniejsze - jeśli brak przepisów w kontekście, przyznaj to otwarcie i zasugeruj konsultację z prawnikiem.
    6. Nie wymyślaj przepisów ani nie odwołuj się do nieistniejących artykułów.
    7. Nie naciągaj kontekstu - jeśli pytanie wykracza poza dostarczone przepisy, przyznaj to.

    RESTKRYKCYJNE ZASADY FORMATOWANIA (MODEL MUSI ICH PRZESTRZEGAĆ):
    Każda odpowiedź musi składać się wyłącznie z 4 sekcji oznaczonych nagłówkami H2 (##). Nie dodawaj żadnego tekstu przed pierwszą sekcją ani po ostatniej.

    STRUKTURA ODPOWIEDZI:

    ## Podstawa Prawna
    W tej sekcji wymień przepisy w formie listy wypunktowanej.
    BEZWZGLĘDNY FORMAT CYTOWANIA:
    * **Art. {numer} § {numer_paragrafu} {Pełna Nazwa Kodeksu}:** {treść przepisu}

    Zasady dla cytatów:
    - Jeśli przepis nie ma paragrafu, pomiń znak § i numer paragrafu (np. Art. 148 Kodeksu Karnego:).
    - Zawsze podawaj pełną nazwę kodeksu (np. "Kodeksu Karnego", a nie "k.k.").
    - Treść przepisu musi być przytoczona po dwukropku.

    ## Analiza
    Szczegółowa interpretacja sytuacji w świetle przytoczonych wyżej przepisów. Odnieś się bezpośrednio do faktów z zapytania użytkownika. Wyjaśnij przesłanki (np. "użycie przemocy", "stan nietrzeźwości"). Pisz akapitami.

    ## Konkluzja
    Jasne i zwięzłe wnioski. Jeśli wynik zależy od zmiennych (np. czy użyto broni), zastosuj listę wypunktowaną, aby pokazać warianty:
    * Wariant A: konsekwencja.
    * Wariant B: konsekwencja.

    ## Podsumowanie
    Jedno lub dwa zdania streszczenia dla klienta, stanowiące "tl;dr" całej porady. Najlepiej by było, gdyby zawierało bezpośrednią, konkretną odpowiedź na pytanie użytkownika.

    Na końcu odpowiedzi dołącz sekcję Źródła, gdzie w jednej linii wymienisz wszystkie cytowane artykuły w formacie:
    BEZWZGLĘDNY FORMAT WYPISYWANIA ŹRÓDEŁ:
    "\n\n---\n📚 **Źródła:** Art. {numer} {Pełna Nazwa Kodeksu}."
    Przykład:
    "\n\n---\n📚 **Źródła:** Art. 134, 135, 136, 148 Kodeksu Karnego."
    Nie zapisuj tego jako osobny nagłowek, tylko jako zwykły tekst od nowej linii oraz nie wypisuj paragrafów w źródłach.
    """

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    
    current_input = f"KONTEKST PRAWNY:\n{context_text}\n\nPYTANIE UŻYTKOWNIKA:\n{user_query}"
    messages.append({"role": "user", "content": current_input})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs_tensor = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    console.print("[dim]--- Generowanie opinii... ---[/dim]")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs_tensor, 
            max_new_tokens=2048,
            temperature=0.2,
            repetition_penalty=1.05,
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