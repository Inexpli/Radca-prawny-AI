import os
import json
import streamlit as st
from typing import List, Dict, Tuple


st.set_page_config(
    page_title="Radca Prawny AI",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Radca Prawny AI")
st.markdown("Twój prywatny asystent prawny.")

loading_placeholder = st.empty()
loading_placeholder.info("🚀 Inicjalizacja systemu... \n\n 🛠️ Ładowanie bibliotek AI (to może chwilę potrwać)...")

HISTORY_FILE = "chat_history.json"

def load_chat_history() -> List[Dict]:
    """Wczytuje historię z pliku JSON jeśli istnieje."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat_history(messages) -> None:
    """Zapisuje historię do pliku JSON."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Błąd zapisu historii: {e}")

@st.cache_resource
def load_resources() -> Tuple:
    """Ładuje zasoby AI: Qdrant, embeddery i model językowy."""
    print("LOG: Importowanie bibliotek...")

    from unsloth import FastLanguageModel
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    from fastembed import SparseTextEmbedding

    MODEL_ID = "speakleash/Bielik-11B-v2.6-Instruct"
    QDRANT_PATH = "./qdrant_data"
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
    SPARSE_MODEL = "Qdrant/bm25"

    client = QdrantClient(path=QDRANT_PATH)

    dense = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
    sparse = SparseTextEmbedding(model_name=SPARSE_MODEL)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_ID,
        max_seq_length = 8192,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    return client, dense, sparse, model, tokenizer

try:
    client, dense_embedder, sparse_embedder, model, tokenizer = load_resources()
    
    loading_placeholder.empty()
    st.toast("System gotowy do pracy!", icon="✅")
except Exception as e:
    st.error(f"Błąd krytyczny podczas ładowania: {e}")
    st.stop()

import torch
from qdrant_client import models

SEARCH_COLLECTIONS = ["polskie_prawo"]

def rewrite_query(user_query, chat_history) -> str:
    """
    Inteligentnie przepisuje krótkie pytania na pełne 
    zapytania do bazy, wykorzystując historię rozmowy.
    """
    if not chat_history:
        return user_query
    
    short_history = chat_history[-4:] 
    
    rewrite_prompt = f"""
    Twoim zadaniem jest przeredagowanie ostatniego pytania użytkownika tak, aby było w pełni zrozumiałe bez znajomości poprzednich wiadomości.
    Musisz dodać brakujący kontekst (np. o czym była mowa wcześniej).

    HISTORIA ROZMOWY:
    {short_history}

    OSTATNIE KRÓTKIE PYTANIE: "{user_query}"

    ZASADA: Nie odpowiadaj na pytanie. Tylko je przepisz na pełne zdanie, które mogę wpisać w Google.

    PEŁNE PYTANIE:
    """

    messages = [{"role": "user", "content": rewrite_prompt}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs_tensor = tokenizer(inputs, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs_tensor, 
            max_new_tokens=128,
            temperature=0.2,  
            do_sample=True,  
            use_cache=True
        )
    
    rewritten = tokenizer.decode(outputs[0][inputs_tensor.input_ids.shape[1]:], skip_special_tokens=True).strip()
    cleaned = rewritten.replace('"', '').replace("PEŁNE PYTANIE:", "").strip()

    if len(cleaned) < 3: 
        return user_query

    return cleaned

def search_law(query: str, top_k: int = 5) -> List[Dict]:
    """
    Szuka w każdej kolekcji, łączy wyniki i zwraca X najlepszych globalnie.
    """
    dense_vec = dense_embedder.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()
    sparse_res = list(sparse_embedder.embed([query]))[0]
    
    qdrant_sparse = models.SparseVector(
        indices=sparse_res.indices.tolist(), 
        values=sparse_res.values.tolist()
    )

    all_hits = []
    for collection in SEARCH_COLLECTIONS:
        if client.collection_exists(collection):
            hits = client.query_points(
                collection_name=collection,
                prefetch=[
                    models.Prefetch(query=dense_vec, using="dense", limit=20),
                    models.Prefetch(query=qdrant_sparse, using="sparse", limit=20),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k
            ).points
            all_hits.extend(hits)
            
    return all_hits[:top_k]

with st.sidebar:
    st.title("⚙️ Ustawienia")
    if st.button("Wyczyść historię"):
        st.session_state.messages = []
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.rerun()
        
    st.info("Status: Online 🟢\n\nTryb: Persisted (Dysk)")

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("O co chcesz zapytać?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_history(st.session_state.messages)

    candidates = [] 
    context_text = ""

    with st.status("Analizuję przepisy...", expanded=True) as status:
        st.write("🔄 Analizuję pytanie...")
        search_query = rewrite_query(prompt, st.session_state.messages[:-1])
        
        st.write("🔍 Przeszukuję Kodeksy...")
        hits = search_law(search_query, top_k=5)
        
        if hits:
            for hit in hits:
                meta = hit.payload
                source_label = meta.get('source', 'Akt Prawny')
                article_label = meta.get('article', 'Art. ?')
                text_content = meta.get('full_markdown', meta.get('text', ''))
                
                context_text += f"=== {source_label} | {article_label} ===\n{text_content}\n\n"
                
                candidates.append({
                    "full_label": f"{article_label} ({source_label})",
                    "article_id": article_label
                })
        else:
            context_text = "Brak przepisów."
            
        status.update(label="Analiza zakończona!", state="complete", expanded=False)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Piszę opinię prawną..."):
            system_prompt = """Jesteś ekspertem od polskiego prawa. Twoim zadaniem jest interpretacja przepisów i udzielenie profesjonalnej porady.
            Działasz w oparciu o dostarczony KONTEKST PRAWNY, który może zawierać różne kodeksy (Karny, Cywilny, Pracy, Wykroczeń) oraz Konstytucję.

            ZASADY:
            1. Hierarchia: Konstytucja > Ustawy (Kodeksy). Jeśli problem dotyczy praw podstawowych, zacznij od Konstytucji.
            2. Kontekst: Używaj tylko przepisów dostarczonych w sekcji KONTEKST.
            3. Precyzja: Odpowiedź musi być konkretna. Jeśli pytanie dotyczy pracy, skup się na Kodeksie Pracy. Jeśli przestępstwa - na Karnym.
            4. Struktura odpowiedzi:
            - Podstawa Prawna (wymień artykuły i nazwy aktów)
            - Analiza (interpretacja sytuacji w świetle przepisów)
            - Konkluzja (jasne wnioski dla klienta)

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

            Pamiętaj: Twoim priorytetem jest poprawność merytoryczna oraz ścisłe trzymanie się formatu "Art. ... § ... Kodeksu ...:".
            """
            
            messages_payload = [{"role": "system", "content": system_prompt}]
            for msg in st.session_state.messages[:-1]:
                messages_payload.append(msg)
                
            current_input = f"KONTEKST PRAWNY:\n{context_text}\n\nPYTANIE:\n{prompt}"
            messages_payload.append({"role": "user", "content": current_input})

            model_inputs = tokenizer.apply_chat_template(messages_payload, tokenize=False, add_generation_prompt=True)
            inputs_tensor = tokenizer(model_inputs, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs_tensor,
                    max_new_tokens=2048,
                    temperature=0.2,
                    repetition_penalty=1.05,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs_tensor.input_ids.shape[1]:], skip_special_tokens=True)
    
            final_sources = []
            for candidate in candidates:
                if candidate["article_id"] in response:
                    final_sources.append(candidate["full_label"])

            if final_sources:
                footer = "\n\n---\n📚 **Źródła:** " + ", ".join(final_sources)
                final_response = response + footer
            else:
                final_response = response

        message_placeholder.markdown(final_response)

    st.session_state.messages.append({"role": "assistant", "content": final_response})
    save_chat_history(st.session_state.messages)