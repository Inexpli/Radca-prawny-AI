import os
import glob
import json
import uuid
import streamlit as st
from datetime import datetime
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

SESSIONS_DIR = "sessions"
SEARCH_COLLECTION = "polskie_prawo"

if not os.path.exists(SESSIONS_DIR):
    os.makedirs(SESSIONS_DIR)

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


def get_session_file_path(session_id):
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")

def save_current_session():
    """Zapisuje bieżącą sesję do pliku JSON."""
    if not st.session_state.messages:
        return
    
    if "title" not in st.session_state:
        user_msgs = [m['content'] for m in st.session_state.messages if m['role'] == 'user']
        if user_msgs:
            question = user_msgs[0]
            st.session_state.title = name_session(question)

    data = {
        "id": st.session_state.session_id,
        "title": st.session_state.get("title", "Bez tytułu"),
        "timestamp": datetime.now().isoformat(),
        "messages": st.session_state.messages
    }
    
    file_path = get_session_file_path(st.session_state.session_id)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_session_by_id(session_id):
    """Wczytuje sesję z pliku."""
    file_path = get_session_file_path(session_id)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            st.session_state.session_id = data["id"]
            st.session_state.messages = data["messages"]
            st.session_state.title = data.get("title", "Bez tytułu")
    else:
        init_new_session()

def init_new_session():
    """Resetuje stan do nowej, czystej rozmowy."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    if "title" in st.session_state:
        del st.session_state.title

def list_past_sessions():
    """Zwraca listę dostępnych plików sesji posortowaną od najnowszej."""
    files = glob.glob(os.path.join(SESSIONS_DIR, "*.json"))
    sessions = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                sessions.append({
                    "id": data["id"],
                    "title": data.get("title", "Bez tytułu"),
                    "path": f,
                    "time": os.path.getmtime(f)
                })
        except:
            continue
    return sorted(sessions, key=lambda x: x["time"], reverse=True)

if "session_id" not in st.session_state:
    init_new_session()

def name_session(question: str) -> str:
    """Nazywa sesję na podstawie pierwszego pytania użytkownika."""
    prompt = f"""
    Jesteś asystentem, który tworzy zwięzłe tytuły dla rozmów na podstawie pierwszego pytania użytkownika.
    ZASADY:
    1. Tytuł musi być krótki (maksymalnie 5 słów).
    2. Tytuł musi być precyzyjny i odzwierciedlać temat pytania.
    3. Unikaj ogólnych fraz jak "Rozmowa z AI" czy "Pytanie prawne".
    4. Używaj języka polskiego.
    5. Wypisz tytuł w formie, którą mogę wpisać w Google.
    PIERWSZE PYTANIE: "{question}"
    TYTUŁ ROZMOWY:
    """
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs_tensor = tokenizer(inputs, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs_tensor, 
            max_new_tokens=32,
            temperature=0.3,  
            do_sample=True,  
            use_cache=True
        )
    
    title = tokenizer.decode(outputs[0][inputs_tensor.input_ids.shape[1]:], skip_special_tokens=True).strip()
    cleaned_title = title.replace('"', '').strip()

    if len(cleaned_title) < 3: 
        return "Nowa rozmowa"

    return cleaned_title

def rewrite_query(user_query, chat_history) -> str:
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

def search_law(query: str, top_k: int = 10, score_threshold: float = 0.6) -> List[Dict]:
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

    if client.collection_exists(SEARCH_COLLECTION):
        hits = client.query_points(
            collection_name=SEARCH_COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=30,
                ),
                models.Prefetch(
                    query=qdrant_sparse,
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

with st.sidebar:
    st.title("Historia")
    
    if st.button("Nowy czat", use_container_width=True, type="secondary"):
        save_current_session()
        init_new_session()
        st.rerun()
    
    st.markdown("---")
    st.caption("Poprzednie rozmowy:")
    
    sessions = list_past_sessions()
    for s in sessions:
        if st.button(s["title"], key=s["id"], use_container_width=True):
            save_current_session()
            load_session_by_id(s["id"]) 
            st.rerun()

    st.markdown("---")
    st.info("Status: Online 🟢  \nTryb: Persisted (Dysk)")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("O co chcesz zapytać?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    save_current_session()

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
        else:
            context_text = "Brak przepisów."
            
        status.update(label="Analiza zakończona!", state="complete", expanded=False)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Piszę opinię prawną..."):
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
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs_tensor.input_ids.shape[1]:], skip_special_tokens=True)

        message_placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    save_current_session()
    if len(st.session_state.messages) == 2:
        st.rerun()