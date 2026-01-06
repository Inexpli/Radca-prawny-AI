import os
import glob
import json
import uuid
import streamlit as st
from datetime import datetime
from typing import Tuple
from config import CONFIG, CSS, PROMPTS


st.set_page_config(
    page_title="Radca Prawny AI",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Radca Prawny AI")
st.markdown("Twój prywatny asystent prawny.")

loading_placeholder = st.empty()
loading_placeholder.info("🚀 Inicjalizacja systemu... \n\n 🛠️ Ładowanie bibliotek AI (to może chwilę potrwać)...")

if not os.path.exists(CONFIG["SESSIONS_DIR"]):
    os.makedirs(CONFIG["SESSIONS_DIR"])

@st.cache_resource
def load_resources() -> Tuple:
    """Ładuje zasoby AI: Qdrant, embeddery i model językowy."""
    print("LOG: Importowanie bibliotek...")
    from core import model, tokenizer

    return model, tokenizer

try:
    model, tokenizer = load_resources()

    loading_placeholder.empty()
    st.toast("System gotowy do pracy!", icon="✅")
except Exception as e:
    st.error(f"Błąd krytyczny podczas ładowania: {e}")
    st.stop()

import torch
from core import search_law, rewrite_query
from transformers import TextIteratorStreamer
from threading import Thread

def get_session_file_path(session_id: str) -> str:
    """Zwraca ścieżkę do pliku sesji na podstawie ID sesji."""
    return os.path.join(CONFIG["SESSIONS_DIR"], f"{session_id}.json")

def save_current_session() -> None:
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

def delete_session_by_id(session_id: str) -> None:
    """Usuwa plik sesji o podanym ID."""
    file_path = get_session_file_path(session_id)
    if os.path.exists(file_path):
        os.remove(file_path)

def load_session_by_id(session_id: str) -> None:
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

def init_new_session() -> None:
    """Resetuje stan do nowej, czystej rozmowy."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    if "title" in st.session_state:
        del st.session_state.title

def list_past_sessions() -> list:
    """Zwraca listę dostępnych plików sesji posortowaną od najnowszej."""
    files = glob.glob(os.path.join(CONFIG["SESSIONS_DIR"], "*.json"))
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
    prompt = PROMPTS["NAMING_SESSION_PROMPT"].format(question=question)

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs_tensor = tokenizer(inputs, return_tensors="pt").to("cuda")
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs_tensor, 
            max_new_tokens=CONFIG["NAME_SESSION_CONFIG"]["max_new_tokens"],
            temperature=CONFIG["NAME_SESSION_CONFIG"]["temperature"],  
            do_sample=True,  
            use_cache=True
        )
    
    title = tokenizer.decode(outputs[0][inputs_tensor.input_ids.shape[1]:], skip_special_tokens=True).strip()
    cleaned_title = title.replace('"', '').strip()

    if len(cleaned_title) < 3: 
        return "Nowa rozmowa"

    return cleaned_title

with st.sidebar:
    if st.button("Nowy czat", use_container_width=True, type="secondary"):
        save_current_session()
        init_new_session()
        st.rerun()

    st.title("Historia")
    
    sessions = list_past_sessions()
    
    def handle_delete(session_id):
        """Obsługuje usuwanie sesji."""
        delete_session_by_id(session_id)
        if st.session_state.session_id == session_id:
            init_new_session()
        st.toast(f"Usunięto rozmowę.", icon="🗑️")

    st.caption("Poprzednie rozmowy:")

    st.markdown(
        CSS, unsafe_allow_html=True
    )

    for s in sessions:
        is_active = s["id"] == st.session_state.session_id
        
        col1, col2 = st.columns([0.85, 0.15])
        
        with col1:
            display_title = s["title"] if len(s["title"]) < 28 else s["title"][:25] + "..."
            
            btn_type = "primary" if is_active else "secondary"
            
            if st.button(
                display_title, 
                key=f"load_{s['id']}", 
                use_container_width=True, 
                type=btn_type,
                help=s["title"]
            ):
                if not is_active:
                    save_current_session()
                    load_session_by_id(s["id"])
                    st.rerun()
        
        with col2:
            if st.button("✕", key=f"del_{s['id']}", type="tertiary", help="Usuń trwale tę rozmowę"):
                handle_delete(s["id"])
                st.rerun()

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

        hits = search_law(search_query)
        
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
        full_response = ""

        if not hits or context_text.strip() == "Brak przepisów.":
            full_response = """
            **Brak danych w bazie.** \n\nNie znalazłem w dostępnych kodeksach przepisów pasujących precyzyjnie do Twojego zapytania. Jako system RAG nie mogę generować porad prawnych z pamięci, aby uniknąć wprowadzenia Cię w błąd nieaktualnymi przepisami.
            """
            message_placeholder.markdown(full_response)
            response = full_response
        
        else:
            with st.spinner("Piszę opinię prawną..."):
                system_prompt = PROMPTS["SYSTEM_PROMPT"]
                
                messages_payload = [{"role": "system", "content": system_prompt}]
                for msg in st.session_state.messages[:-1]:
                    messages_payload.append(msg)
                    
                current_input = f"KONTEKST PRAWNY:\n{context_text}\n\nPYTANIE:\n{prompt}"
                messages_payload.append({"role": "user", "content": current_input})

                model_inputs = tokenizer.apply_chat_template(messages_payload, tokenize=False, add_generation_prompt=True)
                inputs_tensor = tokenizer(model_inputs, return_tensors="pt").to("cuda")

                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens": True})

                generation_kwargs = dict(
                    input_ids=inputs_tensor.input_ids,
                    attention_mask=inputs_tensor.attention_mask,
                    streamer=streamer,
                    max_new_tokens=CONFIG["GENERATING_CONFIG"]["max_new_tokens"],
                    temperature=CONFIG["GENERATING_CONFIG"]["temperature"],
                    repetition_penalty=CONFIG["GENERATING_CONFIG"]["repetition_penalty"],
                    do_sample=True,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id
                )

                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                
                for new_text in streamer:
                    clean_text = new_text.replace("<|im_end|>", "")
                    full_response += clean_text
                    message_placeholder.markdown(full_response + "▌")
                
                thread.join()

            message_placeholder.markdown(full_response)
            response = full_response

    st.session_state.messages.append({"role": "assistant", "content": response})
    save_current_session()
    
    if len(st.session_state.messages) == 2:
        st.rerun()