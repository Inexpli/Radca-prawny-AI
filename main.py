from rich.rule import Rule
from rich.text import Text
from rich.panel import Panel
from rich.live import Live
from threading import Thread
from typing import List, Dict
from rich.console import Console
from rich.markdown import Markdown
from unsloth import FastLanguageModel
from fastembed import SparseTextEmbedding
from transformers import TextIteratorStreamer
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import CONFIG, PROMPTS
from core import search_law, rewrite_query, client_close
import torch

console = Console()

print(Text("\n\n"))
console.print(Rule("Uruchamianie radcy prawnego AI na bazie Bielik-11B-v2.6-Instruct", style="bold blue"))

print("1. Ładowanie Qdrant...")

print(f"2. Ładowanie modeli embeddingowych {CONFIG['MODEL_ID']}...")

print(f"3. Inicjalizacja tokenizera...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = CONFIG["MODEL_ID"],
    max_seq_length = CONFIG["GENERATING_CONFIG"]["max_seq_length"],
    dtype = None,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

def generate_advice(user_query: str, chat_history: List[Dict]) -> str:
    """
    Generuje opinię prawną na podstawie zapytania użytkownika i historii rozmowy.
    Obsługuje streaming z ramką (Live Panel).
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

    console.print("[dim]--- Generowanie opinii... ---[/dim]")

    
    stream_iterator = None

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs_tensor = tokenizer(prompt, return_tensors="pt").to("cuda")

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
    
    stream_iterator = streamer
    
    live_panel = Panel(Markdown("..."), title="Opinia Prawna", border_style="cyan", expand=False)
    full_response = ""

    with Live(live_panel, console=console, refresh_per_second=12) as live:
        for new_text in stream_iterator:
            full_response += new_text
            
            clean_response = full_response.replace("<|im_end|>", "")
            
            md_content = Markdown(clean_response)
            live.update(Panel(md_content, title="Opinia Prawna", border_style="cyan", expand=False))

    return full_response.replace("<|im_end|>", "")


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
            
            history.append({"role": "user", "content": q}) 
            history.append({"role": "assistant", "content": advice})
            
            if len(history) > MAX_HISTORY:
                history = history[-MAX_HISTORY:]

        except KeyboardInterrupt:
            console.print("\n[red]Przerwano przez użytkownika.[/red]")
            break
        except Exception:
            console.print_exception(show_locals=False)

    client_close()