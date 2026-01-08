import torch
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from typing import List, Dict, Text
from transformers import TextIteratorStreamer
from rich.console import Console
from rich.rule import Rule
from config import CONFIG, PROMPTS
from threading import Thread


class LegalAdvisorAI:
    def __init__(self):
        self.console = Console()
        self.console.print(Text("\n\n"))
        self.console.print(Rule(f"Inicjalizacja Systemu Prawnego AI ({CONFIG['MODEL_ID']})", style="bold blue"))
        
        self._init_qdrant()
        self._init_models()
        
    def _init_qdrant(self):
        self.console.print("1. [dim]Ładowanie Qdrant...[/dim]")
        try:
            self.client = QdrantClient(path=CONFIG["QDRANT_PATH"])
            self.client.get_collections()
        except Exception as e:
            self.console.print(f"[bold red]Błąd połączenia z Qdrant: {e}[/bold red]")
            raise e

    def _init_models(self):
        self.console.print("2. [dim]Ładowanie modeli embeddingowych...[/dim]")
        self.dense_embedder = SentenceTransformer(CONFIG["DENSE_MODEL"], device="cuda")
        self.sparse_embedder = SparseTextEmbedding(CONFIG["SPARSE_MODEL"], device="cuda")
        self.reranker = CrossEncoder(CONFIG["RERANKER_MODEL"], device="cuda")

        self.console.print("3. [dim]Inicjalizacja modelu...[/dim]")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=CONFIG["MODEL_ID"],
            max_seq_length=CONFIG["GENERATING_CONFIG"]["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

    def search_law(self, query: str, top_k: int = CONFIG["RAG"]["TOP_K"], fetch_k: int = CONFIG["RAG"]["FETCH_K"]) -> List[Dict]:
        """Logika wyszukiwania hybrydowego (Dense + Sparse + Rerank)."""
        dense_vec = self.dense_embedder.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()
        sparse_res = list(self.sparse_embedder.embed([query]))[0]
        
        qdrant_sparse = models.SparseVector(
            indices=sparse_res.indices.tolist(), 
            values=sparse_res.values.tolist()
        )

        collection = CONFIG["SEARCHING_COLLECTION"]
        if not self.client.collection_exists(collection):
            self.console.print(f"[red]Kolekcja {collection} nie istnieje![/red]")
            return []

        hits = self.client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=fetch_k),
                models.Prefetch(query=qdrant_sparse, using="sparse", limit=fetch_k)
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=fetch_k
        ).points
        
        if not hits:
            return []

        cross_inp = [[query, hit.payload.get('text', '')] for hit in hits]
        cross_scores = self.reranker.predict(cross_inp)

        for idx, hit in enumerate(hits):
            hit.score = cross_scores[idx]
        
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:top_k]

    def rewrite_query(self, user_query: str, chat_history: List[Dict]) -> str:
        """Kontekstualizacja zapytania."""
        if not chat_history:
            return user_query

        last_user_msg = next((m['content'] for m in reversed(chat_history) if m['role'] == 'user'), None)
        
        if last_user_msg:
            score = self.reranker.predict([[last_user_msg, user_query]])[0]
            if score <= CONFIG["RAG"]["RERANKING_THRESHOLD"]:
                return user_query

        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-4:]])
        rewrite_prompt = PROMPTS["REWRITING_PROMPT"].format(short_history=history_text, user_query=user_query)
        
        messages = [{"role": "user", "content": rewrite_prompt}]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_tensor = self.tokenizer(inputs, return_tensors="pt").to("cuda")
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs_tensor, 
                max_new_tokens=CONFIG["REWRITING_CONFIG"]["max_new_tokens"],
                temperature=CONFIG["REWRITING_CONFIG"]["temperature"],
                do_sample=True,
                use_cache=True
            )
        
        rewritten = self.tokenizer.decode(outputs[0][inputs_tensor.input_ids.shape[1]:], skip_special_tokens=True).strip()
        cleaned = rewritten.replace('"', '').replace("PRECYZYJNE ZAPYTANIE:", "").strip()
        
        return cleaned if len(cleaned) > 2 else user_query
    
    def name_session(self, question: str) -> str:
        """Nazywa sesję na podstawie pierwszego pytania użytkownika."""
        prompt = PROMPTS["NAMING_SESSION_PROMPT"].format(question=question)

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_tensor = self.tokenizer(inputs, return_tensors="pt").to("cuda")
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs_tensor, 
                max_new_tokens=CONFIG["NAME_SESSION_CONFIG"]["max_new_tokens"],
                temperature=CONFIG["NAME_SESSION_CONFIG"]["temperature"],  
                do_sample=True,  
                use_cache=True
            )
        
        title = self.tokenizer.decode(outputs[0][inputs_tensor.input_ids.shape[1]:], skip_special_tokens=True).strip()
        cleaned_title = title.replace('"', '').strip()

        if len(cleaned_title) < 3: 
            return "Nowa rozmowa"

        return cleaned_title

    def generate_response(self, user_query: str, chat_history: List[Dict]) -> str:
        """Główna pętla generująca odpowiedź."""

        error = False   

        if chat_history:
            self.console.print("[dim]--- Kontekstualizacja pytania... ---[/dim]")
            try:
                search_query = self.rewrite_query(user_query, chat_history)
                if search_query != user_query:
                    self.console.print(f"[dim]   Oryginał: [red]{user_query}[/red][/dim]")
                    self.console.print(f"[dim]   Szukam:   [green]{search_query}[/green][/dim]")
            except Exception as e:
                self.console.print(f"[red]Błąd rewritingu: {e}. Używam oryginału.[/red]")
                search_query = user_query
        else:
            search_query = user_query

        self.console.print(f"\n[dim]--- Wyszukiwanie przepisów... ---[/dim]")
        hits = self.search_law(search_query)
        
        context_text = ""
        for hit in hits:
            meta = hit.payload
            source = meta.get('source', 'Akt Prawny')
            article = meta.get('article', 'Art. ?')
            text_content = meta.get('full_markdown', meta.get('text', ''))
            context_text += f"=== {source} | {article} ===\n{text_content}\n\n"
        
        if not context_text:
            return None, None, None

        messages = [{"role": "system", "content": PROMPTS["SYSTEM_PROMPT"]}]
        messages.extend(chat_history[-4:])
        full_input = f"KONTEKST PRAWNY:\n{context_text}\n\nPYTANIE UŻYTKOWNIKA:\n{user_query}"
        messages.append({"role": "user", "content": full_input})

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            decode_kwargs={"skip_special_tokens": True, "clean_up_tokenization_spaces": True}
        )
        
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=CONFIG["GENERATING_CONFIG"]["max_new_tokens"],
            temperature=CONFIG["GENERATING_CONFIG"]["temperature"],
            repetition_penalty=CONFIG["GENERATING_CONFIG"]["repetition_penalty"],
            do_sample=True,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id
        )

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        return gen_kwargs, streamer, thread

    def close(self):
        self.client.close()