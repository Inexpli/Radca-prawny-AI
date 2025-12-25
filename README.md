# ⚖️ Radca Prawny AI

**Radca Prawny AI** to zaawansowany system **RAG (Retrieval-Augmented Generation)** zaprojektowany do udzielania porad prawnych w oparciu o polskie ustawodawstwo. Projekt działa w 100% lokalnie, wykorzystując moc obliczeniową karty graficznej, co gwarantuje pełną prywatność danych.

System łączy **Wyszukiwanie Hybrydowe** (Semantyczne + Słowa Kluczowe) z potężnym polskim modelem językowym (**Bielik-11B**), aby dostarczać precyzyjne odpowiedzi sformatowane jak profesjonalne opinie prawne.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Model](https://img.shields.io/badge/LLM-Bielik--11B-orange)
![DB](https://img.shields.io/badge/VectorDB-Qdrant-red)
![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

## 🚀 Możliwości

* **100% Offline & Private:** Żadne dane nie wychodzą poza Twoją maszynę. Idealne do analizy wrażliwych spraw.
* **Hybrid Search (RRF):** System wykorzystuje jednocześnie wektory gęste (rozumienie kontekstu) oraz rzadkie (BM25 - precyzyjne słowa kluczowe), łącząc wyniki algorytmem Reciprocal Rank Fusion.
* **Multi-Code Retrieval:** Przeszukuje jednocześnie wiele aktów prawnych (Kodeksy: Karny, Cywilny, Pracy, Wykroczeń, Rodzinny oraz Konstytucję RP).
* **Context Awareness:** Dzięki mechanizmowi przepisywania zapytań (Query Rewriting), model rozumie kontekst rozmowy (np. pytania nawiązujące do poprzednich odpowiedzi).
* **Profesjonalny Format:** Odpowiedzi są generowane w ustrukturyzowanej formie (Podstawa Prawna -> Analiza -> Konkluzja).
* **Brak Halucynacji Prawnych:** Model bazuje wyłącznie na dostarczonym kontekście (RAG) i cytuje konkretne źródła.

## 🛠️ Stack Technologiczny

* **LLM:** `speakleash/Bielik-11B-v2.6-Instruct` (Kwantyzacja 4-bit NF4).
* **Embeddings (Dense):** `intfloat/multilingual-e5-large`.
* **Embeddings (Sparse):** `Qdrant/bm25` (via FastEmbed).
* **Vector Database:** `Qdrant` (Tryb lokalny/embedded).
* **Ingestion:** `Docling` (Konwersja PDF do Markdown).
* **UI:**
    * **Web:** `Streamlit` (Interaktywny czat z historią i renderowaniem Markdown).
    * **Terminal:** `Rich` (CLI).
* **Engine:** `Unsloth` (Inference optimization) + `BitsAndBytes`.

## 📚 Baza Wiedzy

Projekt automatycznie pobiera, przetwarza i indeksuje następujące akty prawne (aktualne wersje z ISAP):
* Konstytucja Rzeczypospolitej Polskiej
* Kodeks Karny (KK)
* Kodeks Cywilny (KC)
* Kodeks Pracy (KP)
* Kodeks Rodzinny i Opiekuńczy (KRO)
* Kodeks Wykroczeń (KW)

## ⚙️ Instalacja

### Wymagania
* System: Linux (zalecane) lub Windows (WSL2).
* GPU: NVIDIA z min. 16 GB VRAM (zalecane 24 GB dla pełnej wydajności).
* RAM: 16 GB+.
* Python: 3.10+.

### Kroki

1.  **Sklonuj repozytorium:**
    ```bash
    git clone https://github.com/Inexpli/Radca-prawny-AI
    cd Radca-prawny-AI
    ```

2.  **Utwórz wirtualne środowisko i zainstaluj zależności:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/WSL
    # .venv\Scripts\activate   # Windows PowerShell
    
    pip install -r requirements.txt
    ```
    *(Upewnij się, że masz zainstalowany PyTorch z obsługą CUDA)*

3.  **Zbuduj bazę wiedzy (Ingest):**
    Skrypt pobierze PDF-y, przekonwertuje je na Markdown, wygeneruje wektory hybrydowe i zapisze w Qdrant.
    ```bash
    python ingest_data.py
    ```

## ▶️ Użycie

Możesz korzystać z systemu na dwa sposoby.

### 1. Interfejs Graficzny (Rekomendowane)
Uruchamia nowoczesną aplikację w przeglądarce z historią czatu i formatowaniem tekstu.

```bash
streamlit run app.py
```

![alt text](docs/image0.png)


### 2. Wersja CLI
Klasyczny terminal dla szybkiego testowania i debugowania.

```bash
python main.py
```

![alt text](docs/image3.png)
![alt text](docs/image4.png)

## 📄 Licencja
- MIT License
