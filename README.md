# ⚖️ Radca prawny AI

**Radca prawny AI** to zaawansowany system **RAG (Retrieval-Augmented Generation)** zaprojektowany do udzielania porad prawnych w oparciu o polskie ustawodawstwo. Projekt działa w 100% lokalnie, wykorzystując moc obliczeniową karty graficznej, co gwarantuje pełną prywatność danych.

System łączy precyzyjne wyszukiwanie semantyczne (Qdrant) z potężnym polskim modelem językowym (**Bielik-11B**), aby dostarczać odpowiedzi sformatowane jak profesjonalne opinie prawne.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![GPU](https://img.shields.io/badge/GPU-RTX%203090-green)
![Model](https://img.shields.io/badge/LLM-Bielik--11B-orange)
![DB](https://img.shields.io/badge/VectorDB-Qdrant-red)

## 🚀 Możliwości

* **100% Offline & Private:** Żadne dane nie wychodzą poza Twoją maszynę. Idealne do analizy wrażliwych spraw.
* **Multi-Code Retrieval:** System przeszukuje jednocześnie wiele aktów prawnych (Kodeks Karny, Cywilny, Pracy, Wykroczeń, Rodzinny oraz Konstytucję RP).
* **Global Ranking:** Wyniki są sortowane po trafności niezależnie od źródła – system sam ocenia, czy sprawa ma charakter karny czy cywilny.
* **Profesjonalny Format:** Odpowiedzi są generowane w ustrukturyzowanej formie (Podstawa Prawna -> Analiza -> Konkluzja) z wykorzystaniem biblioteki `rich` (TUI).
* **Brak Halucynacji Prawnych:** Model ma surowy zakaz wymyślania przepisów – bazuje wyłącznie na dostarczonym kontekście (RAG).

## 🛠️ Stack Technologiczny

* **LLM:** `speakleash/Bielik-11B-v2.6-Instruct` (Kwantyzacja 4-bit NF4).
* **Embeddings:** `intfloat/multilingual-e5-large` (Model rozumiejący polski kontekst prawny).
* **Vector Database:** `Qdrant` (Tryb lokalny/embedded).
* **Ingestion:** `Docling` (Konwersja PDF do Markdown) + Custom Parsers.
* **UI:** `Rich` (CLI z formatowaniem Markdown i panelami).
* **Engine:** `Unsloth` + `BitsAndBytes`.

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
* GPU: NVIDIA z min. 16 GB VRAM.
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
    Skrypt pobierze PDF-y z sejm.gov.pl, przekonwertuje je na Markdown, podzieli na artykuły i zapisze w Qdrant.
    ```bash
    python ingest_data.py
    ```

## ▶️ Użycie

Uruchom interaktywnego agenta:
```bash
python main.py
```

![alt text](image1.png)
![alt text](image2.png)

