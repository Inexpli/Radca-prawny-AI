import textwrap


MAIN_COLLECTION = "polskie_prawo"


DATA_SOURCES = [
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19970880553/U/D19970553Lj.pdf",
        "file_path": "data/rag/kodeks_karny.md",
        "collection_name": MAIN_COLLECTION,
        "source_label": "Kodeks Karny"
    },
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19640160093/U/D19640093Lj.pdf",
        "file_path": "data/rag/kodeks_cywilny.md",
        "collection_name": MAIN_COLLECTION,
        "source_label": "Kodeks Cywilny"
    },
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19740240141/U/D19740141Lj.pdf",
        "file_path": "data/rag/kodeks_pracy.md",
        "collection_name": MAIN_COLLECTION,
        "source_label": "Kodeks Pracy"
    },
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19640090059/U/D19640059Lj.pdf",
        "file_path": "data/rag/kodeks_rodzinny_i_opiekunczy.md",
        "collection_name": MAIN_COLLECTION,
        "source_label": "Kodeks Rodzinny i Opiekuńczy"
    },
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19710120114/U/D19710114Lj.pdf",
        "file_path": "data/rag/kodeks_wykroczen.md",
        "collection_name": MAIN_COLLECTION,
        "source_label": "Kodeks Wykroczeń"
    },
    {
        "url": "https://isap.sejm.gov.pl/isap.nsf/download.xsp/WDU19970780483/U/D19970483Lj.pdf",
        "file_path": "data/rag/konstytucja_rp.md",
        "collection_name": MAIN_COLLECTION,
        "source_label": "Konstytucja RP"
    }
]


CONFIG = {
    "SEARCHING_COLLECTION": MAIN_COLLECTION,
    "SESSIONS_DIR": "sessions",
    "MODEL_ID": "speakleash/Bielik-11B-v3.0-Instruct",
    "QDRANT_PATH": "./qdrant_data",
    "DENSE_MODEL": "intfloat/multilingual-e5-large",
    "SPARSE_MODEL": "Qdrant/bm25",
    "RERANKER_MODEL": "sdadas/polish-reranker-roberta-v3",

    "RAG": {
        "TOP_K": 12,
        "FETCH_K": 100,
        "RERANKING_THRESHOLD": -4.0,
    },

    "NAME_SESSION_CONFIG": {
        "max_new_tokens": 32,
        "temperature": 0.3,
    },

    "REWRITING_CONFIG": {
        "max_new_tokens": 128,
        "temperature": 0.1,
    },

    "GENERATING_CONFIG": {
        "max_seq_length": 8192,
        "max_new_tokens": 1700,
        "temperature": 0.1,
        "repetition_penalty": 1.05,
    },
}


PROMPTS = {
    "NAMING_SESSION_PROMPT": 
    textwrap.dedent("""
        Jesteś prawnym asystentem, który tworzy zwięzłe tytuły dla rozmów na podstawie pierwszego pytania użytkownika.
        ZASADY:
        1. Tytuł musi być krótki (maksymalnie 5 słów).
        2. Tytuł musi być precyzyjny i odzwierciedlać temat pytania.
        3. Tytuł musi być zapisany prawnym językiem.
        4. Unikaj ogólnych fraz jak "Rozmowa z AI" czy "Pytanie prawne".
        5. Wypisz tytuł w formie, którą mogę wpisać w Google.
        PIERWSZE PYTANIE: "{question}"
        TYTUŁ ROZMOWY:
    """).strip(),

    "REWRITING_PROMPT": 
    textwrap.dedent("""
        Jesteś analitykiem prawnym. Twoim zadaniem jest sformułowanie precyzyjnego pytania do wyszukiwarki na podstawie wpisu użytkownika.

        ZASADY ANALIZY KONTEKSTU:
        1. Przeanalizuj OSTATNIE PYTANIE pod kątem powiązania z HISTORIĄ.
        2. JEŚLI pytanie jest kontynuacją (np. "a co jeśli...", "ile za to grozi?", zaimki "on/ona/to"): POŁĄCZ fakty z historii z nowym pytaniem.
        3. JEŚLI pytanie jest zmianą tematu (nowy wątek, niezwiązany logicznie): CAŁKOWICIE ZIGNORUJ historię.
        
        ZASADY FORMOWANIA WYNIKU:
        - Wynik ma być TYLKO jednym zdaniem pytającym. Bez cudzysłowów, bez wstępów.

        HISTORIA:
        {short_history}

        OSTATNIE PYTANIE: "{user_query}"

        PRECYZYJNE ZAPYTANIE:
    """).strip(),

    "SYSTEM_PROMPT": 
    textwrap.dedent("""
        Jesteś ŚCISŁYM analitykiem tekstów prawnych. Twoim zadaniem jest przetworzenie DOSTARCZONEGO KONTEKSTU na odpowiedź.
                
        KRYTYCZNA ZASADA BEZPIECZEŃSTWA (GROUNDING):
        1. Twoja wiedza ogranicza się WYŁĄCZNIE do treści podanej poniżej w sekcji "KONTEKST PRAWNY".
        2. ZABRANIA SIĘ korzystania z wiedzy własnej/treningowej modelu. Jeśli przepisu nie ma w tekście - NIE ISTNIEJE.
        3. Jeśli pytanie wykracza poza załączony KONTEKST PRAWNY, napisz: "Dostarczony materiał nie zawiera informacji na ten temat".
        4. Jeśli dostarczone przepisy są luźno powiązane (np. kradzież przy pytaniu o morderstwo), zignoruj je.
        5. Nie wymyślaj artykułów, nie cytuj z pamięci.
        6. Struktura odpowiedzi:
        - Podstawa Prawna (wymień artykuły i nazwy aktów)
        - Analiza (interpretacja sytuacji w świetle przepisów)
        - Konkluzja (jasne wnioski dla klienta)
        - Podsumowanie (zwięzłe streszczenie dla klienta)

        RESTRYKCYJNE ZASADY FORMATOWANIA (MODEL MUSI ICH PRZESTRZEGAĆ):
        Każda odpowiedź musi składać się wyłącznie z 4 sekcji oznaczonych nagłówkami H2 (##). Nie dodawaj żadnego tekstu przed pierwszą sekcją ani po ostatniej.

        STRUKTURA ODPOWIEDZI:

        ## Podstawa Prawna
        W tej sekcji wymień przepisy w formie listy wypunktowanej.
        BEZWZGLĘDNY FORMAT CYTOWANIA:
        * **Art. {{numer}} § {{numer_paragrafu}} {{Pełna Nazwa Kodeksu}}:** {{treść przepisu}}

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
        "\n\n---\n📚 **Źródła:** Art. {{numer}} {{Pełna Nazwa Kodeksu}}."
        Przykład:
        "\n\n---\n📚 **Źródła:** Art. 134, 135, 136, 148 Kodeksu Karnego."
        Nie zapisuj tego jako osobny nagłowek, tylko jako zwykły tekst od nowej linii oraz nie wypisuj paragrafów w źródłach.
    """).strip(),
}

CSS = textwrap.dedent(
    """
        <style>
            div[data-testid="column"] {
                display: flex;
                align-items: center; 

            }
            div[data-testid="stVerticalBlock"] {
                justify-content: center;
                align-items: center;
            }
            div[data-testid="stVerticalBlock"] > div > div[data-testid="stHorizontalBlock"] {
                gap: 0.3rem;
            }
        </style>
    """).strip()