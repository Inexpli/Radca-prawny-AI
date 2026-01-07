from config import CONFIG
from core import LegalAdvisorAI
from rich.live import Live
from rich.panel import Panel
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
import re as regex

if __name__ == "__main__":
    advisor = LegalAdvisorAI()

    history = [] 
    MAX_HISTORY = CONFIG["MAX_HISTORY"]
    
    try:
        while True:
            q = advisor.console.input("\n[bold green]Podaj pytanie[/] (lub 'exit'): ").strip()
            
            if q.lower() in ['exit', 'wyjdz']: break
            if q.lower() == 'clear':
                history = []
                advisor.console.print("[yellow]Wyczyszczono historię.[/yellow]")
                continue
            if not q: continue
            
            gen_kwargs, streamer, thread = advisor.generate_response(q, history)

            full_response = ""
            live_panel = Panel(Markdown("..."), title="Opinia Prawna", border_style="cyan", expand=False)
        
            SPECIAL_TOKEN_RE = regex.compile(
                r"<\|[^>]+\|>|\<\s*END_OF_TEXT\s*\>|\n{3,}"
            )

            with Live(live_panel, console=advisor.console, refresh_per_second=15) as live:
                for new_text in streamer:
                    full_response += new_text
                    cleaned_partial = SPECIAL_TOKEN_RE.sub("", full_response).strip()
                    live.update(Panel(Markdown(cleaned_partial), title="Opinia Prawna", border_style="cyan", expand=False))
            
            cleaned_response = SPECIAL_TOKEN_RE.sub("", full_response).strip().rstrip('"').rstrip("'")
            
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": cleaned_response})

            if len(history) > MAX_HISTORY:
                history = history[-MAX_HISTORY:]
            
    except KeyboardInterrupt:
        print("\nPrzerwano.")
    finally:
        advisor.close()