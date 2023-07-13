from typing import Tuple
from pathlib import Path
from config import NEGOTIATIONS_DIR

def return_agent_prompts(name: str) -> Tuple[str, str]:
    if name not in ["cpc", "hp_costa"]:
        raise NameError("Name must be either 'cpc' or 'hp_costa'")
    
    main_text = open(NEGOTIATIONS_DIR / f"{name}.txt").read()
    main_payoff = open(NEGOTIATIONS_DIR / f"{name}_payoff.txt").read()

    return main_text, main_payoff

def notes_prompts() -> Tuple[str, str]:
    init_notes = open(NEGOTIATIONS_DIR / f"initialize_notes.txt").read()
    update_notes = open(NEGOTIATIONS_DIR / f"update_notes.txt").read()
    return init_notes, update_notes

if __name__=="__main__":
    print(return_agent_prompts("cpc"))