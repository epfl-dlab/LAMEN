import json
from typing import Tuple
from pathlib import Path
from config import NEGOTIATIONS_DIR
from omegaconf import DictConfig, OmegaConf

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

def format_skeleton(agent_cfg, other_agent_cfg):
    name = agent_cfg["name"]
    company_name = agent_cfg["company_name"]
    skeleton_path = agent_cfg["story_path"]
    payoff_file = agent_cfg["payoff_file"]
    qualities = agent_cfg["qualities"]
    
    other_agent_name = other_agent_cfg["name"]
    other_agent_company_name = other_agent_cfg["company_name"]
    
    f = open(skeleton_path, "r")
    
    f.replace("{name}", name).replace("{occupation}", agent_cfg["qualities"]["profession"])


def get_api_key(fname, key):
    try:
        with open(fname) as f:
            api_key = json.load(f)[key]
    except (FileExistsError, FileNotFoundError, KeyError) as e:
        print(f'error: {e}')
        return None

    return api_key

    


if __name__=="__main__":
    print(return_agent_prompts("cpc"))