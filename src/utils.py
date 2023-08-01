import json
from typing import Tuple
from pathlib import Path
from config import NEGOTIATIONS_DIR
from omegaconf import DictConfig, OmegaConf


def printv(msg, v=0, v_min=0):
    # convenience print function
    if v > v_min:
        print(msg)


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


def get_api_key(fname='secrets.json', key='dlab_openai_key'):
    try:
        with open(fname) as f:
            api_key = json.load(f)[key]
    except Exception as e:
        print(f'error: unable to load api key {key} from file {fname} - {e}')
        return None

    return api_key


def read_json(path_name: str):
    f = open(path_name, "r")
    json_file = json.load(f)
    return json_file


def format_dictionary(dictionary, indent=0):
    result = ""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            result += f"{' ' * indent}{key}:\n{format_dictionary(value, indent + 4)}"
        else:
            result += f"{' ' * indent}{key}: {value}\n"
    return result


def dictionary_to_string(dictionary):
    return format_dictionary(dictionary)


if __name__ == "__main__":
    print(return_agent_prompts("cpc"))
