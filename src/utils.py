from pathlib import Path
from config import NEGOTIATIONS_DIR

def concatenate_story_payoff_table(name):
    if name not in ["cpc", "hp_costa"]:
        raise NameError("Name must be either 'cpc' or 'hp_costa'")
    
    main_text = open(NEGOTIATIONS_DIR / f"{name}.txt").read()
    main_payoff = open(NEGOTIATIONS_DIR / f"{name}_payoff.txt").read()

    return main_text + "\n\n" + main_payoff

if __name__=="__main__":
    print(concatenate_story_payoff_table("cpc"))