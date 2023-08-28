import time
import subprocess

EXPERIMENTS = [
    "negotiation_32_32_150_150",
    "negotiation_32_32",
    "negotiation_32_150",
    "negotiation_150_32_32_150",
    "negotiation_150_32",
    "negotiation_150_150_32_32",
    "negotiation_150_150"
]
GAMES_DIR="data/games/"

GAMES = [
    # "rio_copa_integrative_compatible.yaml", 
    # "rio_copa_integrative_non-equal_weights.yaml", 
    # "rio_copa_non-integrative_compatible.yaml", 
    # "rio_copa_non-integrative_equal_weights.yaml", 
    "rio_copa_integrative_one_side_equal.yaml"
]

if __name__=="__main__":
    for e in EXPERIMENTS:
        for g in GAMES:
            subprocess.run(f"python src/run.py experiments={e} ++experiments.game.kwargs={GAMES_DIR}{g} ++experiments.negotiation_protocol.start_agent_index=1", shell=True)   
            time.sleep(10)