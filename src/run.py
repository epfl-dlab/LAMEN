import hydra
import json
import fire
from omegaconf import DictConfig, OmegaConf
from utils import read_json, dictionary_to_string, get_api_key

from agents import NegotiationAgent, NegotiationProtocol
from games import load_game, Game, Issue

from dotenv import load_dotenv
import os 

load_dotenv()  # take environment variables from .env.

class AgentMetadata:
    def __init__(self, agent_description, communication, generation_parameters):
        self.description = read_json(agent_description)
        self.init_description = dictionary_to_string(self.description) # format the dictionary to add to prompt
        self.commmuncation_protocol = read_json(communication)
        self.generation_parameters = read_json(generation_parameters)       


@hydra.main(version_base=None, config_path="configs", config_name="inference_root")
def main(cfg: DictConfig):
    cfgg = cfg
    cfg = cfg["experiments"]
    agents_raw = cfg["agents"]
    
    # initialize agents with their stories
    agents = [] # list will store agent metadata 
    for agent in agents_raw:
        agent = list(agent.values())[0]     # stupid python dictionary workaround 
        agent_data = AgentMetadata(**agent)
        agents.append(NegotiationAgent(
            **agent_data.commmuncation_protocol, 
            **agent_data.generation_parameters,
            init_description=agent_data.init_description)
        )
    print(f"We have {len(agents)}. There descriptions are {agents[0]}{agents[1]}")

    # initilaize game
    game_information = load_game(cfg["game"]["file"])
    game = Game.from_dict(game_information)
    
    game_shared_description = game.description
    issues_format = game.format_all_issues(0)
    agent_side = game.sides[0][0]

    agents[0].create_static_system_prompt(game_shared_description, agent_side, issues_format)
    agents[1].create_static_system_prompt(game_shared_description, agent_side, issues_format)

    save_folder = cfgg["output_dir"]
    max_rounds = cfg["max_rounds"]

    negotiation = NegotiationProtocol(agents=agents, game=game,
                                      max_rounds=max_rounds, save_folder=save_folder)
    negotiation.run()
    
    
if __name__ == "__main__":
    main()
