import hydra
import json
import fire
import os
import uuid
from omegaconf import DictConfig, OmegaConf
from utils import read_json, dictionary_to_string, get_api_key

from agents import NegotiationAgent, NegotiationProtocol
from games import load_game, Game, Issue


class AgentMetadata:
    def __init__(self, agent_description, communication, generation_parameters):
        self.description = read_json(agent_description)
        self.agent_name = self.description.get('name', str(uuid.uuid4()))
        self.init_description = dictionary_to_string(self.description)  # format the dictionary to add to prompt
        self.communication_protocol = read_json(communication)
        self.generation_parameters = read_json(generation_parameters)


@hydra.main(version_base=None, config_path="configs", config_name="inference_root")
def main(cfg: DictConfig):
    cfgg = cfg
    cfg = cfg["experiments"]
    agents_raw = cfg["agents"]

    # initialize agents with their stories
    agents = []  # list will store agent metadata
    for agent in agents_raw:
        agent = list(agent.values())[0]  # stupid python dictionary workaround
        agent_data = AgentMetadata(**agent)
        agents.append(NegotiationAgent(
            **agent_data.communication_protocol,
            **agent_data.generation_parameters,
            agent_name=agent_data.agent_name,
            init_description=agent_data.init_description)
        )
    print(f"running negotiations with {len(agents)} agents:\n{agents[0]}\n{agents[1]}")

    # initialize game
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
