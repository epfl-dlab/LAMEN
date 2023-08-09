import hydra
import uuid
from omegaconf import DictConfig
from utils import read_json, dictionary_to_string

from agents import NegotiationAgent
from protocols import NegotiationProtocol
from games import load_game, Game


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
    debug_mode = cfgg["debug_mode"]
    verbosity = cfgg["verbosity"]
    start_agent_index = cfg["start_agent_index"]
    check_faithfulness = cfgg["check_faithfulness"]

    # initialize agents with their stories
    agents = []  # list will store agent metadata
    for agent in agents_raw:
        agent = list(agent.values())[0]  # stupid python dictionary workaround
        agent_data = AgentMetadata(**agent)
        agents.append(NegotiationAgent(
            **agent_data.communication_protocol,
            **agent_data.generation_parameters,
            agent_name=agent_data.agent_name,
            init_description=agent_data.init_description, 
            debug_mode=debug_mode, verbosity=verbosity)
        )
    print(f"running negotiations with {len(agents)} agents:\n{agents[0]}\n{agents[1]}")

    # initialize game
    if "general_rules" in cfg["game"].keys():
        game_information = load_game(cfg["game"]["file"], cfg["game"]["general_rules"])
    else:
        game_information = load_game(cfg["game"]["file"])

    game = Game.from_dict(game_information)

    save_folder = cfgg["output_dir"]
    max_rounds = cfg["max_rounds"]

    negotiation = NegotiationProtocol(agents=agents, game=game,
                                      max_rounds=max_rounds, save_folder=save_folder,
                                      start_agent_index=start_agent_index, 
                                      check_faithfulness=check_faithfulness)
    negotiation.run()
    negotiation.evaluate()


if __name__ == "__main__":
    main()
