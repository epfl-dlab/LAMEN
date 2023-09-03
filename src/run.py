import time
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from protocols import NegotiationProtocol, InterrogationProtocol
from utils import unpack_nested_yaml, fill_defaults


@hydra.main(version_base=None, config_path="configs", config_name="inference_root")
def main(cfg: DictConfig):
    with open_dict(cfg['experiments']):
        _ = unpack_nested_yaml(cfg['experiments'])  

    game = instantiate(cfg.experiments.game)
    agent_1 = instantiate(cfg.experiments.agent_1)
    agent_2 = instantiate(cfg.experiments.agent_2)
    negotiation_protocol = NegotiationProtocol(game=game,
                                            agent_1=agent_1,
                                            agent_2=agent_2,
                                            **cfg.experiments.negotiation_protocol) 
    negotiation_protocol.run()
    negotiation_protocol.evaluate()
    negotiation_protocol.interrogate(**cfg.interrogations)
    time.sleep(5)


if __name__ == "__main__":
    main()
