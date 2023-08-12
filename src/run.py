import yaml
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from protocols import NegotiationProtocol
from utils import unpack_nested_yaml


@hydra.main(version_base=None, config_path="configs", config_name="inference_root")
def main(cfg: DictConfig):

    # allow updating omegaconf.dict https://stackoverflow.com/a/66296809/3723434
    with open_dict(cfg['experiments']):
        y = unpack_nested_yaml(cfg['experiments'])
    # print(OmegaConf.to_yaml(y))

    game = instantiate(cfg.experiments.game)
    agent_1 = instantiate(cfg.experiments.agent_1)
    agent_2 = instantiate(cfg.experiments.agent_2)

    negotiation_protocol = NegotiationProtocol(game=game,
                                               agent_1=agent_1,
                                               agent_2=agent_2,
                                               **cfg.experiments.negotiation_protocol)
    negotiation_protocol.run()
    negotiation_protocol.evaluate()

if __name__ == "__main__":
    main()
