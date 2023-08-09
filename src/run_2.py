import yaml
import hydra
from hydra.utils import instantiate
import omegaconf
from omegaconf import DictConfig, OmegaConf, open_dict
from protocols import NegotiationProtocol


def unpack_nested_yaml(x):
    for key, value in x.items():
        if isinstance(value, (dict, omegaconf.dictconfig.DictConfig)):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, str) and nested_value.endswith('.yaml'):
                    with open(nested_value, 'r') as file:
                        yaml_data = yaml.safe_load(file)
                    x[key].update(yaml_data)
                    try:
                        del x[key][nested_key]
                    except KeyError:
                        pass
                    unpack_nested_yaml(x[key])
        elif isinstance(value, str) and value.endswith('.yaml'):
            with open(value, 'r') as file:
                yaml_data = yaml.safe_load(file)
            x.update(yaml_data)
            try:
                del x[key]
            except KeyError:
                pass
            unpack_nested_yaml(x)
    return x


@hydra.main(version_base=None, config_path="configs", config_name="inference_root")
def main(cfg: DictConfig):

    # allow updating dict https://stackoverflow.com/a/66296809/3723434
    with open_dict(cfg['experiments']):
        y = unpack_nested_yaml(cfg['experiments'])
    # print(OmegaConf.to_yaml(y))

    game = instantiate(cfg.experiments.game)
    agent_1 = instantiate(cfg.experiments.agent_1)
    agent_2 = instantiate(cfg.experiments.agent_2)

    negotiation_protocol = NegotiationProtocol(game=game, agent_1=agent_1, agent_2=agent_2,
                                               **cfg.experiments.negotiation_protocol)
    negotiation_protocol.run()


if __name__ == "__main__":
    main()
