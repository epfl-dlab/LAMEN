import json
import yaml
import omegaconf
from hydra import initialize, compose
import hydra

from omegaconf import OmegaConf, open_dict
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig


def printv(msg, v=0, v_min=0):
    # convenience print function
    if v > v_min:
        print(msg)


def get_api_key(fname='secrets.json', provider='openai', key='dlab_key'):
    try:
        with open(fname) as f:
            keys = json.load(f)[provider]
            if key is not None:
                api_key = keys[key]
            else:
                api_key = list(keys.values())[0]
    except Exception as e:
        print(f'error: unable to load {provider} api key {key} from file {fname} - {e}')
        return None

    return api_key


def read_json(path_name: str):
    with open(path_name, "r") as f:
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


def unpack_nested_yaml(x):
    """
    Unpacks a nested yaml file of the following form:
    x = {
      a: 1,
      b: v.yaml
    }
    with v = {b: 2, c: 3}
    -->
    unpack_nested_yaml(x) = {a: 1, b: 2, c: 3}

    :param x: (dict) with yaml references
    :return: (dict)
    """
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


def load_hydra_config(config_path, config_name="config"):
    """
    Loads hydra from a .config run.
    """
    GlobalHydra.instance().clear()
    hydra.initialize(config_path=config_path, version_base=None)
    cfg = hydra.compose(config_name=config_name, return_hydra_config=True)
    HydraConfig().cfg = cfg
    OmegaConf.resolve(cfg)
    with open_dict(cfg['experiments']):
        y = unpack_nested_yaml(cfg['experiments'])
    return cfg


def fill_defaults(x, defaults_file='data/negotiations_defaults.yaml'):
    defaults = yaml.safe_load(defaults_file)

    key_pairing = {
        'game': 'game',
        'agent_1': 'agent',
        'agent_2': 'agent',
        'negotiation_protocol': 'negotiation_protocol'
    }

    def _fill_defaults(x_, d):
        leaf = {'value', 'type', 'desc'}
        for k, v in x_.items():
            d_ = d.get(key_pairing.get(k, k))
            if set(d_.keys() == leaf):
                pass  # check type
            else:
                if isinstance(v, (dict, omegaconf.dictconfig.DictConfig)):
                    # k -> dictionary: check if all mandatory keys are present
                    v_filled = {}
                    for dk, dv in d_.items():
                        v_ = v.get(dk)
                        if v_ is None:
                            v_filled[dk] = dv
                        else:
                            v_filled[dk] = v_
                    x_[k] = v_filled
                    _fill_defaults(x_, d)
                else:
                    # k -> not pointing to a dictionary, replace all keys w/ defaults
                    x_[k] = d_
                    _fill_defaults(x_, d)
        return x_

    x = _fill_defaults(x, defaults)
    return x
