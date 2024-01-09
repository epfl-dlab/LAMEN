import json
import yaml
import omegaconf
import hydra
import re
import os
import random
import pandas as pd
import numpy as np
from omegaconf import OmegaConf, open_dict
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig


def fuzzy_index_matching(lst, value):
    """
    Function to find the index of an element in a list.
    If the element is not found, it returns the index of the element with closest value.
    """

    def extract_digit(value_, re_int=re.compile(r'\d+')):
        offer_ = re_int.search(value_)
        if offer_ is not None:
            offer_ = int(offer_[0])
        return offer_

    # Try to find the index in the list
    idx = None
    try:
        idx = lst.index(value)
    # If value is not in the list
    except ValueError:
        # If value not found, return the lexicographically closest element's index
        try:
            print('getting closest match')
            offer = extract_digit(value)
            if offer is not None:
                payoffs = np.asarray([extract_digit(val) for val in lst if extract_digit(val) is not None])

                payoffs = payoffs - offer
                idx = np.argmin(np.abs(payoffs))

        except Exception as e:
            print(f'error: unable to find closest match - {e}')
    return idx


def extract_dictionary(x):
    if isinstance(x, str):
        regex = r"{.*?}"
        match = re.search(regex, x, re.MULTILINE | re.DOTALL)
        if match:
            try:
                json_str = match.group()
                json_str = json_str.replace("'", '"')
                dict_ = json.loads(json_str)
                return dict_
            except Exception as e:
                print(f"unable to extract dictionary - {e}")
                return None

        else:
            return None
    else:
        return None


def check_if_model_instantiated(instantiated_models, agent_config):
    model_name = agent_config.model_name 
    if model_name in instantiated_models.keys():
        agent_config.model = instantiated_models[model_name]
        print(f"'{model_name}' has already been instantiated, using cached version.")
    return agent_config 


def printv(msg, v=0, v_min=0, c=None, debug=False):
    # convenience print function
    if debug:
        c = 'yellow' if c is None else c
        v, v_min = 1, 0
        printc('\n\n>>>>>>>>>>>>>>>>>>>>>>START DEBUG\n\n', c='yellow')
    if (v > v_min) or debug:
        if c is not None:
            printc(msg, c=c)
        else:
            print(msg)
    if debug:
        printc('\n\nEND DEBUG<<<<<<<<<<<<<<<<<<<<<<<<\n\n', c='yellow')


def printc(x, c='r'):
    m1 = {'r': 'red', 'g': 'green', 'y': 'yellow', 'w': 'white',
          'b': 'blue', 'p': 'pink', 't': 'teal', 'gr': 'gray'}
    m2 = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'pink': '\033[95m',
        'teal': '\033[96m',
        'white': '\033[97m',
        'gray': '\033[90m'
    }
    reset_color = '\033[0m'
    print(f'{m2.get(m1.get(c, c), c)}{x}{reset_color}')


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
    def _update_path(p, r_project=re.compile(r'(^.*)GPTeam')):
        """Ensures local paths are updated for experiments that were run on different machines"""
        from_path = r_project.search(p)
        to_path = r_project.search(os.path.abspath(os.getcwd()))
        if to_path is None or from_path is None:
            return p

        return p.replace(from_path[1], to_path[1])

    # important: use shallow copies of dictionaries to avoid changing the size of the dict during iteration
    for key, value in x.copy().items():
        if isinstance(value, (dict, omegaconf.dictconfig.DictConfig)):
            for nested_key, nested_value in value.copy().items():
                if isinstance(nested_value, str) and nested_value.endswith('.yaml'):
                    nested_value = _update_path(nested_value)
                    with open(nested_value, 'r') as file:
                        yaml_data = yaml.safe_load(file)
                    # 12-07-23: do not override k/v pairs that already exist, e.g., by ++overrides of experiment values
                    yaml_data_ = {k: v for k, v in yaml_data.items() if k not in x[key].keys()
                                  or str(x[key].get(k)).endswith('.yaml')}
                    x[key].update(yaml_data_)
                    # print(nested_key, key, yaml_data, '\n')
                    try:
                        # important: if the unpacked yaml data contains a key that is the same as the key
                        # pointing at the file, it will be overwritten. Hence, we should no longer delete the key!
                        if nested_key not in yaml_data_.keys():
                            del x[key][nested_key]
                    except KeyError:
                        pass
                    unpack_nested_yaml(x[key])
        elif isinstance(value, str) and value.endswith('.yaml'):
            value = _update_path(value)
            with open(value, 'r') as file:
                yaml_data = yaml.safe_load(file)
            yaml_data_ = {k: v for k, v in yaml_data.items() if k not in x.keys()}
            x.update(yaml_data_)
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
        _ = unpack_nested_yaml(cfg['experiments'])
    return cfg


def get_inference_root_overrides(cfg_, inference_root_path='src/configs/inference_root.yaml'):
    """Parse inference_root level overrides, e.g., verbosity, max_rounds, etc."""
    root = yaml.safe_load(open(inference_root_path))
    root_keys = root.keys()
    overrides = {}
    for k, v in cfg_.items():
        if k in root_keys and v != root[k]:
            overrides[k] = v

    return overrides


def _update_model_constructor_hydra(model_provider):
    model_target = "models."

    if model_provider in ['azure', 'openai']:
        model_target += "OpenAIModel"
    elif model_provider == 'anthropic':
        model_target += "AnthropicModel"
    elif model_provider == 'cohere':
        model_target += "CohereModel"
    elif model_provider == 'google':
        model_target += "GoogleModel"
    elif model_provider == 'llama':
        model_target += "HuggingFaceModel"
    else:
        raise NotImplementedError('feel free to extend to with custom models')

    return model_target


def update_model_constructor_hydra(cfg_exp):
    cfg_exp['agent_1']['model']['_target_'] = _update_model_constructor_hydra(cfg_exp['agent_1']['model_provider'])
    cfg_exp['agent_2']['model']['_target_'] = _update_model_constructor_hydra(cfg_exp['agent_2']['model_provider'])


def fill_defaults(x, root_overrides={}, defaults_file='data/negotiation_defaults.yaml'):
    """
    A negotiation is defined as a yaml file containing at least a user-defined 'game' description.
    The other objects are: agent_1, agent_2, and the negotiation_protocol
    For each dictionary, we check for required values in the 'defaults' file.
    Each default_file entry either points to a dictionary or a {value, type, desc} triplet (a 'leaf')
    IF pointing at dictionary: continue recursion
    ELIF: pointing at leaf: fill-in default if empty and exit
    ELSE: pointing at user-defined 'free' variable: exit
    """
    def _is_leaf(k, leaf={'value', 'type', 'desc'}):
        # helper function to determine end of recursion
        return set(k) == leaf

    def _fill_defaults_recursion(x_, d):
        # continue recursion until either: (1) a leave, or (2) a user-defined free variable is found
        for k, v in x_.items():
            d_ = d.get(key_pairing.get(k, k), {})
            if _is_leaf(d_.keys()):
                if v is None or (isinstance(v, (dict, omegaconf.dictconfig.DictConfig)) and not any(v)):
                    # defaults are never None or empty dicts
                    x_[k] = d_['value']
                else:
                    # not implemented: check type
                    pass
            else:
                if isinstance(v, (dict, omegaconf.dictconfig.DictConfig)):
                    # check if all mandatory keys are present
                    v_filled = v
                    for dk, dv in d_.items():
                        v_ = v.get(dk)
                        v_filled[dk] = {} if v_ is None else v_
                    x_[k] = v_filled
                    # recursion step to fill nested dictionaries, e.g., 'internal_description'
                    _fill_defaults_recursion(x_[k], d_)
                else:
                    # user defined 'free' variable, e.g. inside 'internal_description' of agent
                    pass
        return x_

    key_pairing = {
        'game': 'game',
        'agent_1': 'agent',
        'agent_2': 'agent',
        'negotiation_protocol': 'negotiation_protocol'
    }
    # minimal key check: at least a game object must be described
    xk = list(x.keys())
    min_keys = list(key_pairing.keys())
    if 'game' not in xk:
        raise ValueError('error: no game object defined in YAML file!')
    for mk in min_keys:
        if mk not in xk:
            x[mk] = {}

    # change nested default values to run time overrides
    defaults = yaml.safe_load(open(defaults_file))
    for ok, ov in root_overrides.items():
        for default_kv in [defaults[kp] for kp in key_pairing.values()]:
            for default_kv_k in default_kv.copy().keys():
                if default_kv_k == ok:
                    default_kv[default_kv_k]['value'] = ov

    x = _fill_defaults_recursion(x, defaults)

    return x


def sample_run(run_dir='logs/inference/runs'):
    file = random.choice(os.listdir(run_dir))
    return os.path.join(run_dir, file)


def inspect_game(load_path, save_path=None):
    transcript = os.path.join(load_path, "processed_negotiation.csv")
    config_path = os.path.join("..", load_path, ".hydra")
    config_data = load_hydra_config(config_path)
    game = config_data['experiments']['game']
    agent_1_model_name = config_data['experiments']['agent_1'].model_name 
    agent_2_model_name = config_data['experiments']['agent_2'].model_name 
    negotiation_protocol = config_data['experiments']['negotiation_protocol']
    df = pd.read_csv(transcript)

    lj = 25
    for k, v in zip(['file', 'game', 'issues', 'issue_weights', 'format_as_dialogue', 'agent_1_model_name', 'agent_2_model_name'],
                    [load_path, game["name"], game["issues"], game["issue_weights"], negotiation_protocol['format_as_dialogue'], agent_1_model_name, agent_2_model_name]):
        print(f'{k}:'.ljust(lj) + f'{v}')
    else:
        print()

    to_save = []
    for i, row in df.iterrows():
        c = 'r' if row['agent_id'] == 0 else 'b'

        x = f'''<round: {row["round"]}, agent: {row["agent_id"]}>
    [note]\n{row["note"]}\n-->\n{row["offers_in_note"]}\n--\n
    [msg]\n{row["message"]}\n-->\n{row["offers_in_message"]}\n--\n
        '''
        printc(x, c)
        to_save.append(x)

    if save_path is not None:
        with open(os.path.join(load_path, save_path + '.txt'), 'w') as f:
            f.write('\n'.join(to_save))


def get_memory_settings(load_path):
    config_path = os.path.join("..", load_path, ".hydra")
    config_data = load_hydra_config(config_path)

    keys = ['note_input_note_history', 'note_input_msg_history', 'msg_input_note_history', 'msg_input_msg_history']
    agents = ['agent_1', 'agent_2']
    print(f'[memory settings]:')
    for a in agents:
        print(a)
        for k in keys:
            try:
                v = config_data['experiments'][a][k]
            except Exception as e:
                v = 'error'
            print(f'  {k}:'.ljust(20) + f'{v}')

def sample_games_from_pandas(df, **kwargs):
    for key, val in kwargs.items():
        try: 
            df = df[df[key] == val]
        except Exception as e:
            print(f'Failed to limit columns, likely due to inaccurate naming - {e}')
    
    return df['log_path'].sample(1).values[0]

def error_runs(df, agreement_reason=['full agreement','aligning internal states, textual disagreement'], error_type='all'):
    # TODO: this may be problematic if they think they have interests opposite to what they have.
    # TODO: should it be full agreement or synthetic?
    epsilon = 1e-6
    error_1 = ((df['agent_1_normalized_payoff'] == 0) & (
            df['agent_2_normalized_payoff'] == 0) & (df["completion_reason"].isin(agreement_reason))) 
    error_2 = ((1 - df['agent_1_normalized_payoff'] - df['agent_2_normalized_payoff'] > epsilon) & 
            (df["completion_reason"].isin(agreement_reason)) & (df['game_type'] == 'non-integrative distributive'))
    error_3 = ((df['agent_1_normalized_payoff'] - df['agent_2_normalized_payoff'] > epsilon) & 
            (df["completion_reason"].isin(agreement_reason)) & (df['game_type'] == 'non-integrative compatible') &
                (df['num_issues'] == 1))
    if error_type == 'all':
        df = df[error_1 | error_2 | error_3]
    elif error_type == 1:
        print('Both sides have payoff 0 when game completes.')
        df = df[error_1]
    elif error_type == 2:
        print('Non-integrative distributive games but payoff a_1 != 1 - payoff a_2.')
        df = df[error_2]
    elif error_type == 3:
        print('Single issue compatible games but they dont have the same payoff')
        df = df[error_3]
    return df