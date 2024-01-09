# this function sweeps over combinations found in sweep.yaml
import subprocess
import hydra
import multiprocessing
from omegaconf import DictConfig, open_dict
from dataclasses import dataclass, field
import itertools
import pandas as pd
import random
import os
import glob
import pickle

from utils import unpack_nested_yaml, get_inference_root_overrides

LIMIT_MODEL = "None"

# this mapping is important for the subprocess run
# in particular, it converts the sweep file into command line usable 
mapping = {
    "kwargs": "experiments.game.kwargs",
    "issues": "experiments.game.issues",
    # "save_folder": "experiments.negotiation_protocol.save_folder",
    "issue_weights": "experiments.game.issue_weights",
    "agent_1_internal_description": "experiments.agent_1.internal_description",
    "agent_2_internal_description": "experiments.agent_2.internal_description",
    "agent_1_external_description": "experiments.agent_1.external_description",
    "agent_2_external_description": "experiments.agent_2.external_description",
    "agent_1_show_rounds": "experiments.agent_1.show_rounds",
    "agent_2_show_rounds": "experiments.agent_2.show_rounds",
    "agent_1_visibility": "experiments.agent_1.visibility",
    "agent_2_visibility": "experiments.agent_2.visibility",
    "start_agent_index": "experiments.negotiation_protocol.start_agent_index",
    "agent_1_msg_input_note_history": "experiments.agent_1.msg_input_note_history",
    "agent_1_note_input_note_history": "experiments.agent_1.note_input_note_history",
    "agent_1_msg_input_msg_history": "experiments.agent_1.msg_input_msg_history",
    "agent_1_note_input_msg_history": "experiments.agent_1.note_input_msg_history",
    "agent_2_msg_input_note_history": "experiments.agent_2.msg_input_note_history",
    "agent_2_note_input_note_history": "experiments.agent_2.note_input_note_history",
    "agent_2_msg_input_msg_history": "experiments.agent_2.msg_input_msg_history",
    "agent_2_note_input_msg_history": "experiments.agent_2.note_input_msg_history",
    "agent_1_generation_parameters": "experiments.agent_1.generation_parameters",
    "agent_2_generation_parameters": "experiments.agent_2.generation_parameters",
    "agent_1_msg_max_len": "experiments.agent_1.msg_max_len",
    "agent_2_msg_max_len": "experiments.agent_2.msg_max_len",
    "agent_1_note_max_len": "experiments.agent_1.note_max_len",
    "agent_2_note_max_len": "experiments.agent_2.note_max_len",
    "log_path": "hydra.run.dir",
    "format_as_dialogue": "experiments.negotiation_protocol.format_as_dialogue",
}


@dataclass
class SweepConditions:
    """
    Take in the cfg from sweep.yaml
    Process the agent files for if the agents have mirrored characteristics
        or if we allow the crosses also. 
    Convert all these combinations into a product
    Iterate over the product and use the subprocess module to run the experiment.
    Optionally, we allow for multi-processing
        This is usually determined by api considerations.    
    """
    cfg: DictConfig
    overrides: dict = field(default_factory=dict)
    product: list = field(default_factory=list)
    _mirror_keys: list = field(default_factory=list)
    _off_diagonal_keys: list = field(default_factory=list)

    # optional for cross play to only run on games with missing runs
    loading_runs_dir: str = None
    # for extracting already run games
    analysis_fname: str = "evals.csv"

    def __post_init__(self):
        self.num_runs = self.cfg.num_runs
        self._convert_config()
        self.product_dict(**self.game_axes, **self.negotiation_axes, **self.agent_axes)

    def _convert_config(self):
        # load in the cfg vars to object
        self.game_axes = self.cfg.game
        self.negotiation_axes = self.cfg.negotiation_protocol
        self.agent_axes = self.cfg.agent
        self.load_agent()
        self.num_processes = self.cfg.num_processes

    def load_agent(self):
        """
        two options for loading in agent cfgs
        'mirror' (where both agents have the same value)
        'cross' (where we allow exploration where not equal)
        
        update in place the agent dictionary to expand along 'cross' or 'mirror'
        input: agent cfg object 
        return: None 
        """
        agent_key_vals = self.cfg.agent
        # keep track of mirror and cross keys
        self._mirror_keys = []
        self._off_diagonal_keys = []
        new_agent_dictionary = {}
        for key, val in agent_key_vals.items():
            if val[1] == 'cross':
                new_agent_dictionary["agent_1_" + key] = val[0]
                new_agent_dictionary["agent_2_" + key] = val[0]

            elif val[1] == 'off-diagonal':
                new_agent_dictionary["agent_1_" + key] = val[0]
                new_agent_dictionary["agent_2_" + key] = val[0]
                self._off_diagonal_keys.append(key)

            elif val[1] == 'mirror':
                new_agent_dictionary[key] = val[0]
                self._mirror_keys.append(key)

            else:
                raise NotImplementedError(f"{val[1]} is not an accepted value -- must be 'mirror' or 'cross'")

        self.agent_axes = new_agent_dictionary

    def print_number_experiments(self):
        i = 0
        df = pd.DataFrame()
        for k in self.product:
            df = pd.concat([df, pd.DataFrame([(key, val) for (key, val) in k.items() if "history" in key],
                                             columns=['key', 'val'])])
            i += 1
        print(f"Running {i * self.num_runs} experiments")
        # self.product = self.product_dict(**self.game_axes, **self.negotiation_axes, **self.agent_axes)

    def product_dict(self, **kwargs):
        keys = kwargs.keys()
        self.product = []
        for instance in itertools.product(*kwargs.values()):
            self.product.append(self.process_product(dict(zip(keys, instance))))
        # drop empty (after filtering) and shuffle
        self.product = [key for key in self.product if key is not None]
        random.shuffle(self.product)

    def _load_prev_runs(self, runs, output_dir):
        # in case we want to do an update sweep
        #   only update the runs that we missed.
        run_path = os.path.join(output_dir, "*")
        files = glob.glob(run_path)
        eval_files = [os.path.join(k, 'evals.csv') for k in files]
        if len(files) > 0:
            # get the most recently created evals.csv file.
            most_recent_index = eval_files.index(max(eval_files, key=os.path.getmtime))
            most_recent_run = files[most_recent_index]
            run_path = os.path.join(most_recent_run, self.run_log_fname)
            # load in the list of runs
            # TODO: implement a way to check what files we have run before
            raise NotImplementedError('Write a way to limit the scope of games in our run.')

    def process_product(self, key_vals):
        # do this to not explore the whole cross on mirrors
        on_diagonal = False
        for m_key in self._mirror_keys:
            key_vals["agent_1_" + m_key] = key_vals[m_key]
            key_vals["agent_2_" + m_key] = key_vals[m_key]
            del key_vals[m_key]
        # check to see if a diagonal is present
        for m_key in self._off_diagonal_keys:
            if key_vals['agent_1_' + m_key] == key_vals['agent_2_' + m_key]:
                on_diagonal = True
        if on_diagonal:
            return None 
        return key_vals

    def loop_over(self):
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            pool.map(self.run_experiment, self.product)
        # print("Looping over   .")
        # for p in self.product:
        #     self.run_experiment(p)

    def run_experiment(self, settings):
        run_string = self.convert_inputs_to_string(settings)
        if (LIMIT_MODEL in run_string) or (LIMIT_MODEL == "None"): 
            print(run_string)
            for _ in range(self.num_runs):
                subprocess.run(run_string, shell=True)

    def convert_inputs_to_string(self, settings, experiment="defaults_only"):
        # NOTE: the defaults only file is <empty> --> replaced: new_game_test
        # TODO: MAKE SURE SCORES MAKE SENSE
        names = settings.keys()
        vals = settings.values()
        base_script = f"python src/run.py experiments={experiment}"

        for v, n in zip(vals, names):
            if n == "issues":
                n1, issue_name = "issues", v[0]
                n2, issue_weights = "issue_weights", v[1]
                adjustment1 = mapping[n1]
                adjustment2 = mapping[n2]
                base_script += f" '++{adjustment1}={issue_name}'"
                base_script += f" '++{adjustment2}={issue_weights}'"
            else:
                adjustment = mapping[n]
                base_script += f" '++{adjustment}={v}'"

        # # root overrides
        for k, v in self.overrides.items():
            if (k != "save_folder") & (k != "output_dir"):
                base_script += f" '++{k}={v}'"

        return base_script


@hydra.main(version_base=None, config_path="configs", config_name="sweep")
def main(cfg: DictConfig):
    with open_dict(cfg['experiments']):
        _ = unpack_nested_yaml(cfg['experiments'])
    instantiated_models = {}

    sc = SweepConditions(cfg)
    # unable to make part of internal since 'dataclass' in combination with 'hydra' does not like self.cfg operations
    sc.overrides = get_inference_root_overrides(cfg)
    sc.print_number_experiments()

    sc.loop_over()


if __name__ == "__main__":
    main()
