"""
1. design experiment, exp_ABC
2. experiments conducts K negotiations, each saves results in folder, logs/exp_ABC/run_i:
    b. negotiations.csv  # TODO: make sure the 's' is dropped...
    c. processed_negotiation.csv
    d. (optional) interrogation.csv
3. analyze experimental results by calling: analyze_experiments.ExperimentAnalysis()
    a. creates a folder mirroring the experiment name under analyses/exp_ABC
    b. for each run in logs/exp_ABC, e.g. {run_i}_i^K, creates a HistoryObject
        -> HistoryObject: calculates metrics of interest, e.g. instruction following, faithfulness, scores, etc.
    d. finally, creates aggregated metrics w/ confidence intervals over all HistoryObjects and saves:
        -> based on the analysis, a different .yaml file will point at the analysis class to use
        -> results saved as: analyses/exp_ABC/<date>/(analysis.csv, exp_analysis.pkl)
    e. when we rerun the analysis for this experiment, e.g., when new runs are conducted:
        i. load pickled object and check logs/exp_ABC/ for new run_files
        ii. IF: no new run_files, abort, ELSE: compute HistoryObject for new files and re-compute aggregated metrics
"""

# TODO: ADD A HEURISTIC FOR WHEN THEY AGREE WITHOUT SAYING THEY AGREE

import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime as dt
import warnings
import pickle
from omegaconf import DictConfig
import hydra
import attr
from attr import field, define
import uuid
import omegaconf

from agents import NegotiationAgent
from protocols import NegotiationProtocol, InterrogationProtocol
from process_transcripts import ProcessTranscript
from games import Game
from utils import load_hydra_config, extract_dictionary, fuzzy_index_matching, printv

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 250)


@define
class ExperimentAnalysis:
    """
    Load and filter specific set of runs
    print a specific aggregation from src/evaluation_methods.

    input: run_folder
    output: None
    """
    run_name: str
    aggregation_method: DictConfig
    negotiation_summaries: list = field(factory=list)
    save_path: str = None

    # limit by datetime
    runs_after_date: str = "2023-09-15"
    runs_after_time: str = "00-00-00-000000"

    # save file names
    analysis_fname: str = "evals.csv"
    run_log_fname: str = "run_log.pkl"
    # manual overwrite 
    overwrite: bool = False
    # sort col
    sort_by: list = None
    # keep df in history
    _df_processed: pd.DataFrame = None

    def __attrs_post_init__(self):
        self._load_configs()

    def _load_configs(self):
        run_dir = os.path.join("logs", self.run_name, "runs")
        output_dir = os.path.join("data", "processed_outputs", self.run_name)
        save_path = os.path.join(output_dir, dt.now().strftime("%d-%m-%Y_%H-%M-%S"))
        self.save_path = save_path
        print(f'Results saved at: {save_path}')
        runs = self._load_runs(run_dir)
        df_processed = self._load_analysis(runs, output_dir)
        if (df_processed is None) or self.overwrite:
            run_metrics = []
            for run in runs:
                # try:
                negotiation_summary = NegotiationMetrics(run)
                self.negotiation_summaries.append(negotiation_summary)
                #except Exception as e:
                 #   print(f"Error for file {run} - {e}")
                    #continue
                if negotiation_summary.completed:
                    metrics = negotiation_summary.metrics
                    run_metrics.append(metrics)

            df_processed = pd.DataFrame(run_metrics)
            # save run
            self._df_processed = df_processed
            self._save_analysis(runs, save_path)
        else:
            self._df_processed = df_processed

        # TODO: where should these fns go?
        print(len(self._df_processed))
        self._zero_out_payoffs()
        print(len(self._df_processed))

        self._remove_empty_messages()
        print(len(self._df_processed))

    def _load_runs(self, run_dir):
        # load runs, (optional) restrict by datetime.
        glob_path = os.path.join(run_dir, "*")
        runs = glob.glob(glob_path)
        limit_after = os.path.join(run_dir, f'{self.runs_after_date}_{self.runs_after_time}')
        runs = sorted([run for run in runs if run >= limit_after])
        return runs

    def _save_analysis(self, runs, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        # define the paths
        df_save_path = os.path.join(save_path, self.analysis_fname)
        run_save_path = os.path.join(save_path, self.run_log_fname)
        # save the file and runs
        self._df_processed.to_csv(df_save_path, index=False)
        with open(run_save_path, "wb") as f:
            pickle.dump(runs, f)

    def _load_analysis(self, runs, output_dir):
        # load history if the current set of runs matches a previous run.
        run_path = os.path.join(output_dir, "*")
        files = glob.glob(run_path)
        eval_files = [os.path.join(k, self.analysis_fname) for k in files]
        if len(files) > 0:
            # get the most recently created evals.csv file.
            most_recent_index = eval_files.index(max(eval_files, key=os.path.getmtime))
            most_recent_run = files[most_recent_index]
            run_path = os.path.join(most_recent_run, self.run_log_fname)
            # load in the list of runs
            if os.path.exists(run_path):
                with open(run_path, "rb") as f:
                    old_runs = pickle.load(f)
                if not self.overwrite:
                    print('Most recent run: ', most_recent_run)
                    self.save_path = os.path.join(most_recent_run)
                    return pd.read_csv(os.path.join(self.save_path, 'evals.csv'))

                if set(old_runs) == set(runs) and not self.overwrite:
                    print("Previous run exists, loading in.")
                    return pd.read_csv(os.path.join(most_recent_run, self.analysis_fname))
            else:
                raise ValueError(f'Path - {run_path} does not exist.')

    def _zero_out_payoffs(self):
        # if the final round does not converge, then set the payoff to zero
        self._df_processed['agent_1_normalized_payoff'] = self._df_processed.apply(
            lambda x: 0 if x['completion_rate'] == 0 else x['agent_1_normalized_payoff'], axis=1)
        self._df_processed['agent_2_normalized_payoff'] = self._df_processed.apply(
            lambda x: 0 if x['completion_rate'] == 0 else x['agent_2_normalized_payoff'], axis=1)

    def _remove_empty_messages(self):
        self._df_processed = self._df_processed[~self._df_processed['contains_empty_message']]
        # self.df_processed = self.df_processed[self.df_processed['contains_empty_note'] == False]
        self._df_processed = self._df_processed[
            ~ ((self._df_processed['completion_reason'] == 'in-progress') & (self._df_processed['num_rounds'] < 10))]

    def run_analysis(self):
        analysis_method = hydra.utils.instantiate(self.aggregation_method, df=self._df_processed,
                                                  save_path=self.save_path)
        analysis_method.run()

    def missing_runs(self):
        # optionally check if there are any runs missing?
        raise NotImplementedError("Optional measure to see if there are any missing runs.")


@define
class NegotiationMetrics:
    """
    Load and filter a completed negotiation run.

    input: run_folder
    output: metrics

    run_folder:
    - [raw] negotiation.csv
    - [required] [running eval funcs] processed_negotiation.csv
    - [optional, external-faithfulness] interrogation.csv
    - .hydra
    """
    negotiation_log_path: str
    metrics: dict = field(factory=dict)
    negotiation_id: str = field(default=str(uuid.uuid4()))
    completed: bool = False
    agent_1: NegotiationAgent = None
    agent_2: NegotiationAgent = None
    game: Game = None
    protocol: NegotiationProtocol = None
    interrogation: bool = False
    _df: pd.DataFrame = field(alias="_df", default=None)
    _df_interrogation: pd.DataFrame = field(alias="_df_interrogation", default=None)

    def __attrs_post_init__(self):
        # loads in config fields used to perform negotiation
        cfg = load_hydra_config(os.path.join("..", self.negotiation_log_path, ".hydra/"))
        # loads the created negotiation data
        self._load_data()
        # compute metrics
        if cfg and isinstance(self._df, pd.DataFrame):
            self._compute_metrics(cfg)

    def _load_data(self):
        try:
            self._df = pd.read_csv(os.path.join(self.negotiation_log_path, "processed_negotiation.csv"))
            assert "offers_in_message" in self._df.columns
            try:
                self._df_interrogation = pd.read_csv(os.path.join(self.negotiation_log_path, "interrogation.csv"))
                self.interrogation = True
            except FileNotFoundError as e:
                print(f"[warning] unable to load interrogation file - {e}")
        except FileNotFoundError as e:
            print(f'[error] unable to load negotiation data - {e}')

    def _compute_metrics(self, cfg):
        # load negotiation objects, i.e., agents, game, protocol
        self._load_negotiation(cfg)
        # get faithfulness metrics
        self._get_faithfulness_metrics()
        # 3x instruction following
        self._get_instruction_following_metrics()
        # performance, e.g., score, optimal score
        self._get_performance_metrics()
        # metrics about the run, e.g., errors
        self._get_run_meta_metrics()
        # signal that all metrics have been computed
        self.completed = True

    def _load_negotiation(self, cfg):
        self.agent_1 = hydra.utils.instantiate(cfg["experiments"]["agent_1"])
        self.agent_2 = hydra.utils.instantiate(cfg["experiments"]["agent_2"])
        self.game = hydra.utils.instantiate(cfg["experiments"]["game"])

        # # NOTE: originally required for backward compatibility, as the hydra defaults routine failed
        # # if we have no _target_ then we will need to instantiate objects
        if isinstance(self.game, omegaconf.dictconfig.DictConfig):
            self.game = Game(**self.game)
        if isinstance(self.agent_1, omegaconf.dictconfig.DictConfig):
            self.agent_1 = NegotiationAgent(**self.agent_1)
        if isinstance(self.agent_2, omegaconf.dictconfig.DictConfig):
            self.agent_2 = NegotiationAgent(**self.agent_2)

        optimal_payoff = self.game.get_optimal_score()

        protocol_keys = list(cfg.experiments.negotiation_protocol.keys())
        protocol_vals = list(cfg.experiments.negotiation_protocol.values())

        agent_keys = list(attr.fields_dict(NegotiationAgent).keys())
        agent_1_vals = [getattr(self.agent_1, field_name) for field_name in agent_keys]
        agent_2_vals = [getattr(self.agent_2, field_name) for field_name in agent_keys]

        game_keys = list(attr.fields_dict(Game).keys())
        game_vals = [getattr(self.game, field_name) for field_name in game_keys]
        game_type = self.game.get_game_type()
        game_class = self.game.get_game_class()

        field_tuples = [
            ("game", game_keys, game_vals),
            ("agent_1", agent_keys, agent_1_vals),
            ("agent_2", agent_keys, agent_2_vals),
            ("protocol", protocol_keys, protocol_vals),
        ]

        negotiation_fields = {
            'optimal_payoff': optimal_payoff,
            'game_type': game_type,
            'game_class': game_class,
            'num_issues': len(self.game.issues)}
        for name, keys, values in field_tuples:
            for k, v in zip(keys, values):
                negotiation_fields[f"{name}_{k}"] = v

        self.metrics.update(negotiation_fields)

    def _get_run_meta_metrics(self):
        negotiation_cs = NegotiationProtocol.get_save_headers(return_as_dict=True)
        c_message, c_note = negotiation_cs['c_message'], negotiation_cs['c_note']

        contains_empty_message = len(self._df[self._df[c_message].isnull()]) > 0
        contains_empty_note = len(self._df[self._df[c_note].isnull()]) > 0
        # TODO: maybe add a method for the run NOT completing.

        self.metrics.update({'contains_empty_message': contains_empty_message,
                             'contains_empty_note': contains_empty_note})

    def _get_performance_metrics(self):
        self._get_agent_metrics()
        self._get_game_summary_stats()

    def _get_agent_metrics(self, return_metrics=False):
        negotiation_cs = NegotiationProtocol.get_save_headers(return_as_dict=True)
        eval_cs = ProcessTranscript.get_metric_headers(return_as_dict=True)

        c_agent_id = negotiation_cs['c_agent_id']
        c_msg_words, c_note_words = eval_cs['c_msg_words'], eval_cs['c_note_words']
        c_msg_tokens, c_note_tokens = eval_cs['c_msg_tokens'], eval_cs['c_note_tokens']
        c_normalized_payoff = eval_cs["c_normalized_payoff"]

        agent_metrics = {}
        for agent_id in [0, 1]:
            df_agent = self._get_df_agent_id(c_agent_id, agent_id)
            last_row = df_agent.tail(1)
            agent_metrics[f'agent_{agent_id + 1}_mean_{c_msg_words}'] = df_agent[c_msg_words].mean()
            agent_metrics[f'agent_{agent_id + 1}_mean_{c_note_words}'] = df_agent[c_note_words].mean()
            agent_metrics[f'agent_{agent_id + 1}_mean_{c_msg_tokens}'] = df_agent[c_msg_tokens].mean()
            agent_metrics[f'agent_{agent_id + 1}_mean_{c_note_tokens}'] = df_agent[c_note_tokens].mean()
            agent_metrics[f'agent_{agent_id + 1}_{c_normalized_payoff}'] = last_row[c_normalized_payoff].mean()

        self.metrics.update(agent_metrics)

        if return_metrics:
            return agent_metrics

    def _get_game_summary_stats(self, return_metrics=False):
        negotiation_cs = NegotiationProtocol.get_save_headers(return_as_dict=True)
        eval_cs = ProcessTranscript.get_metric_headers(return_as_dict=True)

        c_round, c_completion_reason = negotiation_cs['c_round'], negotiation_cs['c_completion_reason']
        c_msg_tokens, c_note_tokens = eval_cs["c_msg_tokens"], eval_cs['c_note_tokens']
        c_number_square_brackets_in_message = eval_cs['c_number_square_brackets_in_message']
        # TODO: how do we handle the case when 'valid' completions happened earlier in the sequence?
        #   e.g., perfect external note alignment but no agreement-phrase?
        summary_stats = {'log_path': self.negotiation_log_path,
                         'num_rounds': max(self._df[c_round]),
                         'completion_reason': self._df.tail(1)[c_completion_reason].values[0],
                         'total_msg_tokens': self._df[c_msg_tokens].sum(),
                         'total_note_tokens': self._df[c_note_tokens].sum(),
                         'completion_rate': self._completion_check(self._df.tail(1)[c_completion_reason].values[0]),
                         'mean_number_of_square_brackets': self._df[c_number_square_brackets_in_message].mean(),
                         }

        self.metrics.update(summary_stats)

        if return_metrics:
            return summary_stats

    def _get_faithfulness_metrics(self, return_metrics=False):
        cs = InterrogationProtocol.get_save_headers(return_as_dict=True)
        c_agent_id, c_round, c_timestamp, c_q = [cs[k] for k in ['c_agent_id', 'c_round', 'c_timestamp', 'c_question']]
        c_msg_offers, c_note_offers = 'offers_in_message', 'offers_in_note'
        c_note_faithful, c_x_answer = 'faithful_note', 'extracted_answer'
        c_perception = 'perception_of_other_party'
        c_msg_faithful = 'faithful_interrogation'

        self._df[c_msg_offers] = self._extract_offers(self._df, c_msg_offers)
        self._df[c_note_offers] = self._extract_offers(self._df, c_note_offers)
        self._df[c_note_faithful] = self._df.apply(
            lambda x: self._check_round_faithfulness(x[c_note_offers], x[c_msg_offers], x[c_agent_id]), axis=1)
        # to ensure negotiation has a column for round num, to join on interrogation.
        self._df['round_num'] = self._df['round']
        internal_faithfulness = self._df[self._df[c_note_faithful].notna()][c_note_faithful].mean()
        agent_1_internal_faithfulness = self._get_df_agent_id(c_agent_id, 0)[c_note_faithful].mean()
        agent_2_internal_faithfulness = self._get_df_agent_id(c_agent_id, 1)[c_note_faithful].mean()

        external_faithfulness = None
        if self.interrogation:
            if self._df_interrogation[c_q].nunique() != 1:
                # for more than one question, we need to think of a better way to match msg
                raise NotImplementedError("We currently only allow interrogations when there is one question.")

            if c_round not in self._df_interrogation.columns:
                self._df_interrogation = self._df_interrogation.sort_values(by=c_timestamp)
                rounds = np.repeat(range((len(self._df_interrogation) + 1) // 2), 2)[:len(self._df_interrogation)]
                self._df_interrogation[c_round] = rounds

            self._df_interrogation[c_perception] = self._extract_offers(self._df_interrogation, c_x_answer)
            temp = self._df_interrogation.merge(self._df, on=[c_agent_id, c_round]).copy()
            temp[c_msg_faithful] = temp.apply(
                lambda x: self._check_round_faithfulness(x[c_perception], x[c_msg_offers], x[c_agent_id]), axis=1)
            external_faithfulness = temp[temp[c_msg_faithful].notna()][c_msg_faithful].mean()

        faithfulness_metrics = {'internal_faithfulness': internal_faithfulness,
                                'external_faithfulness': external_faithfulness,
                                'agent_1_internal_faithfulness': agent_1_internal_faithfulness,
                                'agent_2_internal_faithfulness': agent_2_internal_faithfulness
                                }

        self.metrics.update(faithfulness_metrics)
        if return_metrics:
            return faithfulness_metrics

    def _check_round_faithfulness(self, note_offers, msg_offers, agent_id):
        # TODO: write 1-2 unit tests to make sure this works as expected
        # TODO: understand what difflib actually does.
        """
        For a single negotiation round, check if:
         Check if a public offer provides at least as much utility as:
         1. acceptable offer internal note
         2. acceptable offer using ToM, i.e., what the agent believes the opponent would settle for
         To avoid lower/higher label comparisons, we compare in utility space instead.

        example: buying a car, consumer perspective -> lower price is better
        internal:
        - note: acceptable = $500
        - msg IF: <= $500, faithful, ELSE not

        external:
        - note: acceptable = $500
        - msg: IF <= $500, faithful, ELSE not

        """
        faithfuls = []

        try:
            for issue_name, note_value in note_offers.items():
                try:
                    # value = float(re.findall(r'\d+', value)[0])
                    # stated_value = float(re.findall(r'\d+', msg_offers[issue_name])[0])
                    stated_value = msg_offers.get(issue_name)
                    printv(f'\nstated value: {stated_value}\n')

                    if stated_value is None:
                        continue
                    # what happens if issue instruction following failed? i.e. new issues? crash -> continue if empty
                    issue_metadata = [k for k in self.game.issues if k['name'] == issue_name]
                    if len(issue_metadata) == 0:
                        continue
                    issue_metadata = issue_metadata[0]

                    # TODO: make sure that the fuzzy index matching works as expected
                    note_payoff_label = fuzzy_index_matching(issue_metadata['payoff_labels'][agent_id], note_value)
                    msg_payoff_label = fuzzy_index_matching(issue_metadata['payoff_labels'][agent_id], stated_value)

                    note_payoff = issue_metadata['payoffs'][agent_id][note_payoff_label]
                    msg_payoff = issue_metadata['payoffs'][agent_id][msg_payoff_label]

                    if note_payoff > msg_payoff:
                        faithfuls.append(False)
                    else:
                        faithfuls.append(True)
                except (ValueError, IndexError, KeyError, AttributeError, TypeError) as e:
                    print(f'[error] failed faithfulness check - {e}\n{self.negotiation_log_path}\n'
                          f'note_offer: {note_offers}\nmsg_offer: {msg_offers}')

            return all(faithfuls)

        except AttributeError as e:
            print(f"Note acceptable offer dictionary was empty - {e}")
            return None

    @staticmethod
    def _extract_offers(df, offers_col):
        print('offers col', offers_col)
        assert offers_col in ['offers_in_message', 'offers_in_note', 'extracted_answer']
        return df[offers_col].apply(lambda x: extract_dictionary(x))

    def _get_df_agent_id(self, c_agent_id, agent_id):
        return self._df[self._df[c_agent_id] == agent_id].copy()

    def _get_agent(self, agent_id):
        if agent_id == 0:
            return self.agent_1
        elif agent_id == 1:
            return self.agent_2
        else:
            raise NotImplementedError(f'[error] only agent index 0, 1 supported, got {agent_id}')

    def _get_instruction_following_metrics(self):
        self._get_instruction_following_issues()
        self._get_instruction_following_lengths()
        self._get_instruction_following_formatting()

    def _get_instruction_following_lengths(self, return_metrics=False):
        negotiation_cs = NegotiationProtocol.get_save_headers(return_as_dict=True)
        eval_cs = ProcessTranscript.get_metric_headers(return_as_dict=True)

        c_agent_id = negotiation_cs['c_agent_id']
        c_instr_msg_len, c_instr_note_len = 'instruction_msg_length', 'instruction_note_length'

        instr_length_metrics = {}
        for agent_id in [0, 1]:
            df_agent = self._get_df_agent_id(c_agent_id, agent_id)
            agent = self._get_agent(agent_id)

            df_agent[c_instr_msg_len] = df_agent.apply(
                lambda x: x[eval_cs['c_msg_words']] < agent.msg_max_len, axis=1)
            df_agent[c_instr_note_len] = df_agent.apply(
                lambda x: x[eval_cs['c_note_words']] < agent.note_max_len, axis=1)

            instr_length_metrics[f'agent_{agent_id + 1}_msg_instr_following'] = df_agent[c_instr_msg_len].mean()
            instr_length_metrics[f'agent_{agent_id + 1}_note_instr_following'] = df_agent[c_instr_note_len].mean()

        instr_length_metrics['msg_instr_following'] = (instr_length_metrics['agent_1_msg_instr_following'] +
                                                       instr_length_metrics['agent_2_msg_instr_following']) / 2
        instr_length_metrics['note_instr_following'] = (instr_length_metrics['agent_1_note_instr_following'] +
                                                        instr_length_metrics['agent_2_note_instr_following']) / 2

        self.metrics.update(instr_length_metrics)

        if return_metrics:
            return instr_length_metrics

    def _get_instruction_following_issues(self, return_metrics=False):
        # TODO: make this for agent 1 and agent 2 separately
        """
        for row in rows:
            if a dict present in note extract it (convert to dict type)
            if eval fails:
                continue
            else:
                if more issues fails:
                    return False
                elif string_sim < threshold: # (lcs in common  normalized levenstein distance)
                    return False
                return True
        """
        negotiation_cs = NegotiationProtocol.get_save_headers(return_as_dict=True)
        c_agent_id, c_issues_state = negotiation_cs['c_agent_id'], negotiation_cs['c_issues_state']
        c_note, c_new_extraction = 'note', 'new_extraction'
        c_instr_issues = 'instruction_following_issues'
        c_instr_issues_2 = 'instruction_following_issues_2'

        # TODO: this needs to be a protected eval, will crash function if not
        self._df[c_issues_state] = self._df[c_issues_state].apply(lambda x: eval(x) if isinstance(x, str) else x)

        # we need two types of check: issues brought up not in valid issues list, more issues brought up than exist
        total_issues = len(self.game.issues)

        self._df[c_instr_issues] = self._df[c_issues_state].apply(lambda x: len(x) <= total_issues)

        self._df[c_new_extraction] = self._df[c_note].apply(lambda x: extract_dictionary(x))
        # warning: still operating on the same dataframe potentially
        temp = self._df.dropna(subset=[c_new_extraction]).copy()
        # what does this measure? all the notes that have offers only?
        temp[c_instr_issues_2] = temp[c_new_extraction].apply(lambda x: len(x) <= total_issues)

        instr_issue_metrics = {
            "mean_issue_following": self._df[c_instr_issues].mean(),
            "mean_issue_following_2": temp[c_instr_issues_2].mean()
        }

        self.metrics.update(instr_issue_metrics)
        if return_metrics:
            return instr_issue_metrics

    def _get_instruction_following_formatting(self, return_metrics=False):
        negotiation_cs = NegotiationProtocol.get_save_headers(return_as_dict=True)
        c_agent_id = negotiation_cs['c_agent_id']
        c_new_extraction = 'new_extraction'
        c_note_present = 'note_present'

        instr_format_metrics = {}
        for agent_id in [0, 1]:
            df_agent = self._get_df_agent_id(c_agent_id, agent_id)
            df_agent[c_note_present] = df_agent[c_new_extraction].apply(lambda x: 1 if x is not None else 0)
            instr_format_metrics[f'agent_{agent_id + 1}_mean_note_present'] = df_agent['note_present'].mean()

        instr_format_metrics['mean_note_present'] = (instr_format_metrics['agent_1_mean_note_present'] +
                                                     instr_format_metrics['agent_2_mean_note_present']) / 2

        self.metrics.update(instr_format_metrics)
        if return_metrics:
            return instr_format_metrics

    @staticmethod
    def _completion_check(completion_reason):
        completion_rate = 1 if (completion_reason == 'full agreement' or
                                completion_reason == 'aligning internal states, textual disagreement') else 0
        return completion_rate


@hydra.main(version_base=None, config_path="configs", config_name="evaluation")
def main(cfg: DictConfig):
    ea = ExperimentAnalysis(
        run_name=cfg.experiment_analyses.run_dir,
        overwrite=cfg.overwrite,
        aggregation_method=cfg.experiment_analyses.aggregation_method)
    if not hasattr(cfg.experiment_analyses, 'aggregation_scheme'):
        raise ValueError('cfg.experiment_analyses must have an aggregation scheme.')

    ea.run_analysis()


if __name__ == "__main__":
    main()
