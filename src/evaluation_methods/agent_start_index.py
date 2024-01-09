from attr import define, field
from .abstract_eval import AbstractEval
import pandas as pd 

@define
class SelfPlayEval(AbstractEval):
    # how many points to sample in cross play
    sample_n: int = field(default=None)
    # limit games the agents by number of issues
    limit_issues: int = field(default=2)
    run_analysis: str = field(default="all")

    def run(self):
        self._preprocess()
        self._process()

    def _preprocess(self, df):
        """preprocess the run to make it fit for cross-play eval"""
        if len(self.limit_models) > 0:
            df = self._limit_models(df, self.limit_models)

        if self.ensure_complete_runs:
            df = self._ensure_no_errors(df)

        # limit to games where different models
        self.df = self._limit_to_self_play(df)
        self.completion = self.self.get_completed()

    def _process(self)
        self.df_processed = self.completion
        agent_1_start = self.df_processed[self.df_processed['protocol_start_agent_index'] == 0]
        agent_2_start = self.df_processed[self.df_processed['protocol_start_agent_index'] == 1]

        start_payoff_1 = agent_1_start['agent_1_normalized_payoff'].mean()
        end_payoff_1 = agent_1_start['agent_2_normalized_payoff'].mean()

        start_payoff_2 = agent_2_start['agent_2_normalized_payoff'].mean()
        end_payoff_2 = agent_2_start['agent_1_normalized_payoff'].mean()
        
        print(f'start average payoff {(start_payoff_1 + start_payoff_2) /2}')
        print(f'end average payoff {(end_payoff_1 + end_payoff_2) /2}')
        