from attr import define, field
from .abstract_eval import AbstractEval
import pandas as pd 

@dataclass 
class SelfPlayEval(AbstractEval):
    # how many points to sample in cross play
    sample_n: int = field(default=None)
    # limit games the agents by number of issues
    limit_issues: int = field(default=2)
    run_analysis: str = field(default="all")

    def run(self):
        self._preprocess()
        self._process()

    def _preprocess(self): 
        """preprocess the run to make it fit for cross-play eval"""
        if len(limit_models) > 0:
            df = self._limit_models(df, self.limit_models)

        if self.ensure_complete_runs:
            df = self._ensure_no_errors(df)
        self._restrict_issues(df, self.limit_issues, limit_type="equal")

        # limit to games where different models
        self.df = self._limit_to_self_play(df)
        self.completion = self.self.get_completed()

    def _process(self):
        self.df_processed = completion
        self.df_processed['agent_1_internal_description'] = self.df_processed['agent_1_internal_description'].apply(lambda x: 'Expert' if 'expert' in x else 'Awful' if 'awful' in x else 'No description')
        self.df_processed['agent_2_internal_description'] = self.df_processed['agent_2_internal_description'].apply(lambda x: 'Expert' if 'expert' in x else 'Awful' if 'awful' in x else 'No description')

        df_agent_1 = self.df_processed[['agent_1_internal_description', 'agent_2_internal_description', 'agent_1_normalized_payoff']].copy()
        df_agent_2 = self.df_processed[['agent_1_internal_description', 'agent_2_internal_description', 'agent_2_normalized_payoff']].copy()
        
        df_agent_1 = df_agent_1.rename({'agent_1_internal_description':'source', 'agent_2_internal_description':'target', 'agent_1_normalized_payoff':'payoff'}, axis=1)
        df_agent_2 = df_agent_2.rename({'agent_2_internal_description':'source', 'agent_1_internal_description':'target', 'agent_2_normalized_payoff':'payoff'}, axis=1)
        
        df = pd.concat([df_agent_1, df_agent_2], axis=0)
        
        mean = df.groupby(['source','target'])['payoff'].mean().reset_index().pivot(columns="target", index="source")
        std = df.groupby(['source','target'])['payoff'].std().reset_index().pivot(columns="target", index="source")
        count = df.groupby(['source','target'])['payoff'].count().reset_index().pivot(columns="target", index="source")


        print(self.add_stds(mean, std, count).to_latex(escape=False))


        mean = pd.DataFrame(df.groupby(['source'])['payoff'].mean())
        std = pd.DataFrame(df.groupby(['source'])['payoff'].std())
        count = pd.DataFrame(df.groupby(['source'])['payoff'].count())
        print(self.add_stds(mean, std, count).to_latex(escape=False))