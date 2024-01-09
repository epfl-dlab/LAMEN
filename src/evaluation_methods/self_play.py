from attr import define, field
import pandas as pd
import os

from .abstract_eval import AbstractEval


@define
class SelfPlayEval(AbstractEval):
    sample_n: int = 250
    # limit games the agents by number of issues
    limit_issues: int = field(default=2)
    run_analysis: str = field(default="all")

    _completion: pd.DataFrame = field(default=None)

    def run(self):
        self._preprocess()

        if self.run_analysis == "all":
            self._meta_performance()
            self._completion_payoff_by_game_type()
            
            for num_issues in range(1,3): 
                for integrative in [True, False]:
                    self._main_table(num_issues=num_issues, integrative=integrative)


    def _preprocess(self, agreement_types=['full agreement', 'aligning internal states, textual disagreement']): #'aligning internal states, textual disagreement', 
        # metrics we use. we take a mean over these during debiasing.
        target_cols = ['internal_faithfulness', 'external_faithfulness', 'note_instr_following', 'msg_instr_following',
                     'mean_note_present', 'completion_rate', 'num_rounds', 'agent_1_normalized_payoff', 
                     'agent_2_normalized_payoff','full_completion_rate']
        
        # TODO: COME BACK AND REMOVE ONCE WE DECIDE ON COMPLETION REASON
        self._df['completion_rate'] = self._df["completion_reason"].apply(lambda x: 1 if x in agreement_types else 0)
        self._df = self._zero_out_payoffs(self._df)
        self._df['full_completion_rate'] = self._df["completion_reason"].apply(lambda x: 1 if x in ['full agreement'] else 0)


        _df = self._df
        """preprocess the run to make it fit for cross-play eval"""
        if len(self.limit_models) > 0:
            _df = self._limit_models(_df, self.limit_models)
        if self.ensure_complete_runs:
            _df = self._ensure_no_errors(_df)

        # limit to games where same models
        self._df = self._limit_self_play(_df)
        self._df = self.remap_names(self._df)

        # _, _df = self._sample_games(self._df)
        self._completion = self._debias_dataframe(self.get_completed(), target_cols=target_cols)
        self._df = self._debias_dataframe(self._df, target_cols=target_cols)

        # _, self._completion = self._sample_games(self._completion)
        
    def _debias_dataframe(self, _df, target_cols):
        debiasing_cols = ['agent_1_model_name', 'protocol_start_agent_index', 'game_type', 'num_issues']
        _df = _df.groupby(debiasing_cols)[target_cols].mean().reset_index()
        return _df
        
    def _sample_games(self, _df, print_pre_counts=True):
        # 0. groupby (game_name +) game-type + num_issues
        # 2. groupby agent start index
        # --> upsample each group
        # --> average each group        print(f'Sampling {self.sample_n} games.')
        debiasing_cols = ['agent_1_model_name', 'protocol_start_agent_index', 'game_type', 'num_issues']
        for_print = ", ".join(debiasing_cols)

        if print_pre_counts:
            print(f"_df.groupby({for_print})['game_type'].count()")
            print(_df.groupby(debiasing_cols)['game_type'].count())
            print(_df.groupby(['agent_1_model_name', 'game_type', 'num_issues'])['game_type'].count())
        original = _df.copy()
        _df = _df.groupby(debiasing_cols).sample(self.sample_n, replace=True)
        return original, _df

    def _meta_performance(self):
        """Performance on faithfulness, instruction following, ..."""
        print("\nMeta performance:\n")
        df_one_model = self._df.copy()
        df_one_model.rename({'agent_1_model_name': 'Model name', 'num_issues': 'Issues'})

        ablations_1 = ['agent_1_model_name']
        metrics_1 = ['internal_faithfulness', 'external_faithfulness', 'note_instr_following', 'msg_instr_following',
                     'mean_note_present', 'completion_rate', 'full_completion_rate', 'num_rounds']

        means = df_one_model.groupby(ablations_1)[metrics_1].mean().round(2)
        stds = df_one_model.groupby(ablations_1)[metrics_1].std().round(2)
        count = df_one_model.groupby(ablations_1)[metrics_1].count().round(2)

        self.add_stds(means, stds, count).to_latex(os.path.join(
                        self.save_path, 'self_play_meta_performance.tex'), escape=False)
        print("Count:")
        print(df_one_model.groupby(ablations_1)[metrics_1].count().round(2).to_latex())

    def _completion_payoff_by_game_type(self):
        df_one_model = self._df.copy()
        completion = self._completion.copy()
        completion['total_payoff'] = (completion['agent_1_normalized_payoff'] + completion['agent_2_normalized_payoff']) / 2
        df_one_model['total_payoff'] = (df_one_model['agent_1_normalized_payoff'] + df_one_model['agent_2_normalized_payoff']) / 2
        print(completion)
        ablation_2 = ['agent_1_model_name', 'game_type']
        metrics = ['completion_rate']
        metrics_c = ['total_payoff']

        mean_payoff = df_one_model.groupby(ablation_2)[metrics].mean().reset_index()
        print("len(mean_payoff)", len(mean_payoff))
        completed_mean = completion.groupby(ablation_2)[metrics_c].mean().reset_index().rename(
            {'total_payoff': 'total_payoff_c'}, axis=1)
        std_payoff = df_one_model.groupby(ablation_2)[metrics].std().reset_index()
        completed_std = completion.groupby(ablation_2)[metrics_c].std().reset_index().rename(
            {'total_payoff': 'total_payoff_c'}, axis=1)
        count_payoff = df_one_model.groupby(ablation_2)[metrics].count().reset_index()
        completed_count = completion.groupby(ablation_2)[metrics_c].count().reset_index().rename(
            {'total_payoff': 'total_payoff_c'}, axis=1)
        # merge
        mean_payoff = mean_payoff.merge(completed_mean, how='left', on=ablation_2)
        print("len(mean_payoff)", len(mean_payoff))

        std_payoff = std_payoff.merge(completed_std, how='left', on=ablation_2)
        count_payoff = count_payoff.merge(completed_count, how='left', on=ablation_2)

        mean_payoff = mean_payoff.pivot(index=['agent_1_model_name'], columns=['game_type'],
                                        values=['total_payoff_c', 'completion_rate']).round(2)
        count_payoff = count_payoff.pivot(index=['agent_1_model_name'], columns=['game_type'],
                                          values=['total_payoff_c', 'completion_rate']).round(2)
        std_payoff = std_payoff.pivot(index=['agent_1_model_name'], columns=['game_type'],
                                      values=['total_payoff_c', 'completion_rate']).round(2)

        f = self.add_stds(mean_payoff, std_payoff, count_payoff)
        f.columns = f.columns.swaplevel(0, 1)
        f.sort_index(axis=1, level=[0, 1], ascending=[False, True], inplace=True)

        f.to_latex(os.path.join(self.save_path, 'self_play_payoffs_by_game.tex'), escape=False)

    def _multi_issue(self):
        print('Multi issue still needs to be implemented.')

    def _main_table(self, num_issues, integrative):
        one_issue = self._df[(self._df['num_issues'] == num_issues)].copy()
        completion = self._completion.copy()
        completion = completion[completion['num_issues'] == num_issues].copy()
        if integrative:
            one_issue = one_issue[~one_issue['game_type'].str.contains('non-')]
            completion = completion[~completion['game_type'].str.contains('non-')]
        else: 
            one_issue = one_issue[one_issue['game_type'].str.contains('non-')]
            completion = completion[completion['game_type'].str.contains('non-')]

        print(completion.head())
        ablations = ['agent_1_model_name', 'game_type']
        metrics = ['completion_rate', 'total_payoff']

        one_issue['total_payoff'] = (one_issue['agent_1_normalized_payoff'] + one_issue['agent_2_normalized_payoff']) / 2
        completion['total_payoff'] = (completion['agent_1_normalized_payoff'] + completion['agent_2_normalized_payoff']) / 2

        g1 = one_issue.groupby(ablations)[metrics].mean().round(2)
        print(g1)
        g1_count = one_issue.groupby(ablations)[metrics].count().round(2)
        print(g1_count)
        g1_std = one_issue.groupby(ablations)[metrics].std().round(2)

        ablations = ['agent_1_model_name', 'game_type']
        metrics = ['total_payoff']

        g2 = completion.groupby(ablations)[metrics].mean().round(2)
        print(g2)

        g2_std = completion.groupby(ablations)[metrics].std().round(2)
        g2_count = completion.groupby(ablations)[metrics].count().round(2)

        g1['total_payoff_c'] = g2['total_payoff']
        g1_std['total_payoff_c'] = g2_std['total_payoff']
        g1_count['total_payoff_c'] = g2_count['total_payoff']

        g1.reset_index(inplace=True)
        g1_std.reset_index(inplace=True)
        g1_count.reset_index(inplace=True)

        f = g1.pivot(index='agent_1_model_name', columns='game_type',
                     values=['total_payoff', 'completion_rate', 'total_payoff_c'])
        f_std = g1_std.pivot(index='agent_1_model_name', columns='game_type',
                             values=['total_payoff', 'completion_rate', 'total_payoff_c'])
        f_count = g1_count.pivot(index='agent_1_model_name', columns='game_type',
                                 values=['total_payoff', 'completion_rate', 'total_payoff_c'])

        f.columns = f.columns.swaplevel(0, 1)
        # f.sort_index(axis=1, level=0, inplace=True)
        f_std.columns = f_std.columns.swaplevel(0, 1)
        f_std.sort_index(axis=1, level=0, inplace=True)
        f_count.columns = f_count.columns.swaplevel(0, 1)
        f_count.sort_index(axis=1, level=0, inplace=True, ascending=False)

        print(f_count)
        dfff = self.add_stds(f, f_std, f_count)
        dfff = dfff.sort_index(axis=1, level=[0,1], ascending=[False, True])
        dfff.to_latex(os.path.join(
                        self.save_path, f"self_play_{num_issues}_issue_{'integrative' if integrative else 'non_integrative'}.tex"), escape=False)
