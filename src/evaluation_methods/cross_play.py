from attr import define, field
from .abstract_eval import AbstractEval
import pandas as pd
import os


@define
class CrossPlayEval(AbstractEval):
    # how many points to sample in cross play
    sample_n: int = field(default=50)
    # limit games the agents by number of issues
    limit_issues: int = field(default=2)
    run_analysis: str = field(default="all")
    # pre-upsample df
    _original: pd.DataFrame = None
    _completed: pd.DataFrame = None
    _completed_full: pd.DataFrame = None

    def run(self):
        self._preprocess()

        if self.run_analysis == "all":
            self._meta_performance()
            self._model_performance_summary()
            self._head_to_head()
            self._game_type_breakdown()
            self._model_completion_and_count()

    def _preprocess(self, agreement_types=['full agreement',
                                           'aligning internal states, textual disagreement']):
        # filter out data not useful
        # upsample to balance all sub-bias groups
        # duplicate and reorder to have both role results on same side (source -> target)
        # add winners column
        target_cols = ['internal_faithfulness', 'note_instr_following', 'msg_instr_following',
                       'mean_note_present', 'completion_rate', 'num_rounds', 'payoff', 'target_payoff',
                       'full_completion_rate']

        self._df['completion_rate'] = self._df["completion_reason"].apply(lambda x: 1 if x in agreement_types else 0)
        self._df = self._zero_out_payoffs(self._df)
        self._df['full_completion_rate'] = self._df["completion_reason"].apply(
            lambda x: 1 if x in ['full agreement'] else 0)

        _df = self._df
        """preprocess the run to make it fit for cross-play eval"""
        self.limit_models = ['command-light']
        if len(self.limit_models) > 0:
            _df = self._limit_models(_df, self.limit_models)

        if self.ensure_complete_runs:
            _df = self._ensure_no_errors(_df)
        # limit to games where different models
        _df = self._limit_to_cross_play(_df)
        _df = self.remap_names(_df)

        # drop for renaming later
        _df.drop(['internal_faithfulness', 'note_instr_following',
                  'msg_instr_following', 'mean_note_present'], axis=1, inplace=True)
        # limit number of issues
        self._restrict_issues(_df, self.limit_issues, limit_type="less than")

        # TODO: discuss if this is a better sampling strategy.
        #  original, _df = self._sample_games(_df)

        # rename columns to source target pairs
        model_mappings = {'AGENT_model_name': 'source',
                          'OTHER_model_name': 'target',
                          'AGENT_normalized_payoff': 'payoff',
                          'OTHER_normalized_payoff': 'target_payoff',
                          'AGENT_internal_faithfulness': 'internal_faithfulness',
                          'AGENT_note_instr_following': 'note_instr_following',
                          'AGENT_msg_instr_following': 'msg_instr_following',
                          'AGENT_mean_note_present': 'mean_note_present'}

        _df_agent_1 = self._column_mapping(_df.copy(), model_mappings, 1)
        _df_agent_2 = self._column_mapping(_df.copy(), model_mappings, 2)

        _df = pd.concat([_df_agent_1, _df_agent_2], axis=0)

        self._df = _df
        self._completed = self._debias_dataframe(self.get_completed(), target_cols=target_cols)
        self._completed_full = self._debias_dataframe(self.get_completed(
            agreement_types=['full agreement']), target_cols=target_cols)

        _df = self._debias_dataframe(_df, target_cols=target_cols)

        # _df_agent_1 = self._column_mapping(original.copy(), model_mappings, 1)
        # _df_agent_2 = self._column_mapping(original.copy(), model_mappings, 2)

        # self._original = pd.concat([_df_agent_1, _df_agent_2], axis=0)

        print("_df.groupby(['source', 'target', 'game_type'])['game_type'].count()")
        print(_df.groupby(['source', 'target', 'game_type'])['game_type'].count())

        _df['winner'] = _df.apply(lambda x: None if x['completion_rate'] == 0
        else 1 if x['payoff'] > x['target_payoff']
        else 0.5 if x['payoff'] == x['target_payoff']
        else 0, axis=1)
        self._df = _df

    def _meta_performance(self, fname='meta_performance.tex'):
        print('Calculating meta performance.')
        # output
        # latex table of instruction-following, faithfulness, number-rounds, completion-rate
        _df_one_model = self._df.copy()
        _df_one_model.rename({'source': 'Model name', 'num_issues': 'Issues'})

        ablations_1 = ['source']
        metrics_1 = ['internal_faithfulness', 'note_instr_following', 'msg_instr_following',
                     'mean_note_present', 'completion_rate', 'full_completion_rate', 'num_rounds']

        # game_name / game_type / num_issues

        means = _df_one_model.groupby(ablations_1)[metrics_1].mean().round(2)
        stds = _df_one_model.groupby(ablations_1)[metrics_1].std().round(2)
        count = _df_one_model.groupby(ablations_1)[metrics_1].count().round(2)
        print(means)
        self.add_stds(means, stds, count).to_latex(os.path.join(self.save_path, fname), escape=False)

    def _debias_dataframe(self, _df, target_cols):
        debiasing_cols = ['source', 'target', 'protocol_start_agent_index', 'game_type', 'num_issues', 'game_class']
        _df = _df.groupby(debiasing_cols)[target_cols].mean().reset_index()
        return _df

    def _sample_games(self, _df, print_pre_counts=True):
        # 0. groupby (game_name +) game-type + num_issues
        # 1. groupby agent models
        # 2. groupby agent start index
        # --> should have a 4-way split per model_name tuple
        # --> upsample each group
        # --> average each group        print(f'Sampling {self.sample_n} games.')
        debiasing_cols = ['agent_1_model_name', 'agent_2_model_name',
                          'game_type', 'num_issues', 'protocol_start_agent_index']
        for_print = ", ".join(debiasing_cols)

        if print_pre_counts:
            print(f"_df.groupby({for_print})['game_type'].count()")
            print(_df.groupby(debiasing_cols)['game_type'].count())
            print('blah')
            print(_df.groupby(['agent_1_model_name', 'game_type', 'num_issues'])['game_type'].count())
        original = _df.copy()
        _df = _df.groupby(debiasing_cols).sample(self.sample_n, replace=True)
        return original, _df

    def _head_to_head(self, fname='{}_average_payoff.tex'):
        print('overall head to head')
        _df = self._df.copy()
        # groupings game_name / game_type / num_issues
        for completion_type in ['full', 'partial']:
            if completion_type == 'partial':
                _df = self._completed
            else:
                _df = self._completed_full

            mean = _df.groupby(['source', 'target'])[['payoff']].mean() \
                .reset_index().pivot(index='source', columns=['target'])
            std = _df.groupby(['source', 'target'])[['payoff']].std() \
                .reset_index().pivot(index='source', columns=['target'])
            count = _df.groupby(['source', 'target'])[['payoff']].count() \
                .reset_index().pivot(index='source', columns=['target'])

            with_stds = self.add_stds(mean, std, count)
            # format columns properly
            with_stds.columns = with_stds.columns.swaplevel(0, 1)
            with_stds.sort_index(axis=1, level=0, inplace=True)
            # print latex for paper
            with_stds.to_latex(os.path.join(self.save_path, fname.format(completion_type)), escape=False)

    def _model_completion_and_count(self, fname='completion_rate.tex'):
        print('model performance summary')

        _df = self._df.copy(deep=True)
        _df = _df.groupby(['source', 'game_class', 'num_issues'])['completion_rate'].transform('count')
        g1 = _df.groupby(['source', 'game_class', 'num_issues'])[['completion_rate']].mean().reset_index().pivot(
            index='source', columns=['num_issues', 'game_class'])
        g2 = _df.groupby(['source', 'game_class', 'num_issues'])[['completion_rate']].std().reset_index().pivot(
            index='source', columns=['num_issues', 'game_class'])
        g3 = _df.groupby(['source', 'game_class', 'num_issues'])[['completion_rate']].count().reset_index().pivot(
            index='source', columns=['num_issues', 'game_class'])

        with_stds = self.add_stds(g1, g2, g3)
        with_stds.columns = with_stds.columns.swaplevel(0, 1)
        with_stds.columns = with_stds.columns.swaplevel(1, 2)

        with_stds.sort_index(axis=1, level=0, inplace=True)
        with_stds.to_latex(os.path.join(self.save_path, fname), escape=False)
        original = self._original.groupby(['source', 'game_class', 'num_issues'])[
            ['completion_rate']].count().reset_index().pivot(
            index='source', columns=['num_issues', 'game_class'])
        original.to_latex(os.path.join(self.save_path, 'number_games.tex'))
        print(_df.groupby(['source', 'game_class', 'num_issues'])['winner'].count().reset_index().pivot(
            index='source', columns=['num_issues', 'game_class']))

    def _model_performance_summary(self, fname='payoff_win_rate_{}.tex'):
        #
        print('model performance summary')
        for i in range(1, 3):
            _df = self._df.copy(deep=True)
            completed = self._completed
            _df = _df[_df['num_issues'] == i]
            completed = completed[completed['num_issues'] == i]
            # _df = self.get_completed()

            g1 = _df.groupby(['source', 'game_class', 'num_issues'])[['completion_rate', 'payoff']].mean().reset_index()
            # g1 = _df.groupby(['source', 'game_class', 'num_issues'])[
            #             ['winner', 'payoff']].mean().reset_index().pivot(
            #             index='source', columns=['num_issues', 'game_class'])
            g2 = _df.groupby(['source', 'game_class', 'num_issues'])[['completion_rate', 'payoff']].std().reset_index()
            g3 = _df.groupby(['source', 'game_class', 'num_issues'])[
                ['completion_rate', 'payoff']].count().reset_index()

            c1 = completed.groupby(['source', 'game_class', 'num_issues'])[['payoff']].mean().reset_index()
            # g1 = _df.groupby(['source', 'game_class', 'num_issues'])[
            #             ['winner', 'payoff']].mean().reset_index().pivot(
            #             index='source', columns=['num_issues', 'game_class'])
            c2 = completed.groupby(['source', 'game_class', 'num_issues'])[['payoff']].std().reset_index()
            c3 = completed.groupby(['source', 'game_class', 'num_issues'])[['payoff']].count().reset_index()

            g1['u_star'] = c1['payoff']
            g2['u_star'] = c2['payoff']
            g3['u_star'] = c3['payoff']

            g1 = g1.pivot(index='source', columns=['num_issues', 'game_class'])
            g2 = g2.pivot(index='source', columns=['num_issues', 'game_class'])
            g3 = g3.pivot(index='source', columns=['num_issues', 'game_class'])

            with_stds = self.add_stds(g1, g2, g3)
            with_stds.columns = with_stds.columns.swaplevel(0, 1)
            with_stds.columns = with_stds.columns.swaplevel(1, 2)

            with_stds.sort_index(axis=1, level=0, inplace=True)
            with_stds.to_latex(os.path.join(self.save_path, fname.format(i)), escape=False)
            print(_df.groupby(['source', 'game_class', 'num_issues'])['winner'].count().reset_index().pivot(
                index='source', columns=['num_issues', 'game_class']))
        # print(_df.groupby(['source', 'game_class'])['winner'].count())

        # print(_df.groupby(['game_class', 'source', 'target'])['payoff'].mean().reset_index().pivot(
        #     columns=['game_class', 'source'], index=['target']).to_latex())

    def _game_type_breakdown(self):
        """
        For comparing how the different models perform in the cooperative vs. competitive setting
        Figure in the appendix of paper
                              | model 1 | model 2 | ... |
        competitive | model 1 | payoff  | 
        ... 
        cooperative | model 1 |
        cooperative | model 2 |
        ...
        """
        print('Game type breakdown')

        _df = self._df

        _completed = self._completed
        for completion_type in ['payoff', 'full', 'partial']:
            if completion_type == 'partial':
                _completed = self._df
                metric = 'completion_rate'
            elif completion_type == 'full':
                _completed = self._df
                metric = 'full_completion_rate'
            else:
                _completed = self._completed
                metric = 'payoff'
            print("\n\nFor competitive games:\n")
            _df_1 = _completed[(_completed["game_class"] == "competitive")]
            g1 = _df_1.groupby(['source', 'target', 'game_class'])[[metric]].mean().reset_index().pivot(
                index='source', columns=['game_class', 'target'])
            g2 = _df_1.groupby(['source', 'target', 'game_class'])[[metric]].std().reset_index().pivot(
                index='source', columns=['game_class', 'target'])
            g3 = _df_1.groupby(['source', 'target', 'game_class'])[[metric]].count().reset_index().pivot(
                index='source', columns=['game_class', 'target'])

            with_stds = self.add_stds(g1, g2, g3)
            with_stds.columns = with_stds.columns.swaplevel(0, 1)

            with_stds.sort_index(axis=1, level=0, inplace=True)
            with_stds.to_latex(os.path.join(self.save_path, f'{completion_type}_competitive_head2head.tex'),
                               escape=False)

            print("\n\nFor cooperative games:\n")
            _df_2 = _completed[(_completed["game_class"] == "cooperative")]
            g1 = _df_2.groupby(['source', 'target', 'game_class'])[[metric]].mean().reset_index().pivot(
                index='source', columns=['game_class', 'target'])
            g2 = _df_2.groupby(['source', 'target', 'game_class'])[[metric]].std().reset_index().pivot(
                index='source', columns=['game_class', 'target'])
            g3 = _df_2.groupby(['source', 'target', 'game_class'])[[metric]].count().reset_index().pivot(
                index='source', columns=['game_class', 'target'])

            with_stds = self.add_stds(g1, g2, g3)
            with_stds.columns = with_stds.columns.swaplevel(0, 1)

            with_stds.sort_index(axis=1, level=0, inplace=True)
            with_stds.to_latex(os.path.join(self.save_path, f'{completion_type}_cooperative_head2head.tex'),
                               escape=False)
