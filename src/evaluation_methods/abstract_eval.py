from attr import define, field
from abc import ABC
import pandas as pd
import numpy as np

model_mapping = {
    'gpt-4': '\\four{}$^*$',
    'gpt-3.5-turbo': '\\turbo{}',
    'chat-bison': '\\bison{}',
    'command': '\\cohere{}',
    'command-light': '\\coherelight{}$^\dagger$',
    'claude-2': '\\claude{}',
}


@define
class AbstractEval(ABC):
    _df: pd.DataFrame
    save_path: str

    limit_models: list = field(factory=list)
    ensure_complete_runs: bool = field(default=True)

    def get_completed(self, agreement_types=['full agreement',
                                             'aligning internal states, textual disagreement']):  # 'aligning internal states, textual disagreement',
        return self._df[self._df["completion_reason"].isin(agreement_types)]

    @staticmethod
    def remap_names(_df):
        _df['agent_1_model_name'] = _df['agent_1_model_name'].apply(lambda x: model_mapping[x])
        _df['agent_2_model_name'] = _df['agent_2_model_name'].apply(lambda x: model_mapping[x])
        return _df

    @staticmethod
    def _limit_models(_df, limit_models):
        # limit to models in the list
        _df = _df[~((_df['agent_1_model_name'].isin(limit_models)) |
                    (_df['agent_2_model_name'].isin(limit_models)))]
        return _df

    @staticmethod
    def _limit_to_cross_play(_df):
        _df = _df[_df['agent_1_model_name'] != _df['agent_2_model_name']]
        return _df

    @staticmethod
    def _limit_self_play(_df):
        _df = _df[_df['agent_1_model_name'] == _df['agent_2_model_name']]
        return _df

    @staticmethod
    def _restrict_issues(_df, num_issues, limit_type="equal"):
        if limit_type == "equal":
            return _df[_df["num_issues"] == num_issues]
        elif limit_type == "less than":
            return _df[_df["num_issues"] <= num_issues]
        elif limit_type == "greater than":
            return _df[_df["num_issues"] >= num_issues]
        else:
            raise NotImplementedError(
                f"Limit type must be 'equal', 'less than', 'greater than' - selected '{limit_type}'")

    @staticmethod
    def _column_mapping(_df, mapping: dict, agent_id: int):
        other_agent_id = 2 if agent_id == 1 else 1
        mapping = {key.replace("AGENT", f"agent_{agent_id}").replace("OTHER", f"agent_{other_agent_id}"): val
                   for key, val in mapping.items()}
        _df = _df.rename(mapping, axis=1)
        return _df

    @staticmethod
    def _ensure_no_errors(df, agreement_reason=['full agreement', 'aligning internal states, textual disagreement']):
        # TODO: this may be problematic if they think they have interests opposite to what they have.
        # TODO: should it be full agreement or synthetic?
        epsilon = 1e-6
        error_1 = ((df['agent_1_normalized_payoff'] == 0) & (
                df['agent_2_normalized_payoff'] == 0) & (df["completion_reason"].isin(agreement_reason)))
        error_2 = ((1 - df['agent_1_normalized_payoff'] - df['agent_2_normalized_payoff'] > epsilon) &
                   (df["completion_reason"].isin(agreement_reason)) & (
                               df['game_type'] == 'non-integrative distributive'))
        error_3 = ((df['agent_1_normalized_payoff'] - df['agent_2_normalized_payoff'] > epsilon) &
                   (df["completion_reason"].isin(agreement_reason)) & (
                               df['game_type'] == 'non-integrative compatible') &
                   (df['num_issues'] == 1))
        df = df[~error_1 & ~error_2 & ~error_3]
        return df

    def add_stds(self, mean, std, count):
        def convert_latex(val):
            if val == '--':
                return val
            else:
                return " \std{{{}}}".format('{0:.2f}'.format(val))

        def convert_mean(col):
            max_val = max(col)

            formatted_list = []
            all_equal = []

            for v in col.values:
                print("v", v)
                if v == max_val:
                    all_equal.append(True)
                else:
                    all_equal.append(False)
            all_equal = all(all_equal)
            for v in col.values:
                if np.isnan(v):
                    formatted_list.append('--')
                elif (v == max_val) and (not all_equal):
                    formatted_list.append("\\textbf{{{}}}".format('{0:.2f}'.format(v)))
                else:
                    formatted_list.append('{0:.2f}'.format(v))
            return [k.replace('nan', '--') for k in formatted_list]

        mean = mean.round(2)
        for col in mean.columns:
            mean[col] = convert_mean(mean[col])
        std = std / count ** .5
        std = std.round(2).applymap(convert_latex)
        return mean + std

    @staticmethod
    def _zero_out_payoffs(df):
        # if the final round does not converge, then set the payoff to zero
        df['agent_1_normalized_payoff'] = df.apply(
            lambda x: 0 if x['completion_rate'] == 0 else x['agent_1_normalized_payoff'], axis=1)
        df['agent_2_normalized_payoff'] = df.apply(
            lambda x: 0 if x['completion_rate'] == 0 else x['agent_2_normalized_payoff'], axis=1)
        return df
