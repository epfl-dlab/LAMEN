""" Methods to create two-player games.

From Michal email, support following issues types:
1. Compatible: (both sides want the same thing) and max it out.
2. Integrative: (it is worth more for one side than to the other) and max it out.
3. Distributive ($1 loss for you is $1 gain for me) and distribute them.

- A 'game' consists of one or more 'issues'
- Each 'issue' is a two-way payoff matrix with meta details, e.g., name, type
- For now, we assume payoff matrices for the same issue have the same number of options

Games are saved in .json files:
- creator: (str)
- date: (str), yyyymmdd_hhmmss
- description: (str)
- role_1: (str)
- role_2: (str)
- issue_importance: list
- issues: [issue_0, issue_1, ..., issue_k]

Each issue is a dictionary of format:
- name:
- description_1:
- description_2:
- payoff_1:
- payoff_2:
"""
import numpy
import numpy as np
import json


def create_issue(issue_name: str, issue_type: str, num_steps: int, scale=(1, 1), issue_descriptions=None,
                 step_descriptions=None) -> dict:
    """

    :param issue_name:
    :param issue_type:
    :param num_steps:
    :param scale:
    :param step_descriptions:
    :param issue_descriptions:
    :return:
    """
    issue, po1, po2 = {}, {}, {}
    if issue_type == 'compatible':
        pass
    elif issue_type == 'integrative':
        pass
    elif issue_type == 'distributive':
        pass
    elif issue_type == 'custom':
        pass
    else:
        raise NotImplemented(f'error: issue type {issue_type} not in [compatible, integrative, distributive, custom]')

    return issue


def create_game(issue_types: list, issue_importance: list, issue_names=None, issue_descriptions=None,
                issue_step_descriptions=None, num_steps=10, scale=(1, 1), save_path=None) -> dict:
    """
    Create a game consisting of one or more issues
    :param issue_types:
    :param issue_importance: list of tuples
    :param issue_names:
    :param issue_descriptions:
    :param issue_step_descriptions:
    :param num_steps:
    :param scale:
    :param save_path:
    :return:
    """
    # TODO: some checks on lengths, types, etc.

    if issue_names is None:
        issue_names = [f'issue {i}' for i in range(len(issue_types))]
    if isinstance(num_steps, int):
        num_steps = [num_steps] * len(issue_types)

    issue_importance = np.asarray(issue_importance) / sum(issue_importance)

    issues = []
    for name, it, ns, gi, i_desc, i_s_desc in zip(issue_names, issue_types, num_steps, issue_importance,
                                                  issue_descriptions, issue_step_descriptions):
        issue = create_issue(issue_name=name, issue_type=it, num_steps=ns, issue_descriptions=i_desc,
                             step_descriptions=i_s_desc, scale=(gi[0] * scale[0], gi[1] * scale[1]))
        issues.append(issue)

    game = {}

    if save_path is not None:
        pass

    return game


def load_game(game_name: str) -> dict:
    pass
