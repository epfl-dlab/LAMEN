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
import os
import numpy as np
import json
from datetime import datetime as dt


class Issue:
    def __init__(self, issue_name=None, issue_type=None, issue_descriptions=None, payoffs=None, num_steps=10,
                 step_descriptions=None, **kwargs):
        self.issue_name = issue_name
        self.issue_type = issue_type
        self.issue_descriptions = issue_descriptions
        self.payoffs = payoffs
        self.step_descriptions = step_descriptions

        # both sides want the same thing
        if issue_type == 'compatible':
            m1 = np.linspace(0, 1, num_steps)
            m2 = m1
        # it is worth more to one side than the other
        # NOTE: this is tricky when there are multiple issues in a single game, i.e., how does rescaling/weighing work?
        elif issue_type == 'integrative':
            m1 = np.linspace(0, 1, num_steps)
            m2 = np.flip(m1) * 0.5
        # sides want opposite things, e.g., +1 for me is -1 for you
        elif issue_type == 'distributive':
            m1 = np.linspace(0, 1, num_steps)
            m2 = np.flip(m1)
        # user-defined issue
        elif issue_type == 'custom':
            m1, m2 = payoffs
        else:
            raise NotImplemented(
                f'error: issue type {issue_type} not in [compatible, integrative, distributive, custom]')

        self.payoffs = [m1, m2]

    def to_dict(self):
        return vars(self)

    @staticmethod
    def from_dict(d):
        return Issue(**d)

    def save(self, fname):
        d = self.to_dict()
        with open(os.path.join(fname + '.json')) as f:
            json.dump(d, f)

    @staticmethod
    def load(fname):
        with open(fname) as f:
            issue = json.load(f)

        return Issue.from_dict(issue)


class Game:
    def __init__(self, name, description, issues, issue_weights, scale=(1, 1), sides=None, **kwargs):
        self.name = name
        self.description = description
        self.sides = sides
        self.issues = issues
        self.issue_weights = issue_weights
        self.scale = scale

    def reweigh_issues(self):
        # normalize weights to [0, 1]
        issue_weights = np.assarray(self.issue_weights) / np.sum(self.issue_weights, axis=1, keepdims=True)
        for issue, w in zip(self.issues, issue_weights.transpose()):
            payoffs = []
            for po, w_, s in zip(issue.payoffs, w, self.scale):
                po = po * w_ * s
                payoffs.append(po)
            issue.payoffs = payoffs

    def get_system_msg(self):
        pass

    def to_dict(self):
        return vars(self)

    @staticmethod
    def from_dict(d):
        return Game(**d)


def load_game(game_name: str) -> dict:
    pass


def create_experiment():
    pass
