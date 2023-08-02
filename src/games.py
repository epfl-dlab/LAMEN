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
from utils import read_json


class Issue:
    def __init__(self, name, issue_type='custom', descriptions=None, payoffs=None, num_steps=10,
                 payoff_labels=None, **kwargs):
        self.name = name
        self.issue_type = issue_type
        self.descriptions = descriptions
        self.payoffs = payoffs
        self.payoff_labels = payoff_labels

        # both sides want the same thing
        # TODO: venia overwrote all of these linspaces. they don't work.
        if issue_type == 'compatible':
            m1 = np.linspace(0, 1, num_steps)
            m2 = m1
        # sides want opposite things, e.g., +1 for me is -1 for you
        elif issue_type == 'distributive':
            m1 = np.linspace(0, 1, num_steps)
            m2 = np.flip(m1)
        # it is worth more to one side than the other
        # NOTE: this is tricky when there are multiple issues in a single game, i.e., how does rescaling/weighing work?
        elif issue_type == 'integrative':
            m1 = np.linspace(0, 1, num_steps)
            m2 = np.flip(m1) * 0.5
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

    def split_dict_by_index(self, category):
        if category in ["payoffs", "payoff_labels", "descriptions"]:
            dict_0 = {k: v[0] if isinstance(v, list) else v for k, v in category.items()}
            dict_1 = {k: v[1] if isinstance(v, list) else v for k, v in category.items()}
            return dict_0, dict_1
        else:
            raise NotImplementedError(f"Category not in payoffs, payoff_labels, descriptions")
    
    def format_issue(self, idx):
        issue_format = self.name + "\n"
        for label, payoff in zip(self.payoff_labels[idx], self.payoffs[idx]):
            issue_format += f"{label}, {payoff}\n"
        return issue_format

    def __repr__(self):
        s = ''
        lj = 20
        for k, v in zip(['issue', 'type', 'payoffs', 'payoff_labels', 'descriptions'],
                        [self.name, self.issue_type, self.payoffs, self.payoff_labels, self.descriptions]):

            s += f'{k}'.ljust(lj) + f'{v}\n'
        return s


class Game:
    def __init__(self, name, description, issues, issue_weights, scale=(1, 1), sides=None, **kwargs):
        self.name = name
        self.description = description
        self.sides = sides
        self.issues = issues
        self.load_issues()  # load in the issues in correct format
        
        self.issue_weights = issue_weights
        self.scale = scale

        # scale issues to their importance
        self.reweigh_issues()

    def reweigh_issues(self):
        # normalize weights to [0, 1]
        issue_weights = np.asarray(self.issue_weights) / np.sum(self.issue_weights, axis=1, keepdims=True)
        for issue, w in zip(self.issues, issue_weights.transpose()):
            payoffs = []
            for po, w_, s in zip(issue.payoffs, w, self.scale):
                po = (np.asarray(po) * w_ * s).astype(int)  # integer values only, TODO: make sure rounding is sensible!
                payoffs.append(po)
            issue.payoffs = payoffs

    def get_system_msg(self):
        pass
    
    def load_issues(self, issues_path="data/issues/"):
        issues = []
        for issue in self.issues:
            if isinstance(issue, str):
                fname = os.path.join(issues_path, issue + '.json')
                issues.append(Issue.load(fname))

        if len(issues) > 0:
            self.issues = issues
        
    def to_dict(self):
        return vars(self)

    def format_all_issues(self, agent_idx):
        issues_text = ""
        for issue in self.issues:
            issues_text += f"{issue.format_issue(agent_idx)}\n"
            
        return issues_text.strip()
    
    @staticmethod
    def from_dict(d):
        return Game(**d)

    def __str__(self):
        return f"Game: {self.name}"


def load_game(game_path: str) -> dict:
    return read_json(game_path)


def create_experiment():
    pass
