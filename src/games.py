""" Methods to create two-player games.

From Michal email, support following issues types:
1. Compatible: (both sides want the same thing) and max it out.
2. Integrative: (it is worth more for one side than to the other) and max it out.
3. Distributive ($1 loss for you is $1 gain for me) and distribute them.

- A 'game' consists of one or more 'issues'
    - Each 'issue' is a two-way payoff matrix with meta details, e.g., name, type
    - For now, we assume payoff matrices for the same issue have the same number of options

Games are saved in YAML files:
- creator: (str)
- date: (str), yyyymmdd_hhmmss
- description: (str)
- sides: []
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
import yaml
from attr import define
from typing import Optional
from logger import get_logger

log = get_logger()

@define
class Game:
    name: str
    description: str
    issues: list
    issue_weights: list
    scale: tuple = (1, 1)
    sides: list = None
    rules: list = []
    rules_prompt: str = ''

    def __attrs_post_init__(self):
        # load in the issues in correct format
        self.load_issues()
        # scale issues to their importance
        self.reweigh_issues()
        # add general rules
        self.add_general_rules()

    def reweigh_issues(self):
        # normalize weights to [0, 1]
        issue_weights = np.asarray(self.issue_weights) / np.sum(self.issue_weights, axis=1, keepdims=True)
        for issue, w in zip(self.issues, issue_weights.transpose()):
            payoffs = []
            for po, w_, s in zip(issue.payoffs, w, self.scale):
                po = (np.asarray(po) * w_ * s).round(3)  # integer values only, TODO: make sure rounding is sensible!
                payoffs.append(po)
            issue.payoffs = payoffs

    def add_general_rules(self):
        if self.rules_prompt is not None and (isinstance(self.rules, list) and any(self.rules)):
            self.description = self.description + " " + self.rules_prompt
            for rule in self.rules:
                self.description += '\n' + rule

    def get_system_msg(self):
        pass

    def load_issues(self, issues_path="data/issues/"):
        issues = []
        for issue in self.issues:
            if isinstance(issue, str):
                fname = os.path.join(issues_path, issue + '.yaml')
                issues.append(Issue.load(fname))
        if len(issues) > 0:
            self.issues = issues

    def to_dict(self):
        return vars(self)

    def get_issue(self, issue_name):
        issue = [k for k in self.issues if k.name.replace("_", " ") == issue_name.replace("_", " ")]
        if len(issue) > 0:
            return issue[0]
        else:
            raise NotImplementedError("Issue not found")

    def format_all_issues(self, agent_idx):
        issues_text = ""
        for issue in self.issues:
            issues_text += f"{issue.format_issue(agent_idx)}\n"

        return issues_text.strip()

    @staticmethod
    def from_dict(d):
        return Game(**d)


@define
class Issue:
    name: str
    descriptions: str
    payoffs: list
    payoff_labels: list
    num_steps: int = 10
    issue_type: str = 'custom'

    def __attrs_post_init__(self):
        self.set_payoff_table()

    def set_payoff_table(self):
        log.debug("Setting payoff tables")
        # both sides want the same thing
        if self.issue_type == 'compatible':
            m1 = np.linspace(0, 1, self.num_steps)
            m2 = m1
        # sides want opposite things, e.g., +1 for me is -1 for you
        elif self.issue_type == 'distributive':
            m1 = np.linspace(0, 1, self.num_steps)
            m2 = np.flip(m1)
        # it is worth more to one side than the other
        # NOTE: this is tricky when there are multiple issues in a single game, i.e., how does rescaling/weighing work?
        elif self.issue_type == 'integrative':
            m1 = np.linspace(0, 1, self.num_steps)
            m2 = np.flip(m1) * 0.5
        # user-defined issue
        elif self.issue_type == 'custom':
            m1, m2 = self.payoffs
        else:
            raise NotImplemented(
                f'error: issue type {self.issue_type} not in [compatible, integrative, distributive, custom]')

        self.payoffs = [m1, m2]

    def to_dict(self):
        return vars(self)

    @staticmethod
    def from_dict(d):
        return Issue(**d)

    def save(self, fname):
        d = self.to_dict()
        with open(os.path.join(fname + '.json')) as f:
            yaml.dump(d, f)

    @staticmethod
    def load(fname):
        with open(fname) as f:
            issue = yaml.safe_load(f)

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


def load_game(game_path: str, general_rules: Optional[str] = None) -> dict:
    """
    game_path (str): path to the game file
    general_rules (str): [optional] path to general rules to be added to the description.
    """
    with open(game_path, 'r') as f:
        game = yaml.safe_load(f)

    if general_rules is not None:
        with open(general_rules, 'r') as f:
            general_rules_data = yaml.safe_load(f)

        game.rules_prompt = general_rules_data['rules_prompt']
        game.rules = general_rules_data['rules']
        game.add_general_rules()

    return game
