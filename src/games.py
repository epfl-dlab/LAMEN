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
from omegaconf.listconfig import ListConfig as omega_list
from attr import define, field
from typing import Optional
from logger import get_logger

log = get_logger()


@define
class Game:
    name: str
    description: str
    issues: list
    issue_weights: list
    scale: tuple = field(default=(1, 1))
    sides: list = field(default=None)
    parties: list = field(default=None)
    rules: list = field(factory=list)
    rules_prompt: str = field(default='')

    def __attrs_post_init__(self):
        # load in the issues in correct format
        self.load_issues()
        # scale issues to their importance
        self.reweigh_issues()

    def reweigh_issues(self):
        # normalize weights to [0, 1]
        iw = np.asarray(self.issue_weights)
        iws = np.sum(iw, axis=1, keepdims=True)
        issue_weights = (iw / iws)
        for issue, w in zip(self.issues, issue_weights.transpose()):
            payoffs = []
            for po, w_, s in zip(issue.payoffs, w, self.scale):
                # integer values only, TODO: make sure rounding is sensible!
                po = (np.asarray(po) / max(po) * w_ * s).round(3)
                payoffs.append(po)
            issue.payoffs = payoffs

    def get_general_rules(self):
        game_rules = ''
        if self.rules_prompt is not None and (isinstance(self.rules, (list, omega_list)) and any(self.rules)):
            game_rules += self.rules_prompt
            for rule in self.rules:
                game_rules += '\n- ' + rule
        return game_rules

    def get_system_msg(self, agent_id, agent_desc_int, other_agent_desc=None, visibility=0):
        # TODO: think on how to best format agent internal/external description for visiblity > 1
        log.debug(f"Agent {agent_id} side description: {self.sides[agent_id]}")

        shared_c = self.description
        side_c = self.sides[agent_id]
        agent_c = ''
        if agent_desc_int != {'name': 'You'}:
            agent_c = f'Description of who you are:\n{agent_desc_int}'
        po = self.format_all_issues(agent_id)
        payoffs = f'Your payoffs and possible values are listed in the payoff tables below.'\
                  f'\nAdopt these payoffs as your preferences while negotiating.\n{po}'

        system_msg = f'{shared_c}\n{side_c}'
        if len(agent_c) > 0:
            system_msg += f'\n{agent_c}'
        system_msg += f'\n{payoffs}'

        if visibility > 0:
            other_agent_id = 1 if agent_id == 0 else 0
            po_ = self.format_all_issues(other_agent_id)
            other_side_payoffs = f'\n\nThe other side their payoff table are listed below:\n{po_}'
            system_msg += other_side_payoffs
        if visibility > 1:
            if other_agent_desc != {'name': 'You'}:
                o_agent_c = f'\n\nThe other side is represented by:\n{other_agent_desc}'
                system_msg += o_agent_c

        game_rules = self.get_general_rules()
        if len(game_rules) > 0:
            system_msg += f'\n\n{game_rules}'

        return system_msg

    def load_issues(self, issues_path="data/issues/"):
        issues = []
        for issue in self.issues:
            if isinstance(issue, str):
                if not issue.endswith('.yaml'):
                    issue += '.yaml'
                fname = os.path.join(issues_path, issue)
                issues.append(Issue.load(fname))
            if isinstance(issue, dict):
                issues.append(Issue.from_dict(issue))
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

    def get_optimal_score(self) -> float:
        """
        NOTE: this only works for (1) distributive issues, (2) compatible issues w/ equal preference weights

        For N issues, two agents have a payoff per-issue, {a_i}N, {b_i}N, s.t.: sum a_i = sum b_i = 1
        The task is to allocate weights {x_i}, where x_i in [0, 1], s.t.:
        max_xi f({a_i}, {b_i}, {x_i}) = sum ai * xi + sum bi * xi, s.t. sum a_i ^ x_i = sum b_i * x_i

        Given the weights of {a_i}, {b_i}, there might be various allocations of the {x_i} that satisfy this.
        However, we are only interested in the optimal achievable score, not in the specific allocation.
        As such, it can be shown that this is equal to: 1/2 * sum_i max(a_i, b_i)

        :issue_weights:
        :payoffs: Nx2 np.array, where N is the number of issues
        :return optimal_score: optimal achievable equilibrium score
        """

        iw = np.asarray(self.issue_weights)
        iws = np.sum(iw, axis=1, keepdims=True)
        x = (iw / iws) * np.array(self.scale)[:, None]

        assert np.shape(x)[0] == 2, f'error: expecting payoffs ndarray of shape 2xN'
        idx = np.where(np.asarray([1 if issue.issue_type == 'compatible' else 0 for issue in self.issues]) == 1)[0]
        x_comp = np.sum(x, axis=0)
        max_score_per_issue = np.max(x, axis=0)
        max_score_per_issue[idx] = x_comp[idx]
        max_cum_score = np.sum(max_score_per_issue)
        optimal_score = max_cum_score / 2.

        return optimal_score

    def get_game_type(self) -> str: 
        """
        If one issues:
            - if compatible 
                return 'non-integrative compatible'
            - else:
                return 'non-integrative distributive'

        Four types of games:
            - non-integrative distributive: 
                - payoff weights align
                - no compatible issues 
            - non-integrative compatible 
                - payoff weights algin 
                - atleast one compatible issues 
            - integrative distributive
                - misaligning payoff weights
            - integrative compatible
                - misaligning payoff weights
                - atleast one issue is compatible 
        """

        if len(self.issues) == 0:
            if self.issues[0].issue_type=='compatible':
                return 'non-integrative compatible'
            else:
                return 'non-integrative distributive'

        compatible_issues = [issue for issue in self.issues if issue.issue_type=='compatible']
        compatible = True if len(compatible_issues) > 0 else False 
        integrative = False if self.issue_weights[0] == self.issue_weights[1] else True

        if integrative:
            if compatible: 
                return 'integrative compatible'
            else:
                return 'integrative distributive'

        else: 
            if compatible: 
                return 'non-integrative compatible'
            else:
                return 'non-integrative distributive'

    def get_game_class(self):
        # higher-level category of games
        # cooperative or competitive
        classes = {'integrative compatible': 'cooperative',
                   'integrative distributive': 'cooperative',
                    'non-integrative compatible': 'cooperative',
                   'non-integrative distributive': 'competitive'}

        return classes[self.get_game_type()]

@define
class Issue:
    name: str
    descriptions: str
    payoffs: list
    payoff_labels: list
    num_steps: int = field(default=10)
    issue_type: str = field(default='custom')

    def __attrs_post_init__(self):
        if not any(self.payoffs):
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
        # note: we should specify column --> value, payoff
        issue_format = self.name + "\nvalue, payoff\n"
        # descending order on issue weights for prompt
        idx = int(idx)
        indices = np.argsort(self.payoffs[idx])[::-1]
        payoffs = np.array(self.payoffs)[idx][indices]
        payoff_labels = np.asarray(self.payoff_labels[idx])[indices]
        for label, payoff in zip(payoff_labels, payoffs):
            issue_format += f"{label}, {payoff}\n"
        return issue_format
    
    def __getitem__(self, key):
        return getattr(self, key)


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
