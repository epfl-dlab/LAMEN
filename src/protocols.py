import time
import os
import csv
from datetime import datetime as dt
from attr import define, field
from utils import printv
from agents import NegotiationAgent
from games import Game, Issue
from evaluation import EvaluateNegotiations
from model_utils import ChatModel


@define
class InterrogationProtocol:
    agent: NegotiationAgent
    transcript: str
    config: str

    def __attrs_post_init__(self):
        # return if agent object already passed in
        if self.agent is not None:
            return

        if self.transcript is None and self.config is None:
            raise TypeError(f'error: must provide either agent OR (config and transcript)')

        self.agent = NegotiationAgent().init_agent_from_transcript()

    def query_agent(self):
        pass

    def start_session(self):
        # command line interface
        pass

    def end_session(self):
        pass

    def save_conversation(self):
        pass


@define
class NegotiationProtocol:
    """
    Run negotiations
    """
    game: Game
    agent_1: NegotiationAgent
    agent_2: NegotiationAgent
    max_rounds: int = 10
    stop_condition: str = 'max_rounds'
    start_agent_index: int = 0
    num_agreed_issues: int = 0
    save_folder: str = 'data/logs'
    verbosity: int = 0

    def __attrs_post_init__(self):
        os.makedirs(self.save_folder, exist_ok=True)

        [a.copy_game_data(self.game, i) for i, a in enumerate([self.agent_1, self.agent_2])]
        # move the agent order to respect start agent index
        if self.start_agent_index != 0:
            self.agent_1, self.agent_2 = self.agent_2, self.agent_1

    def run(self):
        completed = False
        round_num = 0
        while not completed:
            ts = []
            t = time.time()
            self._format_round_print(round_num=round_num, total_rounds=self.max_rounds, start=True)
            self.agent_1.step(self.agent_2.msg_history)
            completed = self.check_completion(agent=self.agent_2,
                                              c_msg_history=self.agent_1.msg_history,
                                              num_rounds=round_num)
            self.save_results(agent=self.agent_1, round_num=round_num, agent_id=0)
            ts.append(time.time() - t)
            t = time.time()

            if not completed:
                self.agent_2.step(self.agent_1.msg_history)
                self.save_results(agent=self.agent_2, round_num=round_num, agent_id=1)
                completed = self.check_completion(agent=self.agent_1,
                                                  c_msg_history=self.agent_2.msg_history,
                                                  num_rounds=round_num)
            ts.append(time.time() - t)
            self._format_round_print(round_num=round_num, total_rounds=self.max_rounds,
                                     t1=ts[-1], t2=ts[-2])
            round_num += 1

    def evaluate(self):
        nego_eval = EvaluateNegotiations(self.save_folder, self.game)
        nego_eval.compute_metrics()

    def _format_round_print(self, round_num, total_rounds, t1=0., t2=0., start=False):
        prompt_costs, completion_costs = 0, 0
        for a in [self.agent_1, self.agent_2]:
            prompt_costs += a.model.session_prompt_costs
            completion_costs += a.model.session_completion_costs
        tc = prompt_costs + completion_costs

        if start:
            s = f'\n\nROUND [{round_num}/{total_rounds}]: ' \
                f'session costs: ${tc : .2f} (prompt: ${prompt_costs : .1f} | completion: ${completion_costs :.1f}), ' \
                f'agreed issues: {self.num_agreed_issues}\n\n'
        else:
            s = f'\n\nROUND END [{round_num}/{total_rounds}]: time {t1: .1f}s | {t2: .1f}s\n\n'

        printv(s, self.verbosity)

    def check_completion(self, agent, c_msg_history, num_rounds) -> bool:
        # 1. all issues are agreed upon
        # 2. max rounds reached
        # 3. ran out of context tokens
        #   -> a. stop
        #   -> b. drop history
        #   -> c. fancier stuff, e.g., memory bank search etc.
        # 4. ran out of compute budget

        completed = False

        # 1
        agreed, state = self.compare_issues(return_issues=True)
        if agreed:
            completed = True
            printv(f'[completed] (issues) - {state}', self.verbosity)
        # 2
        if num_rounds >= self.max_rounds:
            completed = True
            printv(f'[completed] (num_rounds) - {num_rounds}/{self.max_rounds}', self.verbosity)
        # 3
        if not agent.check_context_len_step(c_msg_history):
            completed = True
            printv(f'[completed] (context overflow)', self.verbosity)

        # 4
        if agent.model.budget <= 0:
            completed = True
            printv(f'[completed] (ran out of compute budget)', self.verbosity)

        return completed

    def compare_issues(self, return_issues=False):
        is1 = self.agent_1.get_issues_state()
        is2 = self.agent_2.get_issues_state()

        if not any(is1) or not any(is2):
            agreed, agreed_issues = False, []
        else:
            is_keys = is2.keys()
            agreed_issues = [k for k in is_keys if is1.get(k) == is2.get(k)]
            agreed = len(agreed_issues) == len(is_keys)
            self.num_agreed_issues = len(agreed_issues)

        if return_issues:
            return agreed, agreed_issues

        return agreed

    def save_results(self, agent, round_num, agent_id):
        headers = ['agent_name', "agent_id", 'round', 'note', 'message', 'issues_state', 'timestamp', 'model_name']
        fname = 'negotiations.csv'
        csv_path = os.path.join(self.save_folder, fname)
        csv_path_exists = os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            if not csv_path_exists:
                writer.writerow(headers)

            note = '' if len(agent.notes_history) == 0 else agent.notes_history[-1][1]
            msg = '' if len(agent.msg_history) == 0 else agent.msg_history[-1][1]
            issues_state = agent.get_issues_state()
            timestamp = dt.strftime(dt.now(), '%Y%m%d_%H%M%S')
            model_name = agent.model_name
            data = [agent.agent_name, agent_id, round_num, note, msg, issues_state, timestamp, model_name]
            try:
                writer.writerow(data)
            except Exception as e:
                print(f'warning: failed to write row! - {e}')
