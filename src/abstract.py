from abstract import ABC 
from utils import printv
from evaluation import EvaluateNegotiations
import os
import csv
from logger import get_logger
from datetime import datetime as dt


log = get_logger()

class NegotiationProtocol(ABC):
    def evaluate(self):
        nego_eval = EvaluateNegotiations(save_dir=self.save_folder, game=self.game,
                                         check_message_for_offers=self.check_message_for_offers)
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

    def check_completion(self, agent, c_msg_history, num_rounds) -> (bool, str):
        # 1. all issues are agreed upon
        # 2. max rounds reached
        # 3. ran out of context tokens
        #   -> a. stop
        #   -> b. drop history
        #   -> c. fancier stuff, e.g., memory bank search etc.
        # 4. ran out of compute budget

        completed = False
        completion_reason = "in-progress"
        # 1
        agreed, state = self.compare_issues(return_issues=True)
        if agreed:
            completed = True
            completion_reason = "issues agreed upon"
            printv(f'[completed] (issues) - {state}', self.verbosity)
        # 2
        if num_rounds >= self.max_rounds:
            completed = True
            completion_reason = "max rounds reached"
            printv(f'[completed] (num_rounds) - {num_rounds}/{self.max_rounds}', self.verbosity)
        # 3
        if not agent.check_context_len_step(c_msg_history):
            completed = True
            completion_reason = "context overflow"
            printv(f'[completed] (context overflow)', self.verbosity)
        # 4
        if agent.model.budget <= 0:
            completed = True
            completion_reason = "out of compute budget"
            printv(f'[completed] (ran out of compute budget)', self.verbosity)

        return completed, completion_reason

    def compare_issues(self, return_issues=False):
        is1 = self.agent_1.get_issues_state()
        log.debug(f"Issues for agent 1: {is1}")
        is2 = self.agent_2.get_issues_state()
        log.debug(f"Issues for agent 2: {is2}")

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

    def save_results(self, agent, round_num, agent_id, completion_reason):
        headers = ['agent_name', "agent_id", 'round', 'note', 'message', 'issues_state', 'timestamp', 'model_name',
                   'completion_reason']
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
            data = [agent.agent_name, agent_id, round_num, note, msg, issues_state, timestamp, model_name,
                    completion_reason]
            try:
                writer.writerow(data)
            except Exception as e:
                print(f'warning: failed to write row! - {e}')
