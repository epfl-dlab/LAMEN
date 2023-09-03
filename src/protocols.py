import time
from typing import List
import os
import csv
from datetime import datetime as dt
from attr import define, field
import attr
from utils import printv
from agents import NegotiationAgent
from games import Game
from evaluation import EvaluateNegotiations
from model_utils import HumanMessage, AIMessage
# import logging

from logger import get_logger

log = get_logger()



@define
class InterrogationProtocol:
    """
    Run negotiations
    """
    game: Game = attr.ib()
    agent_1: NegotiationAgent = attr.ib()
    agent_2: NegotiationAgent = attr.ib()
    questions: list = field(factory=list)
    style: str = field(default="")
    start_agent_index: int = field(default=0)
    save_folder: str = field(default="data/interrogation_logs/")
    verbosity: int = field(default=1)
    question_history = field(factory=list)
    transcript: str = field(default=None)
    answer_history = field(factory=list)

    def __attrs_post_init__(self):
        # self.save_folder = "data/interrogation_logs/"
        try: 
            os.makedirs(self.save_folder, exist_ok=True)
        except:
            pass
        [a.set_system_description(self.game, i) for i, a in enumerate([self.agent_1, self.agent_2])]
        # move the agent order to respect start agent index
        if self.start_agent_index != 0:
            self.agent_1, self.agent_2 = self.agent_2, self.agent_1

        if self.transcript is not None:
            self.agent_1.copy_agent_history_from_transcript(transcript=self.transcript, agent_id=0)
            self.agent_2.copy_agent_history_from_transcript(transcript=self.transcript, agent_id=1)
            
    def run(self):
        # iterate through questions and over rounds
        if self.style=="final_round":
            for q in self.questions:
                self.query_agents(query=q)
                
        if self.style=="all_rounds":
            for i in range((max(len(self.agent_1.msg_history), len(self.agent_2.msg_history)))):
                for q in self.questions:
                    self.query_agents(query=q, round_num=i)

    def query_agent(self, agent_id, query, round_num=1e6):
        round_num = int(round_num)
        agent = [self.agent_1, self.agent_2][agent_id]
        # make copy of full history to reset after query
        a_mh = agent.msg_history.copy()
        a_nh = agent.notes_history.copy()
        # set agents' memories to point-in-time
        agent.msg_history = agent.msg_history[:round_num]
        agent.notes_history = agent.notes_history[:round_num]
        # get other agent msg history
        other_agent_idx = [1, 0][agent_id]
        c_msg_history = [self.agent_1, self.agent_2][other_agent_idx].msg_history[:round_num]
        neg_transcript = agent.prepare_msg_note_input(c_msg_history=c_msg_history,
                                                      transcript_only=True)
        if len(neg_transcript) > 0:
            query_context = f'Based on the negotiations transcript so far, I would like to ask you some questions.'
            query_context += neg_transcript
        else:
            query_context = 'Based on the negotiations you are about to enter, I would like to ask you some questions.'

        session = []
        # only take the session questions/answers of relevant agent
        a_qh = [q for (i, q) in self.question_history if i == agent_id]
        a_ah = [a for (i, a) in self.answer_history if i == agent_id]
        if len(a_qh) > 0:
            for i, (q, a) in enumerate(zip(a_qh, a_ah)):
                if i == 0:
                    q = query_context + q
                session.extend([HumanMessage(q), AIMessage(a)])
            session.append(HumanMessage(query))
        else:
            session = [HumanMessage(query_context + query)]

        context = [agent.system_description] + session
        answer = agent.model(context)

        self.question_history.append((agent_id, query))
        self.answer_history.append((agent_id, answer))
        self.save_conversation(agent=agent, agent_id=agent_id)

        # reinstate agent history
        agent.msg_history = a_mh
        agent.notes_history = a_nh

        return answer
    
    def query_agents(self, query, round_num=1e6):
        self.query_agent(agent_id=0, query=query, round_num=round_num)
        self.query_agent(agent_id=1, query=query, round_num=round_num)

    def start_session(self):
        # command line interface
        self.question_history = []
        self.answer_history = []

    def end_session(self):
        # detect termination
        pass

    def save_conversation(self, agent, agent_id):

        headers = ['agent_name', "agent_id", 'question', 'answer', 'timestamp', 'model_name']
        fname = 'interrogation.csv'
        csv_path = os.path.join(self.save_folder, fname)
        csv_path_exists = os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            if not csv_path_exists:
                writer.writerow(headers)

            question = self.question_history[-1][1]
            answer = self.answer_history[-1][1]
            timestamp = dt.strftime(dt.now(), '%Y%m%d_%H%M%S')
            model_name = agent.model_name
            data = [agent.agent_name, agent_id, question, answer, timestamp, model_name]
            try:
                writer.writerow(data)
            except Exception as e:
                print(f'warning: failed to write row! - {e}')


@define
class NegotiationProtocol:
    """
    Run negotiations
    """
    game: Game
    agent_1: NegotiationAgent
    agent_2: NegotiationAgent
    max_rounds: int = field(default=10)
    stop_condition: str = field(default='max_rounds')
    start_agent_index: int = field(default=0)
    num_agreed_issues: int = field(default=0)
    save_folder: str = field(default='data/logs')
    check_message_for_offers: bool = field(default=False)
    verbosity: int = field(default=1)
    transcript: str = None
    round_num: int = 0

    def __attrs_post_init__(self):
        try: 
            os.makedirs(self.save_folder, exist_ok=True)
        except:
            pass
        # combine game information and agent information to create the system init description
        [a.set_system_description(self.game, i) for i, a in enumerate([self.agent_1, self.agent_2])]
        # move the agent order to respect start agent index
        if self.start_agent_index != 0:
            self.agent_1, self.agent_2 = self.agent_2, self.agent_1
            
        if self.transcript is not None: 
            self.save_folder = "/".join(self.transcript.split("/")[:-1])
            self.agent_1.copy_agent_history_from_transcript(transcript=self.transcript, agent_id=0)
            self.agent_2.copy_agent_history_from_transcript(transcript=self.transcript, agent_id=1)

            if len(self.agent_1.msg_history) > len(self.agent_2.msg_history):
                self.round_num = len(self.agent_1.msg_history)
                self.agent_1, self.agent_2 = self.agent_2, self.agent_1
            else:
                self.round_num = len(self.agent_2.msg_history)

    def run(self):
        printv(f'Starting negotiations protocol with agents:\n'
               f'{self.agent_1.internal_description}\n'
               f'{self.agent_2.internal_description}\n', self.verbosity)

        completed = False
        round_num = self.round_num
        while not completed:
            ts = []
            t = time.time()

            self._format_round_print(round_num=round_num, total_rounds=self.max_rounds, start=True)

            c_msg_history_ext = [(self.agent_2.agent_name_ext, msg) for (_, msg) in self.agent_2.msg_history]
            self.agent_1.step(c_msg_history_ext)
            log.debug(f"Agent 1 note history after step: {self.agent_1.notes_history}")
            completed, completion_reason = self.check_completion(agent=self.agent_1,
                                                                 c_msg_history=self.agent_1.msg_history,
                                                                 num_rounds=round_num)
            self.save_results(agent=self.agent_1,
                              round_num=round_num, agent_id=0,
                              completion_reason=completion_reason)
            ts.append(time.time() - t)
            t = time.time()

            if not completed:
                c_msg_history_ext = [(self.agent_1.agent_name_ext, msg) for (_, msg) in self.agent_1.msg_history]
                self.agent_2.step(c_msg_history_ext)
                completed, completion_reason = self.check_completion(agent=self.agent_2,
                                                                     c_msg_history=self.agent_2.msg_history,
                                                                     num_rounds=round_num)
                self.save_results(agent=self.agent_2, round_num=round_num, agent_id=1,
                                  completion_reason=completion_reason)

            ts.append(time.time() - t)
            self._format_round_print(round_num=round_num, total_rounds=self.max_rounds,
                                     t1=ts[-1], t2=ts[-2])
            round_num += 1

    def evaluate(self):
        nego_eval = EvaluateNegotiations(save_dir=self.save_folder, game=self.game,
                                         check_message_for_offers=self.check_message_for_offers)
        nego_eval.compute_metrics()
        
    def interrogate(self, questions, style):
        nego_interro = InterrogationProtocol(save_folder=self.save_folder, 
                                             game=self.game, agent_1=self.agent_1, 
                                             agent_2=self.agent_2, questions=questions,
                                             style=style, start_agent_index=self.start_agent_index,
                                             verbosity=self.verbosity)
        nego_interro.run()

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
