import time
import os
import csv
from datetime import datetime as dt
import attr
from attr import define, field

from utils import printv, extract_dictionary
from agents import NegotiationAgent
from games import Game
from models.model_utils import HumanMessage, AIMessage, SystemMessage
from models.model_offer_extraction import do_offer_extraction
from logger import get_logger

log = get_logger()


@define
class Protocol:
    game: Game = attr.ib()
    agent_1: NegotiationAgent = attr.ib()
    agent_2: NegotiationAgent = attr.ib()
    start_agent_index: int = field(default=0)
    _ordered_agents: list = field(factory=list)
    style: str = field(default="")
    max_rounds: int = field(default=3)
    agreement_prompt: str = field(default='We agree on all issues.')
    format_as_dialogue: bool = field(default=False)
    transcript: str = field(default=None)
    offer_extraction_model_provider: str = field(default='azure')
    offer_extraction_model_name: str = field(default='gpt-3.5-turbo')
    verbosity: int = field(default=1)
    save_folder: str = field(default="data/interrogation_logs/")

    def __attrs_post_init__(self):
        if isinstance(self.save_folder, str):
            os.makedirs(self.save_folder, exist_ok=True)

        agent_list = [self.agent_1, self.agent_2]
        for i, a in enumerate(agent_list):
            # other agent 
            other_agent_id = (i + 1) % 2 
            # provide ID
            a.agent_id = i
            # IF the agent is a representative AND name parties involved in game stated, update external agent name
            a.update_external_description(game=self.game, agent_id=i)
            agent_list[other_agent_id].update_external_description(game=self.game, agent_id=other_agent_id)
            # combine game information and agent information to create the system init description
            a.set_system_description(game=self.game, agent_id=i,
                                     other_agent_desc=agent_list[other_agent_id].external_description)
            # set agreement prompt
            a.agreement_prompt = self.agreement_prompt
            # update dialogue formatting mode
            a.format_as_dialogue = self.format_as_dialogue

        # move the agent order to respect start agent index
        self._ordered_agents = [self.agent_1, self.agent_2] if self.start_agent_index == 0 else [self.agent_2,
                                                                                                 self.agent_1]
        if self.transcript is not None:
            self.agent_1.copy_agent_history_from_transcript(transcript=self.transcript, agent_id=0)
            self.agent_2.copy_agent_history_from_transcript(transcript=self.transcript, agent_id=1)

        self._subclass_init()

    def _subclass_init(self):
        pass

    def run(self):
        pass

    @staticmethod
    def get_save_headers(return_as_dict=False):
        raise NotImplementedError()

    @staticmethod
    def get_save_fname():
        raise NotImplementedError()

    def save_results(self, **data):
        raise NotImplementedError()


@define
class InterrogationProtocol(Protocol):
    """
    Run negotiations
    """
    questions: list = field(factory=list)
    save_folder: str = field(default="data/interrogation_logs/")
    question_history = field(factory=list)
    answer_history = field(factory=list)
    
    stop_condition: str = field(default='max_rounds')
    num_agreed_issues: int = field(default=0)
    save_folder: str = field(default='data/logs')
    check_message_for_offers: bool = field(default=False)
    round_num: int = field(default=1)
    round_start: bool = field(default=True)
    full_agreement_string: str = field(default='full_agreement_string')

    def _subclass_init(self):
        pass

    def run(self):
        # iterate through questions and over rounds
        if self.style == "final_round":
            for q in self.questions:
                self.query_agents(query=q)

        if self.style == "all_rounds":
            for i in range(len(self.agent_1.msg_history)):
                for q in self.questions:
                    self.query_agent(agent_id=0, query=q, round_num=i)

            for j in range(len(self.agent_2.msg_history)):
                for q in self.questions:
                    self.query_agent(agent_id=1, query=q, round_num=j)

    def query_agents(self, query):
        self.query_agent(0, query)
        self.query_agent(1, query)

    def query_agent(self, agent_id, query, round_num=1e6):
        round_num = int(round_num)
        agent = [self.agent_1, self.agent_2][agent_id]
        # make copy of full history to reset after query
        a_mh = agent.msg_history.copy()
        a_nh = agent.notes_history.copy()
        # get last message
        last_message = a_mh[round_num][1]
        # set agents' memories to point-in-time
        agent.msg_history = agent.msg_history[:round_num]
        agent.notes_history = agent.notes_history[:round_num]
        # get other agent msg history
        other_agent_idx = [1, 0][agent_id]
        c_msg_history = [self.agent_1, self.agent_2][other_agent_idx].msg_history[:round_num]
        neg_transcript = agent.prepare_msg_note_input(c_msg_history=c_msg_history,
                                                      transcript_only=True)

        if len(neg_transcript) > 0:
            if self.format_as_dialogue:
                query_context = f'Based on the negotiations so far, I would like to ask you some questions.'
            else:
                query_context = f'Based on the negotiations transcript so far, I would like to ask you some questions.'
                query_context = query_context + neg_transcript[0].content
        else:
            query_context = 'Based on the negotiations you are about to enter, I would like to ask you some questions.'

        session = neg_transcript if self.format_as_dialogue else []
        # only take the session questions/answers of relevant agent
        a_qh = [q for (i, q) in self.question_history if i == agent_id]
        a_ah = [a for (i, a) in self.answer_history if i == agent_id]
        if len(a_qh) > 0:
            for i, (q, a) in enumerate(zip(a_qh, a_ah)):
                if i == 0:
                    q = query_context + q
                a = AIMessage(a)
                q = SystemMessage(q) if self.format_as_dialogue else HumanMessage(q)
                session.extend([q, a])
            session.append(SystemMessage(query) if self.format_as_dialogue else HumanMessage(query))
        else:
            if self.format_as_dialogue:
                session.append(SystemMessage(query_context + query))
            else:
                session = [HumanMessage(query_context + query)]

        context = [agent.system_description] + session
        answer = agent.model(context)

        self.question_history.append((agent_id, query))
        self.answer_history.append((agent_id, answer))
        self.save_results(agent=agent, agent_id=agent_id, round_num=round_num, last_message=last_message)

        # reinstate agent history
        agent.msg_history = a_mh
        agent.notes_history = a_nh

        return answer

    @staticmethod
    def get_save_headers(return_as_dict=False):
        headers = ['round_num', 'agent_name', "agent_id", 'last_message', 'question', 'answer', 'timestamp',
                   'model_name', 'extracted_answer']
        if return_as_dict:
            headers_d = {
                'c_round': 'round_num',
                'c_agent_name': 'agent_name',
                'c_agent_id': 'agent_id',
                'c_last_msg': 'last_message',
                'c_question': 'question',
                'c_answer': 'answer',
                'c_timestamp': 'timestamp',
                'c_model_name': 'model_name',
                'c_extracted_answer': 'extracted_answer'
            }
            return headers_d

        return headers

    @staticmethod
    def get_save_fname():
        return 'interrogation.csv'

    def save_results(self, agent, agent_id, round_num, last_message, extract_table=True):
        headers = self.get_save_headers()
        fname = self.get_save_fname()

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
            extracted_answer = None
            if extract_table:
                extracted_answer = do_offer_extraction(answer, self.game.issues,
                                                       model_name=self.offer_extraction_model_name,
                                                       model_provider=self.offer_extraction_model_provider)

            data = [round_num, agent.agent_name, agent_id, last_message, question, answer, timestamp, model_name,
                    extracted_answer]

            try:
                writer.writerow(data)
            except Exception as e:
                print(f'warning: failed to write row! - {e}')


@define
class NegotiationProtocol(Protocol):
    """
    Run negotiations
    """
    stop_condition: str = field(default='max_rounds')
    num_agreed_issues: int = field(default=0)
    save_folder: str = field(default='data/logs')
    check_message_for_offers: bool = field(default=False)
    round_num: int = field(default=1)
    round_start: bool = field(default=True)
    full_agreement_string: str = field(default='full_agreement_string')

    def _subclass_init(self):

        if self.transcript is not None:
            self.save_folder = "/".join(self.transcript.split("/")[:-1])
            self.round_num = max(len(self.agent_1.msg_history), len(self.agent_2.msg_history))
            self.round_start = len(self.agent_1.msg_history) == len(self.agent_2.msg_history)

    def run(self):

        self._format_run_print()

        completed = False
        round_num = self.round_num
        a1, a2 = self._ordered_agents
        ts = []
        while not completed:
            t = time.time()
            if self.round_start:
                self._format_round_print(round_num=round_num, total_rounds=self.max_rounds, start=True)

            actor, listener = (a1, a2) if self.round_start else (a2, a1)
            completed = self.take_turn(actor=actor, listener=listener, round_num=round_num, start=self.round_start)
            ts.append(time.time() - t)

            if not self.round_start:
                self._format_round_print(round_num=round_num, total_rounds=self.max_rounds, t1=ts[-1], t2=ts[-2])
                round_num += 1
            self.round_start = not self.round_start

    def take_turn(self, actor, listener, round_num, start):
        # read in the messages sent by the other party so far
        c_msg_history_ext = [(listener.agent_name_ext, msg) for (_, msg) in listener.msg_history]
        # take a note + message step
        actor.step(c_msg_history_ext, round_num=round_num, max_rounds=self.max_rounds)
        # check for completion
        completed, completion_reason = self.check_completion(agent=actor, c_msg_history=listener.msg_history,
                                                             num_rounds=round_num - 1 if start else round_num)
        # save results for analysis
        self.save_results(agent=actor, round_num=round_num, agent_id=actor.agent_id,
                          completion_reason=completion_reason)
        return completed

    def _format_round_print(self, round_num, total_rounds, t1=0., t2=0., start=False):
        prompt_costs, completion_costs = 0, 0
        for a in [self.agent_1, self.agent_2]:
            prompt_costs += a.model.session_prompt_costs
            completion_costs += a.model.session_completion_costs
        tc = prompt_costs + completion_costs

        if start:
            s = f'\n\nROUND [{round_num}/{total_rounds}]: ' \
                f'session costs: ${tc : .4f} (prompt: ${prompt_costs : .3f} | completion: ${completion_costs :.3f}), ' \
                f'agreed issues: {self.num_agreed_issues}\n\n'
        else:
            s = f'\n\nROUND END [{round_num}/{total_rounds}]: f' \
                f'session costs: ${tc : .4f} (prompt: ${prompt_costs : .3f} | completion: ${completion_costs :.3f}), ' \
                f'agreed issues: {self.num_agreed_issues}'\
                f'time {t1: .4f}s | {t2: .4f}s\n\n'

        printv(s, self.verbosity)

    def _format_run_print(self):
        lj = 20
        gs = '\n>Starting negotiation protocol:\n'
        for k, v in zip(['game', 'issues', 'dialogue_style'],
                        [self.game.name, [i.name for i in self.game.issues], self.format_as_dialogue]):
            gs += f'  {k}:'.ljust(lj) + f'{v}\n'

        for i, a in enumerate([self.agent_1, self.agent_2]):
            gs += f'\nagent {i + 1}:\n'
            for k, v in zip(['name internal', 'name external', 'model name', 'preferences'],
                            [a.agent_name, a.agent_name_ext, a.model_name, self.game.issue_weights[i]]):
                gs += f'  {k}:'.ljust(lj) + f'{v}\n'

        printv(gs, self.verbosity)

    def check_completion(self, agent, c_msg_history, num_rounds) -> (bool, str):
        # 0. agree textually 
        # 1. full agreement 
        # 2. agree internally 
        # 3. max rounds reached
        # 4. ran out of context tokens
        #   -> a. stop
        #   -> b. drop history
        #   -> c. fancier stuff, e.g., memory bank search etc.
        # 4. ran out of compute budget

        completed = False
        completion_reason = "in-progress"

        # 0
        agreed, state = self.compare_issues(return_issues=True)
        not_disagree, _ = self.compare_issues(return_issues=True, full_agreement=False)
        last_msg_1 = self.agent_1.msg_history[-1][1] if len(self.agent_1.msg_history) > 0 else ""
        last_msg_2 = self.agent_2.msg_history[-1][1] if len(self.agent_2.msg_history) > 0 else ""
        agreed_phrase = self.agreement_prompt.lower()
        # 1 
        if (agreed_phrase in last_msg_1.lower()) and (agreed_phrase in last_msg_2.lower()):
            completed = True
            completion_reason = "disagreeing internal states, textual agreement"
            # 2 
            if not_disagree:
                completed = True
                completion_reason = self.full_agreement_string
                printv(f'[completed] (full agreement) - {state}', self.verbosity)
            else: 
                printv(f'[not completed] (textual agreement) - {state}', self.verbosity)
        elif agreed: 
            completed = False 
            completion_reason = 'aligning internal states, textual disagreement'
            printv(f'[completed] (issues) - {state}', self.verbosity)
        # 3 
        if not completed and num_rounds >= self.max_rounds:
            completed = True
            completion_reason = "max rounds reached"
            printv(f'[completed] (num_rounds) - {num_rounds}/{self.max_rounds}', self.verbosity)
        # 4
        if not completed and not agent.check_context_len_step(c_msg_history):
            completed = True
            completion_reason = "context overflow"
            printv(f'[completed] (context overflow)', self.verbosity)
        # 5
        if not completed and agent.model.budget <= 0:
            completed = True
            completion_reason = "out of compute budget"
            printv(f'[completed] (ran out of compute budget)', self.verbosity)

        return completed, completion_reason

    def get_full_agreement_string(self):
        return self.full_agreement_string

    def compare_issues(self, return_issues=False, full_agreement=True):
        is1 = self.get_issues_state(issues=self.game.issues, agent=self.agent_1)
        log.debug(f"Issues for agent 1: {is1}")
        is2 = self.get_issues_state(issues=self.game.issues, agent=self.agent_2)
        log.debug(f"Issues for agent 2: {is2}")

        if not any(is1) or not any(is2):
            agreed, agreed_issues = False, []
        else:
            if full_agreement:
                is_keys = is2.keys()
                agreed_issues = [k for k in is_keys if is1.get(k) == is2.get(k)]
                agreed = len(agreed_issues) == len(is_keys)

                self.num_agreed_issues = len(agreed_issues)
            else: 
                keys1 = set(is1.keys())
                keys2 = set(is2.keys())

                intersecting_keys = keys1.intersection(keys2)

                agreed_issues = [k for k in intersecting_keys if is1.get(k) == is2.get(k)]
                agreed = len(agreed_issues) == len(intersecting_keys)

        if return_issues:
            return agreed, agreed_issues

        return agreed

    def get_issues_state(self, issues, agent) -> dict:
        """
        Each mental note should conclude with the current issues state.
        The regext should retrieve the following:

        s = '''this is a story with a proposal.

            acceptable proposal:
            {
                "issue_0": "value"
            }
        '''
        --> {"issue_0": "value"}

        :return: state
        """
        state_dict = {}
        if len(agent.notes_history) > 0:
            _, last_note = agent.notes_history[-1]

            state_dict = extract_dictionary(last_note)
            if state_dict is None:
                try:
                    state_dict = do_offer_extraction(last_note, issues, idx=agent.agent_id, is_note=True,
                                                     model_name=self.offer_extraction_model_name,
                                                     model_provider=self.offer_extraction_model_provider)
                except Exception as e:
                    print(f'error: unable to retrieve valid state from notes - {e}')

            if any(state_dict):
                agent.achievable_payoffs = state_dict

        return state_dict

    @staticmethod
    def get_save_headers(return_as_dict=False):
        headers = ['agent_name', "agent_id", 'round', 'note', 'message', 'issues_state', 'timestamp', 'model_name',
                   'completion_reason', 'prompt_cost', 'completion_cost']
        if return_as_dict:
            return {f"c_{v}": v for v in headers}

        return headers

    @staticmethod
    def get_save_fname():
        return 'negotiations.csv'

    def save_results(self, agent, round_num, agent_id, completion_reason):
        headers = self.get_save_headers()
        fname = self.get_save_fname()

        csv_path = os.path.join(self.save_folder, fname)
        csv_path_exists = os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            if not csv_path_exists:
                writer.writerow(headers)

            note = '' if len(agent.notes_history) == 0 else agent.notes_history[-1][1]
            msg = '' if len(agent.msg_history) == 0 else agent.msg_history[-1][1]
            issues_state = self.get_issues_state(issues=self.game.issues, agent=agent)
            timestamp = dt.strftime(dt.now(), '%Y%m%d_%H%M%S')
            model_name = agent.model_name
            data = [agent.agent_name, agent_id, round_num, note, msg, issues_state, timestamp, model_name,
                    completion_reason, agent.model.session_prompt_costs, agent.model.session_completion_costs]
            try:
                writer.writerow(data)
            except Exception as e:
                print(f'warning: failed to write row! - {e}')
