import os
import time
import csv
from datetime import datetime as dt
from model_utils import SystemMessage, HumanMessage, ChatModel
from evaluation import EvaluateNegotiations
from utils import get_api_key, printv
# from omegaconf import DictConfig
import re
import json

import logging

log = logging.getLogger("my-logger")


# def create_agent_description(agent_dict, path='../data/agents'):
#     # TODO: think of sensible comparison axes, e.g., age, profession, sex, etc.
#     pass
#
#
# def load_agent_description(agent_path) -> str:
#     # TODO: load agent json and join into single string
#     pass


class NegotiationAgent:
    def __init__(self, agent_name=None, init_description=None,
                 msg_prompt=None, note_prompt=None,
                 msg_max_len=64, note_max_len=64,
                 msg_input_msg_history=-1, msg_input_note_history=-1,
                 note_input_msg_history=-1, note_input_note_history=-1,
                 model_name="gpt-4", model_provider='openai', model_key='dlab_openai_key',
                 model_key_path='secrets.json', verbosity=2, debug_mode=False, **kwargs) -> None:
        """
        Basic agent class
        """
        self.init_settings = locals()
        self.verbosity = verbosity

        self.system_description = ''
        self.init_description = init_description
        self.agent_name = agent_name

        # message params
        self.msg_prompt_path, self.msg_max_len = msg_prompt, msg_max_len
        self.msg_prompt = self.get_msg_note_prompt(msg_prompt, is_note=False)
        self.msg_input_msg_history, self.msg_input_note_history = msg_input_msg_history, msg_input_note_history
        # notes params
        self.note_prompt_path, self.note_max_len = note_prompt, note_max_len
        self.note_prompt = self.get_msg_note_prompt(note_prompt, is_note=True)
        self.note_input_msg_history, self.note_input_note_history = note_input_msg_history, note_input_note_history
        # generation parameters
        self.generation_parameters = kwargs
        # model
        self.model_name = model_name
        api_key_mappings = {"openai": "OPENAI_API_KEY", "azure":"AZURE_API_KEY", "anthropic":"ANTHROPIC_API_KEY"}

        api_key = get_api_key(fname=model_key_path, key=api_key_mappings[model_provider])
        self.model = ChatModel(
            model_name=model_name, model_provider=model_provider, model_key=api_key,debug_mode=debug_mode,
            **self.generation_parameters
        )

        # keep track of what agents think they can achieve
        self.achievable_payoffs = {}
        self.issue_payoffs = {}

        self.notes_history = []
        self.msg_history = []

    def generate_message(self, c_msg_history):
        # msg prompt structure
        # system_msg(game, side, agent, issues) + user_msg(c_msg, note, msg, c_msg, note, prompt) -> agent_msg: msg
        user_msg = self.prepare_msg_note_input(c_msg_history)
        printv(f'\nMSG PROMPT\n {user_msg} \n', self.verbosity, 1)
        context = [self.system_description, HumanMessage(user_msg)]
        msg = self.model(context)
        self.msg_history.append((self.agent_name, msg))

        printv(f'[{self.agent_name}] [msg]: {msg}', self.verbosity)

        return msg

    def generate_note(self, c_msg_history):
        # note prompt structure:
        # system_msg(game, side, agent, issues) + user_msg(c_msg, note, msg, c_msg, prompt) --> agent_msg: note
        user_msg = self.prepare_msg_note_input(c_msg_history)
        printv(f'\n[NOTE PROMPT]\n {user_msg} \n', self.verbosity, 1)
        context = [self.system_description, HumanMessage(user_msg)]
        note = self.model(context)
        self.notes_history.append((self.agent_name, note))

        printv(f'\n[{self.agent_name}] [note]: {note}\n', self.verbosity)

        return note

    def prepare_msg_note_input(self, c_msg_history: list, simulate_msg=False) -> str:
        """
        Create input context for writing the next message/note.

        We loop backwards, only added context as far as our 'memory' is allowed to go.
        :param c_msg_history: messages generated by the other agent
        :param simulate_msg:
        :return: context msg
        """
        # to check generative capacity moving forward without overriding real history
        n_h = self.notes_history.copy()
        m_h = self.msg_history.copy()
        cm_h = c_msg_history.copy()

        # is this the first message?
        first_msg_note = (len(n_h) == 0 or len(m_h) == 0) and len(cm_h) == 0

        # IF need to simulate msg create phantom note --> needed to check for context overflow
        if simulate_msg:
            if len(n_h) > 0:
                # repeat previous note
                n_h.append(n_h[-1])
            else:
                # if no note history exists, create a dummy note
                n_h.append((self.agent_name, "hello " * self.note_max_len))

        # more notes than messages indicates next generation will be a message
        msg = len(n_h) > len(m_h)
        max_len = max(len(n_h), len(m_h))
        max_m = self.msg_input_msg_history if msg else self.note_input_msg_history
        max_n = self.msg_input_note_history if msg else self.note_input_note_history
        if max_m < 0:
            max_m = 1e5
        if max_n < 0:
            max_n = 1e5

        neg_history = []
        for i in range(1, max_len+1):

            if msg and (i <= max_n) and (0 < len(n_h)):
                neg_history.append(('mental note', n_h.pop()))

            if i <= max_m:
                if 0 < len(cm_h):
                    neg_history.append(('offer', cm_h.pop()))
                if 0 < len(m_h):
                    neg_history.append(('offer', m_h.pop()))

            if not msg and (i <= max_n) and (0 < len(n_h)):
                neg_history.append(('mental note', n_h.pop()))

            if (i == max_len) and (0 < len(cm_h)) and (i <= max_m):
                neg_history.append(('offer', cm_h.pop()))

        neg_history = neg_history[::-1]
        neg_transcript = ''
        for i_type, (a, i) in neg_history:
            neg_transcript += f'[{a}] [{i_type}]\n{i}\n'

        next_prompt = self.msg_prompt if msg else self.note_prompt
        if first_msg_note:
            to_replace = "Reflect on the negotiations transcript so far."
            replacement = "You are to start the negotiations."
            next_prompt = next_prompt.replace(to_replace, replacement)

            user_msg = next_prompt
            if len(neg_transcript) > 0:
                user_msg += '\nA log of your mental notes so far:\n\n'
                user_msg += '<start transcript>\n' + neg_transcript + '\n<end transcript>\n'

        else:
            user_msg = next_prompt
            user_msg += '\nTranscript of ongoing negotiations and your mental notes so far:\n'
            user_msg += '<start transcript>\n' + neg_transcript + '\n<end transcript>\n\n'

        return user_msg

    def step(self, c_msg_history):
        # create mental note
        self.generate_note(c_msg_history)
        # create external message
        new_msg = self.generate_message(c_msg_history)

        return new_msg

    def check_context_len_step(self, c_msg_history) -> bool:
        # 1. enough context token space to generate note?
        # 2. enough context token space to generate msg?

        user_msg = self.prepare_msg_note_input(c_msg_history)
        context = [self.system_description, HumanMessage(user_msg)]
        capacity = self.model.check_context_len(context=context, max_gen_tokens=self.note_max_len)
        if not capacity:
            return capacity

        user_msg = self.prepare_msg_note_input(c_msg_history, simulate_msg=True)
        context = [self.system_description, HumanMessage(user_msg)]
        capacity = self.model.check_context_len(context=context, max_gen_tokens=self.msg_max_len)
        
        return capacity

    def get_settings(self):
        return self.init_settings
    
    def create_static_system_prompt(self, shared_description, side_description, issues_format):
        initial_story = f"""
{shared_description}
{side_description}
\nDescription of your qualities:
{self.init_description} 
Your payoff values are noted below. Adopt these values as your preferences while negotiating.
{issues_format}"""
        self.system_description = SystemMessage(initial_story)
        printv(f'\n[system prompt]\n{initial_story}\n', self.verbosity, 1)

    def get_msg_note_prompt(self, prompt_name, before='max_words', is_note=False):
        prompt_path = 'data/note_prompts' if is_note else 'data/message_prompts'
        max_len = self.note_max_len if is_note else self.msg_max_len
        prompt = open(os.path.join(prompt_path, prompt_name + ".txt")).read()
        prompt = prompt.replace("{" + before + "}", f"{max_len}")

        return prompt

    def get_issues_state(self) -> dict:
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
        state = {}
        if len(self.notes_history) > 0:
            _, last_note = self.notes_history[-1]

            state_regex = re.compile(r"{[\s\S]*?}" )
            state_str = state_regex.search(last_note)
            if state_str is not None:
                try:
                    state = json.loads(state_str[0])
                except Exception as e:
                    print(f'error: unable to retrieve valid state from notes - {e}')

                if any(state):
                    self.achievable_payoffs = state

        return state

    def __repr__(self):
        return f"{self.init_description}"


class NegotiationProtocol:
    """
    Run negotiations
    """
    def __init__(self, agents, game, start_agent_index=1, stop_condition='context_fill', max_rounds=2,
                 save_folder='data/results', verbosity=2, reverse_agent_order=False):

        os.makedirs(save_folder, exist_ok=True)
        self.save_folder = save_folder
        self.verbosity = verbosity

        self.max_rounds = max_rounds
        self.stop_condition = stop_condition
        self.game = game
        game_shared_description = game.description
        print(game_shared_description)


        self.agents = agents
        for i, agent in enumerate(agents):
            issues_format = game.format_all_issues(i)
            agent_side = self.game.sides[i][0]
            agent.create_static_system_prompt(game_shared_description, agent_side, issues_format)


        # move the agent order to respect start agent index
        self.agents = self.agents if start_agent_index == 0 else [self.agents[1], self.agents[0]]
        self.num_agreed_issues = 0

    def run(self):
        # conduct rounds of negotiation until a stop_condition is hit
        # after each respective agent step, call check_completion()
        completed = False
        round_num = 0
        while not completed:
            ts = []
            t = time.time()
            self._format_round_print(round_num=round_num, total_rounds=self.max_rounds, start=True)
            self.agents[0].step(self.agents[1].msg_history)
            completed = self.check_completion(agent=self.agents[1],
                                              c_msg_history=self.agents[0].msg_history,
                                              num_rounds=round_num)
            self.save_results(agent=self.agents[0], round_num=round_num, agent_id=0)
            ts.append(time.time() - t)
            t = time.time()

            if not completed:
                self.agents[1].step(self.agents[0].msg_history)
                self.save_results(agent=self.agents[1], round_num=round_num, agent_id=1)
                completed = self.check_completion(agent=self.agents[0],
                                                  c_msg_history=self.agents[1].msg_history,
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
        for a in self.agents:
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
        is1 = self.agents[0].get_issues_state()
        is2 = self.agents[1].get_issues_state()

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
        headers = ['agent_name', "agent_id", 'round', 'note', 'message', 'issues_state', 'timestamp','model_name']
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
            data = [agent.agent_name,agent_id, round_num, note, msg, issues_state, timestamp, model_name]
            try:
                writer.writerow(data)
            except Exception as e:
                print(f'warning: failed to write row! - {e}')
