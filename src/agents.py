from uuid import uuid4
import os
import json
import csv
from dlabchain import AIMessage, SystemMessage, HumanMessage, ChatModel

import logging
log = logging.getLogger("my-logger")


def create_agent_description(desc: str, agent_name: str, save_path='data/agent_descriptions', **kwargs):
    # TODO: think of sensible comparison axes, e.g., age, profession, sex, etc.
    agent = locals()
    del agent['save_path']

    with open(os.path.join(save_path, agent_name), 'w') as f:
        json.dump(agent, f)

def load_agent_description(agent_path) -> str:
    # TODO: load agent json and join into single string
    pass


class NegotiationAgent:
    def __init__(self, agent_name=None, init_description=None,
                 msg_prompt=None, note_prompt=None,
                 max_msg_len=64, max_note_len=64,
                 msg_input_msg_history=-1, msg_input_note_history=-1,
                 note_input_msg_history=-1, note_input_note_history=-1,
                 model_name="gpt-4", model_provider='openai', model_temperature=0.,
                 model_key='openai_key', model_key_path=None, verbosity=0) -> None:
        """
        Basic agent class
        """
        self.agent_name = str(uuid4()) if agent_name is None else agent_name
        # TODO: read description from data/agent_descriptions
        self.init_description = load_agent_description(init_description)

        # message params
        self.msg_prompt, self.max_msg_len = msg_prompt, max_msg_len
        self.msg_input_msg_history, self.msg_input_note_history = msg_input_msg_history, msg_input_note_history
        # notes params
        self.note_prompt, self.max_note_len = note_prompt, max_note_len
        self.note_input_msg_history, self.note_input_note_history = note_input_msg_history, note_input_note_history
        # TODO: init model using model params
        self.model = ChatModel()

        self.verbosity = verbosity
        self.notes_history = []
        self.msg_history = []

    def generate_message(self):
        # assume counterparty message history is updated by the NegotiationsProtocol
        # TODO: use msg_input params to determine what the context for the completion inference pass consists of
        msg_history_input = self.msg_history[-self.msg_input_msg_history:]
        notes_history_input = self.notes_history[-self.msg_input_note_history:]

        # TODO: weaving pattern, e.g. system_msg_0, user_msg[note_0, msg_0, note_1 -> msg_prompt]

    def generate_note(self):
        # TODO: use note_input params to determine what the context for the completion inference pass consists of
        msg_history_input = self.msg_history[-self.note_input_msg_history:]
        notes_history_input = self.notes_history[-self.note_input_note_history:]

        # TODO: weaving pattern, e.g. system_msg_0, user_msg[note_0, msg_0, note_1, msg_1 -> note_prompt]

    def get_issues_state(self) -> dict:
        # TODO: get latest issues_proposal from internal notes
        pass

    # might kill
    @staticmethod
    def sync_msg_history(from_agent, to_agent):
        last_msg = from_agent.msg_history[-1]
        to_agent.msg_history.append(last_msg)

    # likely, this could also be a function of the ChatModel() class
    def check_context_len_step(self) -> bool:
        # TODO: get token len of completion input (system_msg, msg_history, note_history, prev_game_history) + note/msg
        # NOTE: openai adds 8 tokens for any request
        # 1. enough context token space to generate note?
        # 2. enough context token space to generate msg?
        got_space = True

        return got_space

    def step(self):
        # create mental note
        self.generate_note()
        # create external message
        self.generate_message()

    def get_settings(self):
        # TODO: return init settings for analysis / saving
        pass


class NegotiationProtocol:
    """
    Run negotiations
    """
    def __init__(self, agents, game, start_agent_index=0, stop_condition='context_fill', max_rounds=1e6,
                 save_folder='data/results'):

        os.makedirs(save_folder, exist_ok=True)
        self.save_folder = save_folder

        self.max_rounds = max_rounds
        self.stop_condition = stop_condition
        self.game = game
        
        # move the agent order to respect start agent index
        self.agents = agents

    def run(self):
        # TODO: conduct rounds of negotiation until a stop_condition is hit
        #       after each respective agent step, call check_completion()
        completed = False
        num_rounds = 0
        while not completed:
            self.agents[0].step()
            completed = self.check_completion(num_rounds=num_rounds)
            self.save_results()
            self.agents[0].sync_msg_history(self.agents[0], self.agents[1])

            if not completed:
                self.agents[1].step()
                completed = self.check_completion(num_rounds=num_rounds)
                self.save_results()
                self.agents[1].sync_msg_history(self.agents[1], self.agents[0])

            num_rounds += 1

    def check_completion(self, num_rounds) -> bool:
        # 1. all issues are agreed upon
        # 2. max rounds reached
        # 3. ran out of context tokens
        #   -> a. stop
        #   -> b. drop history
        #   -> c. fancier stuff, e.g., memory bank search etc.
        completed = False
        # 1
        if self.compare_issues(return_issues=False):
            completed = True
        # 2
        if num_rounds >= self.max_rounds:
            completed = True
        # 3
        if not self.agents[0].check_context_len_step():
            completed = True
        if not self.agents[1].check_context_len_step():
            completed = True

        return completed

    def compare_issues(self, return_issues=False) -> (bool, dict):
        is1 = self.agents[0].get_issues_state()
        is2 = self.agents[1].get_issues_state()

        is_keys = is2.keys()
        agreed_issues = [k for k in is_keys if is1.get(k) == is2.get(k)]
        agreed = len(agreed_issues) == len(is_keys)

        if return_issues:
            return agreed, agreed_issues

        return agreed

    def save_results(self):
        # TODO: save a csv or json row containing:
        # agent; round; note; message; issues_state; timestamp
        headers = ['agent', 'round', 'note', 'message', 'issues_state', 'timestamp']
        csv_path = os.path.join(self.save_folder, 'negotiation.csv')
        pass
