from uuid import uuid4
import os
import json
import csv
from dlabchain import AIMessage, SystemMessage, HumanMessage, ChatModel
from utils import get_api_key
from typing import Tuple
from omegaconf import DictConfig, OmegaConf

import logging
log = logging.getLogger("my-logger")


def create_agent_description(cfg: DictConfig):
    # TODO: think of sensible comparison axes, e.g., age, profession, sex, etc.
    agents = cfg["experiments"]["agents"]
    
    

def load_agent_description(agent_path) -> str:
    # TODO: load agent json and join into single string
    pass

class AgentSystemPrompt:
    def __init__(self, agent, game):
        pass


class NegotiationAgent:
    def __init__(self, agent_name=None, init_description=None,
                 msg_prompt=None, note_prompt=None,
                 msg_max_len=64, note_max_len=64,
                 msg_input_msg_history=-1, msg_input_note_history=-1,
                 note_input_msg_history=-1, note_input_note_history=-1,
                 model_name="gpt-4", model_provider='openai', model_key='OPENAI_API_KEY',
                 model_key_path='secrets.json', **kwargs) -> None:
        """
        Basic agent class
        agent_name
        init_description
        """
        # TODO: read description from data/agent_descriptions
        # self.init_description = load_agent_description(init_description)
        self.init_description = init_description # we load it in as a string unlike ^^^ 
        self.agent_name = str(uuid4()) if agent_name is None else agent_name

        # message params
        self.msg_prompt, self.msg_max_len = msg_prompt, msg_max_len
        self.msg_input_msg_history, self.msg_input_note_history = msg_input_msg_history, msg_input_note_history
        # notes params
        self.note_prompt, self.note_max_len = note_prompt, note_max_len
        self.note_input_msg_history, self.note_input_note_history = note_input_msg_history, note_input_note_history
        # generation parameters
        self.generation_parameters = kwargs
        # model
        api_key = get_api_key(fname=model_key_path, key=model_key)
        self.model = ChatModel(
            model_name=model_name, model_provider=model_provider, model_key=api_key,
            **self.generation_parameters
        )

        self.notes_history = []
        self.msg_history = []

    def generate_message(self):
        msg_history_input = self.msg_history[-self.msg_input_msg_history:]
        notes_history_input = self.notes_history[-self.msg_input_note_history:]
        # msg prompt structure:
        # system_msg(game, side, agent, issues) + user_msg(c_msg, note, msg, c_msg, note, prompt) -> agent_msg: msg
        user_msg = ''  # TODO: call function that weaves msg_history, note_history, msg_len, msg_prompt
        context = self.system_skeleton + user_msg
        msg = self.model(context)
        return msg

    def generate_note(self):
        # TODO: use note_input params to determine what the context for the completion inference pass consists of
        msg_history_input = self.msg_history[-self.note_input_msg_history:]
        notes_history_input = self.notes_history[-self.note_input_note_history:]

        # note prompt structure:
        # system_msg(game, side, agent, issues) + user_msg(c_msg, note, msg, c_msg, prompt) --> agent_msg: note
        user_msg = ''  # TODO: call function that weaves msg_history, note_history, note_len, note_prompt
        context = self.system_skeleton + user_msg
        note = self.model(context)
        return note

    def get_issues_state(self) -> dict:
        # TODO: get latest issues_proposal from internal notes
        pass
    
    def __repr__(self):
        return f"{self.init_description}"

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
        # not clear how to make a scalable approach to max token length across dif models
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
    
    def create_static_system_prompt(self, shared_description, side_description, issues_format):
        intial_story = shared_description +"\n"+ side_description +  "\nDescription of your qualities:\n" + self.init_description + "\n" + issues_format
        self.system_skeleton = SystemMessage(intial_story)
        print(intial_story)


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
        game_shared_description = game.description
        
        # get issues for each agent

        # move the agent order to respect start agent index
        self.agents = agents


        for i, agent in enumerate(agents):
            issues_format = game.format_all_issues(i)
            agent_side = self.game.sides[i][0]
            agent.create_static_system_prompt(game_shared_description, agent_side, issues_format)
        
        
            

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

    def compare_issues(self, return_issues=False) -> Tuple[bool, dict]:
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