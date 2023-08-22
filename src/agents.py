import os

import pandas as pd

from model_utils import SystemMessage, HumanMessage, ChatModel
from utils import get_api_key, printv
import re
import json
from attr import define, field

from logger import get_logger
import copy
log = get_logger()


@define
class NegotiationAgent:
    # message params
    msg_prompt: str
    msg_max_len: int
    msg_input_msg_history: str
    msg_input_note_history: str

    # notes params
    note_prompt: str
    note_max_len: int
    note_input_msg_history: int
    note_input_note_history: int

    # model
    model: ChatModel = None
    model_name: str = field(default='gpt-4')
    model_provider: str = field(default='openai')
    model_key_path: str = field(default='secrets.json')
    model_key: str = field(default='OPENAI_API_KEY')
    model_budget: float = field(default=5.)
    temperature: float = field(default=0.)

    # keep track of what agents think they can achieve
    notes_history: list = field(factory=list)
    msg_history: list = field(factory=list)
    achievable_payoffs: dict = field(factory=dict)

    # agent character
    agent_name: str = None
    agent_name_ext: str = None
    internal_description: dict = field(factory=dict)
    external_description: dict = field(factory=dict)
    system_description: SystemMessage = field(default=SystemMessage(''))

    verbosity: int = 1
    debug_mode: bool = False

    init_settings = locals()

    def __attrs_post_init__(self):
        # read in prompts
        self.note_prompt = self.get_msg_note_prompt(self.note_prompt, is_note=True)
        self.msg_prompt = self.get_msg_note_prompt(self.msg_prompt, is_note=False)
        # get api key
        api_key = get_api_key(fname=self.model_key_path, provider=self.model_provider)
        self.model = ChatModel(
            model_name=self.model_name, model_provider=self.model_provider, model_key=api_key,
            temperature=self.temperature, debug_mode=self.debug_mode, budget=self.model_budget,
        )
        self.agent_name = self.internal_description['name']
        self.agent_name_ext = self.external_description['name']

    def copy_agent_history_from_transcript(self, transcript: str, agent_id: int):
        transcript = pd.read_csv(transcript)
        transcript = transcript[transcript['agent_id'] == agent_id]
        self.msg_history = transcript['message'].to_list
        self.notes_history = transcript['note'].to_list

    def generate_message(self, c_msg_history):
        # msg prompt structure
        # system_msg(game, side, agent, issues) + user_msg(c_msg, note, msg, c_msg, note, prompt) -> agent_msg: msg
        user_msg = self.prepare_msg_note_input(c_msg_history)
        printv(f'\nMSG PROMPT\n {user_msg} \n', self.verbosity, 1)
        context = [self.system_description, HumanMessage(user_msg)]
        log.debug(f"Context: {context}")
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

    def prepare_msg_note_input(self, c_msg_history: list, simulate_msg: bool = False, transcript_only: bool = False
                               ) -> str:
        """
        Create input context for writing the next message/note.

        We loop backwards, only added context as far as our 'memory' is allowed to go.
        :param c_msg_history: messages generated by the other agent
        :param simulate_msg: add 'dummy' msg to transcript to measure contact capacity
        :param transcript_only: only return the transcript history
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

        if len(neg_transcript) > 0:
            if first_msg_note:
                neg_history = f'\nA log of your mental notes so far:\n\n' \
                              f'<start transcript>\n {neg_transcript} \n<end transcript>\n'
            else:
                neg_history = f'\nTranscript of ongoing negotiations and your mental notes so far:\n\n' \
                              f'<start transcript>\n{neg_transcript}\n<end transcript>\n\n'
        if transcript_only:
            return neg_history
        next_prompt = self.msg_prompt if msg else self.note_prompt
        if first_msg_note:
            to_replace = "Reflect on the negotiations transcript so far."
            replacement = "You are to start the negotiations."
            next_prompt = next_prompt.replace(to_replace, replacement)

            user_msg = next_prompt

            if len(neg_transcript) > 0:
                user_msg += neg_history
        else:
            user_msg = next_prompt
            if type(neg_history)==list:
                user_msg += "\n".join(neg_history)
            else:
                user_msg += neg_history

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

            state_regex = re.compile(r"{[\s\S]*?}")
            state_str = state_regex.search(last_note)
            if state_str is not None:
                try:
                    state = json.loads(state_str[0])
                    log.debug(f"Issue state for agent {self.agent_name}: {state}")
                except Exception as e:
                    print(f'error: unable to retrieve valid state from notes - {e}')

                if any(state):
                    self.achievable_payoffs = state

        return state

    def set_system_description(self, game, agent_id):
        system_description = game.get_system_msg(agent_id=agent_id, agent_desc_int=self.internal_description)
        self.system_description = SystemMessage(system_description)
        printv(f'\n[system prompt]\n{system_description}\n', self.verbosity, 1)


