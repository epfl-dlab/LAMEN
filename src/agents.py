import os
import pandas as pd
from typing import Any, List
from attr import define, field

from utils import printv
from models.model_utils import SystemMessage, HumanMessage, AIMessage, BaseMessage
from logger import get_logger

log = get_logger()


@define
class NegotiationAgent:
    # message params
    msg_prompt: str = 'prompt_0'
    msg_max_len: int = 64
    msg_input_msg_history: int = -1
    msg_input_note_history: int = 1

    # notes params
    note_prompt: str = 'prompt_0'
    note_max_len: int = 64
    note_input_msg_history: int = -1
    note_input_note_history: int = -1

    # model
    model: Any = None
    model_name: str = field(default='gpt-3.5-turbo')
    model_provider: str = field(default='openai')
    model_key_path: str = field(default='secrets.json')
    model_key: Any = field(default='dlab_key')
    model_budget: float = field(default=5.)
    temperature: float = field(default=0.)
    
    # visibility of other sides payoffs
    visibility: int = field(default=0)

    # keep track of what agents think they can achieve
    notes_history: list = field(factory=list)
    msg_history: list = field(factory=list)
    achievable_payoffs: dict = field(factory=dict)
    agreement_prompt: str = 'We agree on all issues.'

    # agent character
    agent_name: str = 'You'
    agent_name_ext: str = 'Representative'
    agent_id: int = 0
    internal_description: dict = field(factory=dict)
    external_description: dict = field(factory=dict)
    system_description: SystemMessage = field(default=SystemMessage(''))

    # format negotiation as if the 'user msg' is the other NegotiatingAgent, and the 'system msg' provides prompts
    format_as_dialogue: bool = field(default=False)
    verbosity: int = field(default=1)
    debug_mode: bool = field(default=False)

    show_rounds: bool = True
    _current_round_num: int = field(default=0)
    _max_rounds: int = field(default=0)
    _note_prompt_filename: str = field(default=None) 
    _msg_prompt_filename: str = field(default=None)

    init_settings = locals()

    def __attrs_post_init__(self):
        self._note_prompt_filename = self.note_prompt 
        self._msg_prompt_filename = self.msg_prompt 
        # read in prompts
        self.note_prompt = self.get_msg_note_prompt(self._note_prompt_filename, is_note=True)
        self.msg_prompt = self.get_msg_note_prompt(self._msg_prompt_filename, is_note=False)
        self.model.debug_mode = self.debug_mode
        self.model.budget = self.model_budget

        self.agent_name = self.internal_description['name']
        self.agent_name_ext = self.external_description['name']

    def copy_agent_history_from_transcript(self, transcript: str, agent_id: int):
        transcript = pd.read_csv(transcript)
        transcript = transcript[transcript['agent_id'] == agent_id]
        self.msg_history = [(self.agent_name, m) for m in transcript['message'].to_list()]
        self.notes_history = [(self.agent_name, m) for m in transcript['note'].to_list()]

    def _verbose_generative_print(self, dialogue_context, is_note=False):
        if self.format_as_dialogue:
            x = ['<dialogue style>']
            for c in dialogue_context:
                x.append(f'[{c.role}]\n{c.content}')
            x = '\n'.join(x)
        else:
            x = dialogue_context[0].content

        gen_type = "NOTE" if is_note else "MSG"
        printv(f'\n[{gen_type} PROMPT]\n {x} \n', self.verbosity, 1, c=self._get_print_color(),
               debug=self.verbosity == 2)

    def _correct_transcript_artifact(self, content) -> str:
        # TODO: log these corrections, as they technically break the instructions
        # due to being presented a 'transcript' of the ongoing negotiation, some models try to respond using their
        # transcript tag, e.g. 'You: I disagree', instead of just 'I disagree'.
        # this can be potentially very confusing to the next chat model trying to interpret the transcript
        if not isinstance(content, str):
            return content

        self_check_1 = f'{self.agent_name}:'
        self_check_2 = f'{self.agent_name_ext}:'
        if self_check_1 in content[:len(self_check_1)]:
            return content[len(self_check_1):]
        elif self_check_2 in content[:len(self_check_2)]:
            return content[len(self_check_2):]
        else:
            return content

    def _get_print_color(self):
        return ['blue', 'red'][self.agent_id]

    def generate_message(self, c_msg_history):
        # msg prompt structure
        # system_msg(game, side, agent, issues) + user_msg(c_msg, note, msg, c_msg, note, prompt) -> agent_msg: msg
        dialogue_context = self.prepare_msg_note_input(c_msg_history)
        context = self._prepare_gen_context(dialogue_context)
        self._verbose_generative_print(context, is_note=False)
        msg = self.model(context)
        msg = self._correct_transcript_artifact(msg)
        self.msg_history.append((self.agent_name, msg))

        printv(f'<{self.agent_name_ext} - msg>: {msg}', self.verbosity, c=self._get_print_color())

        return msg

    def generate_note(self, c_msg_history):
        # note prompt structure:
        # system_msg(game, side, agent, issues) + user_msg(c_msg, note, msg, c_msg, prompt) --> agent_msg: note
        dialogue_context = self.prepare_msg_note_input(c_msg_history)
        context = self._prepare_gen_context(dialogue_context)
        self._verbose_generative_print(context, is_note=True)
        note = self.model(context)
        note = self._correct_transcript_artifact(note)
        self.notes_history.append((self.agent_name, note))

        printv(f'\n<{self.agent_name_ext} - note>: {note}\n', self.verbosity, c=self._get_print_color())

        return note

    def _prepare_gen_context(self, dialogue_context):
        if self.model_provider == 'google' and self.format_as_dialogue:
            # NOTE: google-dialogue will only allow a note-history of 1 for messages, and 0 for notes
            # google rules:
            # 1. only 'odd' number of messages, i.e., first-and-last message must be of type HumanMessage
            # 2. only 2-authors, i.e., meta-instructions must be in system-context (SystemMessage)
            # 3. must be alternating: never have two messages of same type repeating
            system_context = self.system_description.copy()
            # most recent message will always be the next prompt
            next_prompt = dialogue_context[-1].content
            # remove from the dialogue history
            dialogue_context = dialogue_context[:-1]
            first_dummy_msg = HumanMessage('Let us start the negotiations')
            if len(dialogue_context) > 0:
                # either: first c_msg OR first note
                first_msg = dialogue_context[0].copy()
                last_msg = dialogue_context[-1].copy()

                if not isinstance(first_msg, HumanMessage):
                    # first message needs to be human
                    dialogue_context = [first_dummy_msg] + dialogue_context
                if isinstance(last_msg, AIMessage):
                    # google does not allow non-alternating message dialogue, i.e. AIMessage -> AIMessage not allowed
                    # previous generation must be a note -> concat to system_msg and remove from dialogue
                    note = f'Also keep in mind the mental note you wrote:\n"{last_msg.content}"\n'
                    dialogue_context = dialogue_context[:-1]
                    system_context = SystemMessage(f'{system_context.content}\n\n{next_prompt}\n\n{note}')
                elif isinstance(last_msg, HumanMessage):
                    # last message was a c_msg, next message will be a note
                    system_context = SystemMessage(f'{system_context.content}\n\n{next_prompt}')
            else:
                # no notes/messages have been written yet -> this will be the first
                dialogue_context = [first_dummy_msg]
                system_context = SystemMessage(f'{system_context.content}\n\n{next_prompt}')

            context = [system_context] + dialogue_context

        else:
            context = [self.system_description] + dialogue_context

        return context

    def prepare_msg_note_input(self, c_msg_history: list, simulate_msg: bool = False, transcript_only: bool = False
                               ) -> List[BaseMessage]:
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
        # is this the first message/note?

        first_msg_note = (len(n_h) == 0 or len(m_h) == 0) and len(cm_h) == 0
        # IF need to simulate msg create phantom note --> needed to check for context overflow
        n_h = self._format_context_simulation(simulate_msg=simulate_msg, notes_history=n_h)
        # more notes than messages indicates next generation will be a message
        msg = len(n_h) > len(m_h)
        # select part of dialogue history to present in the context
        neg_history = self._select_dialogue_history(next_is_msg=msg, notes_history=n_h, msg_history=m_h,
                                                    c_msg_history=cm_h)
        # format history as either (1) ai_msg-human_msg sequence (format_as_dialogue), or (2) human_msg transcript
        neg_transcript = self._format_dialogue_history(neg_history)

        # if no notes appear in the transcript -> do not mention them to avoid unnecessary confusion
        neg_transcript = self._format_dialogue_transcript_preamble(next_is_msg=msg, is_first_msg_note=first_msg_note,
                                                                   neg_transcript=neg_transcript)
        # if 'interrogation' mode is on, we only want the negotiation history so far, since we're providing a prompt
        if transcript_only:
            return neg_transcript if self.format_as_dialogue else [HumanMessage(neg_transcript)]

        generative_input = self._format_next_prompt(is_msg=msg, is_first_msg_note=first_msg_note,
                                                    dialogue=neg_transcript)
        return generative_input

    def _format_context_simulation(self, simulate_msg, notes_history):
        if simulate_msg:
            if len(notes_history) > 0:  # repeat previous note
                notes_history.append(notes_history[-1])
            else:  # if no note history exists, create a dummy note
                dummy_note = "hello " * self.note_max_len
                notes_history.append((self.agent_name, dummy_note))
        return notes_history

    def _select_dialogue_history(self, next_is_msg, notes_history, msg_history, c_msg_history):
        n_h, m_h, cm_h = notes_history, msg_history, c_msg_history
        max_len = max(max(len(n_h), len(m_h)), len(cm_h))
        max_m = self.msg_input_msg_history if next_is_msg else self.note_input_msg_history
        max_n = self.msg_input_note_history if next_is_msg else self.note_input_note_history
        if max_m < 0:
            max_m = 1e5
        if max_n < 0:
            max_n = 1e5

        neg_history = []
        for i in range(1, max_len + 1):

            if next_is_msg and (i <= max_n) and (0 < len(n_h)):
                neg_history.append(('mental note', n_h.pop()))

            if i <= max_m:
                if 0 < len(cm_h):
                    neg_history.append(('offer', cm_h.pop()))
                if 0 < len(m_h):
                    neg_history.append(('offer', m_h.pop()))

            if not next_is_msg and (i <= max_n) and (0 < len(n_h)):
                neg_history.append(('mental note', n_h.pop()))

            if (i == max_len) and (0 < len(cm_h)) and (i <= max_m):
                neg_history.append(('offer', cm_h.pop()))
        # reverse order to have the oldest message/note appear first
        neg_history = neg_history[::-1]

        return neg_history

    def _format_dialogue_history(self, neg_history_):
        neg_transcript = [] if self.format_as_dialogue else ''
        for i_type, (a, i) in neg_history_:
            if self.format_as_dialogue:
                base_msg = AIMessage(i, alt_role=self.agent_name_ext) if a == self.agent_name else HumanMessage(i)
                neg_transcript.append(base_msg)
            else:
                if i_type == 'mental note':
                    neg_transcript += f'\n{a}:  ({i_type}) {i}'
                else:
                    neg_transcript += f'\n{a}:  {i}'

        return neg_transcript

    def _format_dialogue_transcript_preamble(self, next_is_msg, is_first_msg_note, neg_transcript):
        # if no notes appear in the transcript -> do not mention them to avoid unnecessary confusion
        notes_invisible = (next_is_msg and self.msg_input_note_history == 0) or \
                          (not next_is_msg and self.note_input_note_history == 0)
        if not self.format_as_dialogue and len(neg_transcript) > 0:
            if is_first_msg_note:
                if next_is_msg and not notes_invisible:
                    neg_transcript = f'\nA log of your mental notes so far:\n\n' \
                                     f'<start transcript>\n {neg_transcript} \n<end transcript>\n'
            else:
                transcript_pre_amble = '\nTranscript of ongoing negotiations and your mental notes so far:\n'
                # do not mention mental notes if none will be present
                if notes_invisible:
                    transcript_pre_amble.replace('and your mental notes ', '')
                neg_transcript = f'{transcript_pre_amble}\n<start transcript>\n{neg_transcript}\n<end transcript>\n\n'

        return neg_transcript

    def _format_next_prompt(self, is_msg, is_first_msg_note, dialogue):
        self.note_prompt = self.get_msg_note_prompt(self._note_prompt_filename, is_note=True)
        self.msg_prompt = self.get_msg_note_prompt(self._msg_prompt_filename, is_note=False)
        # Add prompt to generate next note/message
        next_prompt = self.msg_prompt if is_msg else self.note_prompt
        agreement_prompt = f"\nOnly if you agree with the other party on acceptable offers for all issues " \
                           f"respond with: '{self.agreement_prompt}', and nothing else."
        if is_first_msg_note:
            to_replace = "Reflect on the negotiations transcript so far"
            replacement = "You are to start the negotiations"
            next_prompt = next_prompt.replace(to_replace, replacement)
        else:
            if is_msg:
                next_prompt += agreement_prompt

        if self.format_as_dialogue:
            # remove mentions of a 'transcript' for dialogue format
            next_prompt = next_prompt.replace('negotiations transcript', 'negotiations')
            if self.model_provider == 'google':
                next_prompt = HumanMessage(next_prompt)
            else:
                next_prompt = SystemMessage(next_prompt)
            generative_input = dialogue + [next_prompt] if len(dialogue) > 0 else [next_prompt]
        else:
            next_prompt = next_prompt + dialogue if len(dialogue) > 0 else next_prompt
            generative_input = [HumanMessage(next_prompt)]

        return generative_input

    def step(self, c_msg_history, round_num, max_rounds):
        # log round num and max rounds
        self._current_round_num = round_num 
        self._max_rounds = max_rounds
        # create mental note
        self.generate_note(c_msg_history)
        # create external message
        new_msg = self.generate_message(c_msg_history)
        return new_msg

    def check_context_len_step(self, c_msg_history) -> bool:
        # 1. enough context token space to generate note?
        # 2. enough context token space to generate msg?
        dialogue_context = self.prepare_msg_note_input(c_msg_history)
        context = [self.system_description] + dialogue_context
        capacity = self.model.check_context_len(context=context, max_gen_tokens=self.note_max_len)
        if not capacity:
            return capacity

        dialogue_context = self.prepare_msg_note_input(c_msg_history, simulate_msg=True)
        context = [self.system_description] + dialogue_context
        capacity = self.model.check_context_len(context=context, max_gen_tokens=self.msg_max_len)
        
        return capacity

    def get_settings(self):
        return self.init_settings

    def get_msg_note_prompt(self, prompt_name, is_note=False):
        prompt_path = 'data/note_prompts' if is_note else 'data/message_prompts'
        max_len = self.note_max_len if is_note else self.msg_max_len
        if not prompt_name.endswith('.txt'):
            prompt_name += '.txt'
        prompt = open(os.path.join(prompt_path, prompt_name)).read()
        if self.show_rounds:
            prompt = f"""You are currently on round {self._current_round_num}. 
You only have {self._max_rounds} to reach an agreement otherwise the payoff will be 0. \n\n""" + prompt
        if is_note:
            if max_len > 0:
                prompt += f'\nYour note can never exceed {max_len} words.\n\n'
            prompt += open(os.path.join(prompt_path, 'note_standard_suffix.txt')).read()
        else:
            if max_len > 0:
                prompt += f'\nYour offer can never exceed {max_len} words.'

        return prompt

    def set_system_description(self, game, agent_id, other_agent_desc=None):
        system_description = game.get_system_msg(agent_id=agent_id, 
                                                 agent_desc_int=self.internal_description,
                                                 other_agent_desc=other_agent_desc,
                                                 visibility=self.visibility)
        self.system_description = SystemMessage(system_description)

        printv(f'\n[system prompt]\n{system_description}\n', self.verbosity, 1, c=self._get_print_color())

    def update_external_description(self, game, agent_id):
        """When the agent plays the role of representative, and the game specifies party names, update external name"""
        if self.agent_name_ext == "Representative":
            try:
                party = game.parties[agent_id]
                self.agent_name_ext = f'{self.agent_name_ext} {party}'
                self.external_description['name'] = self.agent_name_ext
            except Exception as e:
                print(f'warning: unable to update external description - {e}')

    def __getitem__(self, key):
        return getattr(self, key)
