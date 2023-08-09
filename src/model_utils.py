import requests
from typing import List, Union
import tiktoken
from abc import ABC
import json
from retry import retry
from utils import get_api_key
import openai
import time
from datetime import datetime as dt


class BaseMessage(ABC):
    def format_prompt(self, before, to):
        self.content = self.content.replace("{" + before + "}", f"{to}")
        return self

    def prepare_for_generation(self):
        return {"role": self.role, "content": self.content}

    def text(self):
        return self.content

    def __str__(self):
        return self.content


class AIMessage(BaseMessage):
    def __init__(self, content):
        super().__init__()
        self.role = "assistant"
        self.content = content

    def __repr__(self):
        return f'AIMessage("content={self.content}")'


class HumanMessage(BaseMessage):
    def __init__(self, content):
        super().__init__()
        self.role = "user"
        self.content = content

    def __repr__(self):
        return f'HumanMessage("content={self.content}")'


class SystemMessage(BaseMessage):
    def __init__(self, content):
        super().__init__()
        self.role = "system"
        self.content = content

    def __repr__(self):
        return f'SystemMessage("content={self.content}")'

    def __add__(self, otherSystem):
        # i think this is the only class that will require adding.
        return SystemMessage(self.content + "\n" + otherSystem.content)


class ChatModel:
    def __init__(self, model_key: str = None, model_key_path=None, model_key_name=None,
                 model_provider: str = "openai", model_name: str = "gpt-3.5-turbo", temperature: float = 0.0,
                 budget=10., debug_mode=False, **kwargs) -> None:
        """
        Basic LLM API model wrapper class

        # TODO: (1) error handling
        #       (2) expand support
        #       (3) move to aiohttp REST API calls
        #       (4) add streaming mode
        """

        # get the correct api-key from environment
        # TODO: improve how we get API keys.
        # @td NOTE: use simple function 'get_api_key()' I added to utils.py
        if model_key is None:
            model_key = get_api_key(fname=model_key_path, key=model_key_name)

        self.model_provider = model_provider
        openai.api_key = model_key

        self.debug_mode = debug_mode
        #  generation params
        self._model_name, self.model_name = model_name, model_name  #
        self.temperature = temperature
        self.generation_params = kwargs

        # get model api details
        model_details = get_model_details(model_name)
        self.context_max_tokens = model_details['max_tokens']
        self.prompt_cost = model_details['prompt_cost']
        self.completion_cost = model_details['completion_cost']
        self.tpm = model_details['tpm']
        self.rpm = model_details['rpm']

        # keep track of costs
        self.budget = budget
        self.session_prompt_costs = 0
        self.session_completion_costs = 0

        #  only single generations currently implemented
        if self.generation_params.get("n", 1) >= 2:
            raise NotImplementedError("Need to implement for more than one generation.")

        # adjust if we want to use azure as the endpoing
        if self.model_provider == "azure":
            openai.api_base = "https://instance0.openai.azure.com/"
            openai.api_type = "azure"
            openai.api_version = "2023-03-15-preview"
            self.model_name = self._model_name.replace(".", "")  # for the azure naming struct.

    def __call__(self, messages: List[BaseMessage]):
        """
        Generate tokens.
        """
        data = [k.prepare_for_generation() for k in messages]

        if self.debug_mode:
            time.sleep(0.2)  # small wait to see sensible msg timestamps
            return f"<{dt.strftime(dt.now(), '%H%M%S_%f')}> lorem ipsum dolor sit amet"

        response = self._generate(data)

        # TODO: error handling
        prompt_tokens = self.estimate_tokens(messages=messages)
        completion_tokens = 0
        try:
            self.response = AIMessage(response["choices"][0]["message"]["content"])
            prompt_tokens = response['usage']['prompt_tokens']
            completion_tokens = response['usage']['completion_tokens']
        except Exception as e:
            print(f'[error] failed to generate response - {e}')

        # keep track of budget and costs
        pc, cc, _ = self.estimate_cost(input_tokens=prompt_tokens, output_tokens=completion_tokens)
        self.session_prompt_costs += pc
        self.session_completion_costs += cc
        self.budget -= (pc + cc)

        messages.append(self.response)
        self.messages = messages

        return self.response.content

    def history(self):
        # optionally keep a history of interactions
        return self.messages

    def estimate_cost(self, input_tokens: int, output_tokens: int):
        """
        Basic cost estimation
        """
        input_cost = (input_tokens / 1000) * self.prompt_cost
        output_cost = (output_tokens / 1000) * self.completion_cost
        total_cost = input_cost + output_cost

        return input_cost, output_cost, total_cost

    def estimate_tokens(self, messages: Union[List[BaseMessage], str]):
        # estimate the cost of making a call given a prompt
        # and expected length
        # @td NOTE: standard 8 tokens are put per message
        enc = tiktoken.encoding_for_model(self.model_name)
        msg_token_penalty = 8

        if isinstance(messages, list):
            all_messages = ""
            for m in messages:
                all_messages += m.text()
        else:
            all_messages = messages

        tokens = len(enc.encode(all_messages)) + msg_token_penalty

        return tokens

    @retry(requests.exceptions.RequestException, tries=3, delay=2, backoff=2)
    def _generate(self, data):
        # refactoring since silly api differences between azure and openai. 
        # in azure engine = model_name, in openai model =...
        # @td NOTE: using the max_tokens param has undesired behavior, specify max token requirements in prompt
        if self.model_provider == "azure":
            return openai.ChatCompletion.create(
                engine=self.model_name,
                messages=data,
                temperature=self.temperature,
                **self.generation_params
            )
        else:
            return openai.ChatCompletion.create(
                model=self.model_name,
                messages=data,
                temperature=self.temperature,
                **self.generation_params
            )

    def check_context_len(self, context: List[BaseMessage], max_gen_tokens: int) -> bool:
        """Calculate how many tokens we have left. 
        
        messages: List[system_msg, msg_history, note_history, prev_game_history) + note/msg]
        
        Returns:
            got_space (bool)
        """
        # 1. enough context token space to generate note?
        # 2. enough context token space to generate msg?

        # openai adds 8 default tokens per request
        msg_token_penalty = 8
        context_tokens = self.estimate_tokens(context)
        tokens_left = self.context_max_tokens - (context_tokens + msg_token_penalty + max_gen_tokens)
        got_space = tokens_left > 0

        return got_space

    def __repr__(self):
        return f'ChatModel("model_name"={self.model_name}, "model_provider"={self.model_provider})'


def get_model_pricing(model_name):
    model_details = get_model_details(model_name=model_name)
    return model_details['prompt_cost'], model_details['completion_cost']


def get_model_details(model_name, fpath='data/llm_model_details.json'):
    try:
        with open(fpath) as f:
            details = json.load(f)
    except Exception as e:
        print(f'error: unable to load model details - {e}')
        details = {}

    models = details.keys()
    if model_name not in models:
        raise KeyError(f'error: no details available for model {model_name} - pick one of {models}')

    return details[model_name]
