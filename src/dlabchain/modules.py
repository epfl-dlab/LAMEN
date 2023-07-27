import os
import requests
from typing import Callable, List, Tuple, Any, Dict, cast, Union
import tiktoken
from abc import ABC
import json
from retry import retry
# TODO: fix import structure, from utils import get_api_key
import openai


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


# TODO: azure and openai work slightly different, make sure both are supported for v1
# TODO: assume a local secrets.json file that hosts keys to avoid accidental version control commits
# ^ Right now we are using .env files to store API keys. 
class ChatModel:
    def __init__(self, model_key: str = None, model_key_path=None, model_key_name=None,
                 model_provider: str = "openai",
                 model_name: str = "gpt-3.5-turbo", temperature: float = 0.0, max_tokens=None,
                 **kwargs) -> None:
        """ChatModel.
        
        If key empty, we will

        Args:
            openai_api_key (_type_, optional): key. Defaults to None.
        """

        # get the correct api-key from environment
        # TODO: improve how we get API keys.
        # @td NOTE: use simple function 'get_api_key()' I added to utils.py
        if model_key is None:
            # TODO: use get_api_key from utils
            if (model_key == None) & (model_provider=="openai"): openai_api_key = os.getenv("OPENAI_API_KEY")
            if (model_key == None) & (model_provider=="azure"): openai_api_key = os.getenv("AZURE_API_KEY")
            if model_key == None: raise ValueError("No openai key found.")
    
        self.model_provider = model_provider
        openai.api_key = model_key

        # generation params
        self.max_tokens = max_tokens  # NOTE: remove
        self._model_name, self.model_name = model_name, model_name # 
        self.temperature = temperature
        self.generation_params = kwargs

        # get model api details
        model_details = get_model_details(model_name)
        self.context_max_tokens = model_details['max_tokens']
        self.prompt_cost = model_details['prompt_cost']
        self.completion_cost = model_details['completion_cost']
        self.tpm = model_details['tpm']
        self.rpm = model_details['rpm']
        
        # only single generations currently implemented
        if self.generation_params.get("n", 1) >= 2:
            raise NotImplementedError("Need to implement for more than one generation.")

        self.enc = None         # used for measuring cost of generation below

        # adjust if we want to use azure as the endpoing
        if self.model_provider=="azure":
            openai.api_base = "https://instance0.openai.azure.com/"
            openai.api_type = "azure"
            openai.api_version = "2023-03-15-preview"
            self.model_name = self._model_name.replace(".","") # for the azure naming struct.

    @retry(requests.exceptions.RequestException, tries=3, delay=2, backoff=2)
    def __call__(self, messages: List[BaseMessage]):
        """
        Generate tokens.
        """
        data = [k.prepare_for_generation() for k in messages]
        response = self._generate(data)

        self.response = AIMessage(response["choices"][0]["message"]["content"])

        messages.append(self.response)
        self.messages = messages

        return self.response.content

    def history(self):
        # optionally keep a history of interactions
        return self.messages

    def __repr__(self):
        return f'ChatModel("model_name={self.model_name}, max_tokens={self.max_tokens}")'

    def estimate_cost(self, messages: Union[List[BaseMessage],str], estimated_output_tokens):
            """
            Basic cost estimation
            """
            input_tokens = self.estimate_tokens(messages)
            output_tokens = estimated_output_tokens

            price = round((input_tokens * self.prompt_cost + output_tokens * self.completion_cost) / 1000, 4)

            return f"${price} for {input_tokens} input tokens and {output_tokens} output tokens"
        
    def estimate_tokens(self, messages: Union[List[BaseMessage],str]):
        # estimate the cost of making a call given a prompt
        # and expected length
        # @td NOTE: standard 8 tokens are put per message
        if self.enc == None: self.enc = tiktoken.encoding_for_model(self.model_name)

        if type(messages)==list:
            all_messages = ""
            for m in messages: all_messages += m.text()
        else: all_messages = messages
        input_tokens = len(self.enc.encode(all_messages))
        return input_tokens

    def _generate(self, data):
        # refactoring since silly api differences between azure and openai. 
        # in azure engine = model_name, in openai model =...
        # @td NOTE: using the max_tokens param has undesired behavior, specify max token requirements in prompt
        if self.model_provider=="azure":
            return openai.ChatCompletion.create(
            engine=self.model_name, 
            messages=data,
            temperature=self.temperature,
            # max_tokens=self.max_tokens,
            **self.generation_params
        )
        else:
            return openai.ChatCompletion.create(
            model=self.model_name, 
            messages=data,
            temperature=self.temperature,
            # max_tokens=self.max_tokens,
            **self.generation_params
        )

    def check_context_len(self, messages: List[BaseMessage], max_tokens: int) -> bool:
        """Calculate how many tokens we have left. 
        
        messages: List[system_msg, msg_history, note_history, prev_game_history) + note/msg]
        
        Returns:
            got_space (bool)
        """
        # TODO: get token len of completion input (system_msg, msg_history, note_history, prev_game_history) + note/msg
        # NOTE: openai adds 8 tokens for any request
        # 1. enough context token space to generate note?
        # 2. enough context token space to generate msg?   
        
        # we can easily add a test for this.
        tokens_left = self.context_max_tokens - 8 - self.estimate_tokens(messages) - max_tokens
        got_space = True if tokens_left > 0 else False
        return got_space

def get_model_pricing(model_name):
    model_details = get_model_details(model_name=model_name)
    return model_details['prompt_cost'], model_details['completion_cost']


def get_model_details(model_name, fpath='data/llm_model_details.json'):

    try:
        with open(fpath) as f:
            details = json.load(f)
    except Exception as e:
        print(f'error: unable to load model details')
        details = {}

    models = details.keys()
    if model_name not in models:
        raise KeyError(f'error: no details available for model {model_name} - pick one of {models}')

    return details[model_name]