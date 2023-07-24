import os
import requests
from typing import Callable, List, Tuple, Any, Dict, cast, Union
import tiktoken
from abc import ABC
import json
from retry import retry

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
    def __init__(self, openai_api_key: str = None,
                 api_endpoint: str = "openai",
                 max_tokens: int = 256,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.7,
                 **kwargs) -> None:
        """ChatModel.
        
        If key empty, we will

        Args:
            openai_api_key (_type_, optional): key. Defaults to None.
        """

        # get the correct api-key from environment
        # TODO: improve how we get API keys.
        if (openai_api_key == None) & (api_endpoint=="openai"): openai_api_key = os.getenv("OPENAI_API_KEY")
        if (openai_api_key == None) & (api_endpoint=="azure"): openai_api_key = os.getenv("AZURE_API_KEY")
        if openai_api_key == None: raise ValueError("No openai key found.")
    
        self.api_endpoint = api_endpoint
        openai.api_key = openai_api_key

        # generation params
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.temperature = temperature
        self.generation_params = kwargs
        
        # only single generations currently implemented
        if self.generation_params.get("n", 1) >= 2:
            raise NotImplementedError("Need to implement for more than one generation.")

        self.enc = None         # used for measuring cost of generation below

        # adjust if we want to use azure as the endpoing
        if self.api_endpoint=="azure":
            openai.api_base = "https://instance0.openai.azure.com/"
            openai.api_type = "azure"
            openai.api_version = "2023-03-15-preview"
            self.model_name = self.model_name.replace(".","") # for the azure naming struct.

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
            Basic cost estimatino
            """
            # estimate the cost of making a call given a prompt
            # and expected length
            if self.enc == None: self.enc = tiktoken.encoding_for_model(self.model_name)

            if type(messages)==list:
                all_messages = ""
                for m in messages: all_messages += m.text()
            else: all_messages = messages

            input_tokens = len(self.enc.encode(all_messages))
            output_tokens = estimated_output_tokens

            in_price, out_price = get_model_pricing(self.model_name)
            price = round((input_tokens * in_price + output_tokens * out_price) / 1000, 4)

            return f"${price} for {input_tokens} input tokens and {output_tokens} output tokens"

    def _generate(self, data):
        # refactoring since silly api differences between 
        # azure and openai. 
        # in azure engine = model_name, in openai its model =...            
        if self.api_endpoint=="azure":
            return openai.ChatCompletion.create(
            engine=self.model_name, 
            messages=data,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.generation_params
        )
        else:
            return openai.ChatCompletion.create(
            model=self.model_name, 
            messages=data,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.generation_params
        )

def get_model_pricing(model_name):
    # these are currently hard-coded. maybe easier to get from api or web.
    if model_name not in ["gpt-4", "gpt-3.5-turbo"]:
        raise NotImplementedError("Only possible for 'gpt-4' or 'gpt-3.5-turbo'")

    input_pricing_dict = {"gpt-4": 0.03, "gpt-3.5-turbo": 0.003}
    output_pricing_dict = {"gpt-4": 0.06, "gpt-3.5-turbo": 0.004}

    return input_pricing_dict[model_name], output_pricing_dict[model_name]
