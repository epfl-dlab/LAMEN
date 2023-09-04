import requests
from typing import List, Union, TypedDict
import tiktoken
from abc import ABC
import json
import yaml
from retry import retry
from utils import get_api_key
import openai
import time
from datetime import datetime as dt
from attr import define, field
from llama import Llama

LLAMA_DIRECTORY = "/home/ubuntu/llama/" 

class BaseMessage(ABC):
    def format_prompt(self, before, to):
        self.content = self.content.replace("{" + before + "}", f"{to}")
        return self

    def prepare_for_generation(self):
        return {"role": self.role, "content": self.content}
    
    def prepare_for_completion(self):
        return self.content

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


@define
class ChatModel:

    """
    Basic LLM API model wrapper class

    # TODO: (1) error handling
    #       (2) expand support
    #       (3) move to aiohttp REST API calls
    #       (4) add streaming mode
    # prepare payload
    # make the call
    # process output
    """
    model_provider: str = 'openai'
    model_name: str = 'gpt-3.5-turbo'
    model_key: str = field(default=None)
    model_key_path: str = field(default='secrets.json')
    model_key_name: str = field(default=None)
    model: str = field(default=None)

    debug_mode: bool = False
    temperature: float = 0.0
    generation_params: dict = field(factory=dict)
    context_max_tokens: int = 256
    prompt_cost: float = 0
    completion_cost: float = 0
    tpm: int = 0
    rpm: int = 0
    api_info: dict = field(factory=dict)
    messages: list = field(factory=list)
    response: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


    # keep track of costs
    budget: float = 10.
    session_prompt_costs: float = 0.
    session_completion_costs: float = 0.

    def __attrs_post_init__(self):

        if self.model_key is None:
            self.model_key = get_api_key(fname=self.model_key_path, provider=self.model_provider,
                                         key=self.model_key_name)
        # get model api details
        model_details = get_model_details(self.model_name)
        print(model_details)
        self.context_max_tokens = model_details['max_tokens']
        print(self.context_max_tokens)
        self.prompt_cost = model_details['prompt_cost']
        self.completion_cost = model_details['completion_cost']
        self.tpm = model_details['tpm']
        self.rpm = model_details['rpm']
        # get api info
        self.api_info = get_api_settings(self.model_provider)

        # only single generations currently implemented
        if self.generation_params.get("n", 1) >= 2:
            raise NotImplementedError("Need to implement for more than one generation.")
        
        # init model if llama
        self.model = self.init_model()

    def __call__(self, messages: List[BaseMessage]):
        """
        Generate tokens.
        """
        data = self.prepare_for_generation(messages, self.model_provider, self.model_name)

        if self.debug_mode:
            time.sleep(0.2)  # small wait to see sensible msg timestamps
            return f"<{dt.strftime(dt.now(), '%H%M%S_%f')}> lorem ipsum dolor sit amet"

        response = self._generate(data)
        # TODO: error handling

        prompt_tokens = self.estimate_tokens(messages=messages)
        # TODO figure out why estimation doesn't work for llama
        completion_tokens = 0


        # keep track of budget and costs
        pc, cc, _ = self.estimate_cost(input_tokens=self.prompt_tokens, output_tokens=self.completion_tokens)
        # TODO figure out why estimation doesn't work for llama
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

    def init_model(self):
        """ In case we want to use a llama or huggingface model.

        Returns:
            _type_: _description_
        """
        if "llama" in self.model_name:
                return Llama.build(
                ckpt_dir=f"{LLAMA_DIRECTORY}{self.model_name}/",
                tokenizer_path=f"{LLAMA_DIRECTORY}tokenizer.model",
                max_seq_len=self.context_max_tokens,
                max_batch_size=1,
            )
        elif self.model_provider == "huggingface":
            # TODO: build an integration for hf models
            pass
        else:
            return None

    def estimate_tokens(self, messages: Union[List[BaseMessage], str]):
        # estimate the cost of making a call given a prompt
        # and expected length
        # @td NOTE: standard 8 tokens are put per message
        if "llama" in self.model_name:
            return 0
        try:
            enc = tiktoken.encoding_for_model(self.model_name)
        except KeyError as e:
            enc = tiktoken.encoding_for_model("gpt-4")
        msg_token_penalty = 8

        if isinstance(messages, list):
            for m in messages:
                all_messages = ""
                if isinstance(m, dict):
                    all_messages += m["content"]
                else:
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
            openai.api_key = self.model_key
            # adjust if we want to use azure as the endpoing
            openai.api_base = self.api_info["api_base"]
            openai.api_type = self.api_info["api_type"]
            openai.api_version = self.api_info["api_version"]
            self.model_name = self._model_name.replace(".", "")  # for the azure naming struct.
            response = openai.ChatCompletion.create(
                engine=self.model_name,
                messages=data,
                temperature=self.temperature,
                **self.generation_params
            )
            try:
                self.response = AIMessage(response["choices"][0]["message"]["content"])
                self.prompt_tokens = response['usage']['prompt_tokens']
                self.completion_tokens = response['usage']['completion_tokens']
            except Exception as e:
                print(f'[error] failed to generate response - {e}')
            return self.response
        
        elif self.model_provider == "openai":
            openai.api_key = self.model_key
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=data,
                temperature=self.temperature,
                **self.generation_params
            )
            try:
                self.response = AIMessage(response["choices"][0]["message"]["content"])
                self.prompt_tokens = response['usage']['prompt_tokens']
                self.completion_tokens = response['usage']['completion_tokens']
            except Exception as e:
                print(f'[error] failed to generate response - {e}')
            
            return self.response
        
        elif self.model_provider == "anthropic":
            api_input = {}
            api_input["prompt"] = data 
            api_input["model"] = self.model_name
            api_input["max_tokens_to_sample"] = self.context_max_tokens
            headers = self.api_info["headers"]
            headers["x-api-key"] = self.model_key
            # try:
            output = requests.post(self.api_info["api_base"], headers=headers, json=api_input).json()
            self.response = AIMessage(output['completion'])
            self.prompt_tokens = self.estimate_tokens(data)
            self.completion_tokens = self.estimate_tokens(self.response.content)
            # except Exception as e:
            # print(f'[error] failed to generate response - {e}')
            
        elif self.model_provider == "meta":
            # prepare AI message of the model outputs
            if "chat" in self.model_name:
                result = self.model.chat_completion(
                    data,
                    max_gen_len=self.context_max_tokens,
                    temperature=self.temperature,
                    top_p=1
                )
            else:
                result = self.model.text_completion(
                    data,
                    max_gen_len=self.context_max_tokens,
                    temperature=self.temperature,
                    top_p=1
                )
            
            self.response = AIMessage(result[0]["generation"]["content"])

            self.prompt_tokens = self.estimate_tokens(data)
            self.completion_tokens = self.estimate_tokens(self.response.content)
            return self.response
        else:
            raise NotImplementedError(f"'{self.model_name}' has not yet been implemented")

    def check_context_len(self, context: List[BaseMessage], max_gen_tokens: int) -> bool:
        """Calculate how many tokens we have left. 
        
        messages: List[system_msg, msg_history, note_history, prev_game_history) + note/msg]
        
        Returns:
            got_space (bool)
        """
        # 1. enough context token space to generate note?
        # 2. enough context token space to generate msg?

        # openai adds 8 default tokens per request¨
        msg_token_penalty = 8
        # context_tokens = self.estimate_tokens(context)
        context_tokens = 0
        tokens_left = self.context_max_tokens - (context_tokens + msg_token_penalty + max_gen_tokens)
        got_space = tokens_left > 0

        return got_space

    def __repr__(self):
        return f'ChatModel("model_name"={self.model_name}, "model_provider"={self.model_provider})'

    @staticmethod
    def prepare_for_generation(messages, model_provider, model_name=None):
        if (model_provider == "openai") or (model_provider == "azure"):
            return [k.prepare_for_generation() for k in messages]
        
        elif model_provider=="anthropic":
            full_prompt = ""
            for msg in messages:
                if (msg.role=="system") or (msg.role=="user"):
                    full_prompt+=f"Human: {msg.content}\n\n"
                else:
                    full_prompt+=f"Assistant: {msg.content}\n\n"
            return full_prompt + "\n\nAssistant:"
        
        elif model_provider=="meta":
            if "chat" in model_name:
                return [k.prepare_for_generation() for k in messages]
            return [k.prepare_for_completion() for k in messages]


def get_model_pricing(model_name):
    model_details = get_model_details(model_name=model_name)
    return model_details['prompt_cost'], model_details['completion_cost']


def get_model_details(model_name, fpath='data/llm_model_details.yaml'):
    try:
        with open(fpath) as f:
            details = yaml.safe_load(f)
    except Exception as e:
        print(f'error: unable to load model details - {e}')
        details = {}

    models = details.keys()
    if model_name not in models:
        raise KeyError(f'error: no details available for model {model_name} - pick one of {models}')

    return details[model_name]


def get_api_settings(api_provider, fpath='data/api_settings/apis.yaml'):
    try:
        with open(fpath) as f:
            details = yaml.safe_load(f)
    except Exception as e:
        print(f'error: unable to load model details - {e}')
        details = {}

    models = details.keys()
    if api_provider not in models:
        raise KeyError(f'error: no details available for model {api_provider} - pick one of {models}')

    return details[api_provider]

def check_text_for_offers(message, issues, model_provider="openai", model_name="gpt-3.5-turbo", model_key=None):
        api_key = get_api_key(provider=model_provider, key=model_key)
        model = ChatModel(model_name=model_name, model_key=api_key)
        issues = ", ".join([issue.name for issue in issues])
        message_offer_prompt = """
In the following message, if an issue is being discussed extract the offer being provided
Format each offer as follows:
{
    "issue_name_0": "<stated offer>",
    "issue_name_1": "<stated offer>",
    ...
}
Make sure that the name of the issue is spelled exactly as provided and that only issues from those that are provided are included.

Example 1: 
Issues: price
Message: After considering your offer of $7,200, I believe we can reach an agreement. How about we settle on $6,800? 
Offers:
{
    "price": "$6,800"
}

Example 2: 
Issues: family employees
Message: Thank you for your offer. How about instead of hiring 5 family employees, we will hire 7. 
Offers:
{
    "family employees": "7"
}
```

Issues: These are the issues being discusses {issues}. 
Message: {message}
Offers:
""".replace("{issues}", issues).replace("{message}", message)
        try:
            output = model([HumanMessage(message_offer_prompt)])
        except Exception as e:
            output = {}
            print(f'error: failed to extract offers from message - {e}')
        return output