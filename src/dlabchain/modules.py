# in case we want to create our own langchain.
# for asynch calls
# plus langchain is a little confusing
import os 
import requests
from typing import Callable, List, Tuple, Any, Dict, cast, Union
import tiktoken
from abc import ABC
import json


class BaseMessage(ABC):
    def format_prompt(self, before, to):
        self.content = self.content.replace("{"+before+"}", f"{to}")
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
    def __init__(self, openai_api_key:str=None, 
                        max_tokens:int=256, 
                        model_name:str="gpt-3.5-turbo", 
                        temperature:float=0.7,
                        **kwargs) -> None:
        """ChatModel.
        
        If key empty, we will

        Args:
            openai_api_key (_type_, optional): key. Defaults to None.
        """
        if openai_api_key==None: openai_api_key=os.getenv("OPENAI_API_KEY")
        if openai_api_key==None: raise ValueError("No openai key found.")
        
        self.max_tokens = max_tokens
        self.model_name = model_name 
        self.temperature = temperature
        self.generation_params = kwargs
        
        self.enc = None 
                
        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }

        
    def estimate_cost(self, messages: List[BaseMessage], estimated_output_tokens):
        # estimate the cost of making a call given a prompt
        # and expected length
        
        # TODO: allow estimation for strings        
        if self.enc==None: self.enc = tiktoken.encoding_for_model(self.model_name)
        
        all_messages = ""
        for m in messages: all_messages+=m.text()
        
        input_tokens = len(self.enc.encode(all_messages))
        output_tokens = estimated_output_tokens
        
        in_price, out_price = get_model_pricing(self.model_name)
        price = round(input_tokens*in_price + output_tokens*out_price, 2)
        
        return f"${price}"
        
    
    def __call__(self, messages: List[BaseMessage]):
        data = [k.prepare_for_generation() for k in messages]
        data = {
            "model": self.model_name,
            "messages": data
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
        self.response = AIMessage(response.json()["choices"][0]["message"]["content"])
        messages.append(self.response)
        self.messages = messages
        return self.response
    
    def history(self):
        # optionally keep a history of interactions
        return self.messages
    
    def __repr__(self):
        return f'ChatModel("model_name={self.model_name}, max_tokens={self.max_tokens}")'
    

def get_model_pricing(model_name):
    if model_name not in ["gpt-4", "gpt-3.5-turbo"]:
        raise NotImplementedError("Only possible for 'gpt-4' or 'gpt-3.5-turbo'")
    
    input_pricing_dict = {"gpt-4": 0.03, "gpt-3.5-turbo":0.003}
    output_pricing_dict = {"gpt-4": 0.06, "gpt-3.5-turbo": 0.004}
    
    return input_pricing_dict[model_name], output_pricing_dict[model_name]