"""
source documentation: https://docs.cohere.com/reference/chat-1
"""
from typing import List
from attr import define, field
import requests
from retry import retry
import sys
sys.path.append("src/")
from .model_utils import ChatModel, BaseMessage, AIMessage, SystemMessage, HumanMessage


@define
class CohereModel(ChatModel):
    model_provider: str = field(default="cohere")
    model_name: str = field(default="command")  # command-light is alternative
    api_endpoint: str = field(default="https://api.cohere.ai/v1/chat")
    role_mapping = field(default={'role': 'role', 'content': 'message', 'assistant': 'CHATBOT', 'user': 'USER'})

    @retry(requests.exceptions.RequestException, tries=3, delay=2, backoff=2)
    def _generate(self, data) -> AIMessage:
        url, headers, package = self._preprocess(data)
        response = requests.post(url=url, headers=headers, json=package)
        ai_msg = self._postprocess(response)

        return ai_msg

    def _preprocess(self, data: List[BaseMessage]) -> (str, dict, dict):
        next_msg = data[-1].content
        _messages = [m.prepare_for_generation(role_mapping=self.role_mapping) for m in data]
        _messages = _messages[:-1]

        url = self.api_endpoint
        headers = {
            "Authorization": f"Bearer {self.model_key}",
            "Content-type": "application/json"
        }

        package = {
            "chat_history": _messages,
            "message": next_msg,
            "temperature": self.temperature
        }

        return url, headers, package

    @retry(requests.exceptions.RequestException, tries=30, delay=2, backoff=2)
    def _postprocess(self, response) -> AIMessage:
        content = ""
        try:
            body = response.json()
            content = body['text']
            self.prompt_tokens = body["token_count"]["prompt_tokens"]
            self.completion_tokens = body["token_count"]["response_tokens"]
        except Exception as e:
            print(f'error: failed to unpack cohere API response - {e}')

        return AIMessage(content)


if __name__ == "__main__":
    messages = [SystemMessage("This is fun, right?"), HumanMessage("Test 1, 2, 3.")]

    model = CohereModel()
    print(model(messages))
