from typing import List
import requests
from retry import retry
import sys
sys.path.append("src/")
from attr import define, field
from models.model_utils import ChatModel, BaseMessage, AIMessage, SystemMessage, HumanMessage


@define
class AnthropicModel(ChatModel):
    model_provider: str = field(default='anthropic')
    model_name: str = field(default='claude-2')
    
    @retry(requests.exceptions.RequestException, tries=3, delay=2, backoff=2)
    def _generate(self, data):

        url, headers, package = self._preprocess(data)
        response = requests.post(url, headers=headers, json=package)
        ai_msg = self._postprocess({'response': response, 'prompt': package['prompt']})

        return ai_msg

    def _preprocess(self, messages: List[BaseMessage]):
        # anthropic updated API docs to have the 'major', i.e., claude-2, automatically use the latest minor
        # hence, the first version that used claude-2, now needs to be specified to claude-2.0
        # see: https://docs.anthropic.com/claude/reference/selecting-a-model, for details
        model_name_ = self.model_name
        if model_name_ == "claude-2":
            model_name_ = "claude-2.0"
        _role_mapping = {"user": "Human", "assistant": "Assistant",
                         "system": "Human" if model_name_ == 'claude-2.0' else "System"}
        full_prompt = ""
        for msg in messages:
            full_prompt += f"{_role_mapping.get(msg.role)}: {msg.content}\n\n"
        full_prompt += f"\n\n{_role_mapping.get('assistant')}:"

        package = {"prompt": full_prompt,
                   "model": model_name_,
                   "max_tokens_to_sample": self.context_max_tokens
                   }
        headers = self.api_info["headers"]
        headers["x-api-key"] = self.model_key

        url = self.api_info["api_base"]

        return url, headers, package
    
    def _postprocess(self, data) -> AIMessage:
        content = ""
        try:
            content = data['response'].json()['completion']
            self.prompt_tokens = self.estimate_tokens(data['prompt'])
            self.completion_tokens = self.estimate_tokens(content)
        except Exception as e:
            print(f'error: failed to parse response - {e}')

        return AIMessage(content)


if __name__ == "__main__":
    messages = [SystemMessage("This is fun, right?"), HumanMessage("Test 1, 2, 3.")]
    
    model = AnthropicModel()
    print(model(messages))
