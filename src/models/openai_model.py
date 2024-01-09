from retry import retry
import openai
from attr import define, field

import sys
sys.path.append("src/")
from models.model_utils import ChatModel, BaseMessage, AIMessage, SystemMessage, HumanMessage


@define
class OpenAIModel(ChatModel):
    model_provider: str = field(default='azure')
    model_name: str = field(default='gpt-3.5-turbo')
    role_mapping = field(default={'role': 'role', 'content': 'content', 'assistant': 'assistant', 'user': 'user',
                                  'system': 'system'})
    
    @retry(Exception, tries=5, delay=2, backoff=2)
    def _generate(self, data):

        kwargs = self._preprocess(data)
        response = openai.ChatCompletion.create(**kwargs)
        ai_msg = self._postprocess(response)
            
        return ai_msg

    def _preprocess(self, data) -> dict:
        _messages = [m.prepare_for_generation(role_mapping=self.role_mapping) for m in data]

        kwargs = {
            "messages": _messages,
            "temperature": self.temperature,
        }
        kwargs.update(self.generation_params)

        if self.model_provider == 'azure':
            openai.api_key = self.model_key
            # adjust if we want to use azure as the endpoint
            openai.api_base = self.api_info["api_base"]
            openai.api_type = self.api_info["api_type"]
            openai.api_version = self.api_info["api_version"]
            self.model_name = self.model_name.replace(".", "")  # for the azure naming struct.
            kwargs["engine"] = self.model_name
        else:
            openai.api_key = self.model_key
            kwargs["model"] = self.model_name

        return kwargs

    def _postprocess(self, data):
        content = ""
        try:
            content = data["choices"][0]["message"]["content"]
            self.prompt_tokens = data['usage']['prompt_tokens']
            self.completion_tokens = data['usage']['completion_tokens']
        except Exception as e:
            print(f'[error] failed to generate response - {e}')

        return AIMessage(content)


if __name__ == "__main__":
    messages_ = [SystemMessage("This is fun, right?"), HumanMessage("Test 1, 2, 3.")]
    model_ = OpenAIModel(model_provider='azure')
    print(model_(messages_))
