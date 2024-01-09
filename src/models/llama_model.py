import requests
from typing import List
from retry import retry
import sys
sys.path.append("src/")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .llama import Llama
from attr import define, field
from .model_utils import ChatModel, BaseMessage, AIMessage, SystemMessage, HumanMessage

LLAMA_DIRECTORY = "/ada1/projects/models/llama/"


@define
class LlamaModel(ChatModel):
    model_provider: str = field(default='meta')
    model_name: str = field(default='llama-2-7')
    role_mapping = field(default={'role': 'role', 'content': 'content', 'assistant': 'assistant', 'user': 'user',
                                  'system': 'system'})

    def __attrs_post_init__(self):
        self.model = self.init_model()

    @retry(requests.exceptions.RequestException, tries=3, delay=2, backoff=2)
    def _generate(self, data):

        kwargs = self._preprocess(data)
        if "chat" in self.model_name:
            result = self.model.chat_completion(**kwargs)
        else:
            result = self.model.text_completion(**kwargs)

        prompt = kwargs['dialogs'] if 'chat' in self.model_name else kwargs['prompts']
        ai_msg = self._postprocess({'result': result, 'prompt': prompt})

        return ai_msg

    def _preprocess(self, data: List[BaseMessage]):

        kwargs = {"max_gen_len": self.context_max_tokens,
                  "temperature": self.temperature,
                  "top_p": 1
                  }
        messages = [[m.prepare_for_generation(role_mapping=self.role_mapping) for m in data]]
        if 'chat' in self.model_name:
            kwargs['dialogs'] = messages
        else:
            kwargs['prompts'] = messages

        return kwargs

    def _postprocess(self, data) -> AIMessage:
        content = ""
        try:
            content = data['result'][0]["generation"]["content"]
            self.prompt_tokens = self.estimate_tokens(data['prompt'])
            self.completion_tokens = self.estimate_tokens(content)
        except Exception as e:
            print(f'error: failed to generate response - {e}')

        return AIMessage(content)

    def init_model(self):
        """ In case we want to use a llama or huggingface model.
        """
        return Llama.build(
            ckpt_dir=f"{LLAMA_DIRECTORY}{self.model_name}/",
            tokenizer_path=f"{LLAMA_DIRECTORY}tokenizer.model",
            max_seq_len=4096,
            max_batch_size=1,
        )


if __name__ == "__main__":
    messages = [SystemMessage("This is fun, right?"), HumanMessage("Test 1, 2, 3.")]
    model = LammaModel()
    print(model(messages))
