# Custom Model: How to evaluate your own LM
___

## No-API Model
To evaluate your own language model you'll need to create a subclass of `ChatModel` (See `src/models/model_utils.py`).
The only required function to implement is: 
- `_generate()`, which takes the current negotiation history as input expects a text response wrapped in `AIMessage`

Additionally, you either need to: define a function which takes the current history and formats it to the context your 
model expects, or specify a `role_mapping` dict that has key mappings for the following:

```json
{
  "role": "<for API models, key pointing to the role>",
  "content": "<for API models, key pointing to the content>",
  "user": "<user/human role key>",
  "assistant": "<ai role key>",
  "system": "<(optional) system role key>"
}
```

For example, a minimal implementation might be:
```python
from attr import define, field
from models.model_utils import ChatModel, AIMessage, HumanMessage, SystemMessage

@define
class YourCustomModel(ChatModel):
    model_provider: str = field(default="my_customs")
    model_name: str = field(default="my_custom")
    api_endpoint: str = field(default="<optional>")
    role_mapping: dict = field(default={})
    
    def _generate(self, data):
        custom_format_used_by_my_model = [m.prepare_for_generation(role_mapping=self.role_mapping) for m in data]
        response = my_model_generation_func(custom_format_used_by_my_model)
        return AIMessage(response)
```
Check out our `src/models/llama_model.py` implementation as an example. 

## API Model
Most models implemented in this repository are called over a third-party API. As such, we tend to implement 
`_preprocess()` and `_postprocess()` functions to make and process API-calls:
```python
from typing import List
from attr import define, field
import requests
from models.model_utils import ChatModel, BaseMessage, AIMessage HumanMessage, SystemMessage

@define
class YourCustomModel(ChatModel):
    model_provider: str = field(default="my_customs")
    model_name: str = field(default="my_custom")
    api_endpoint: str = field(default="<optional>")
    role_mapping: dict = field(default={})

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
        content = your_custom_postprocessing(response)
        return AIMessage(content)
```
Checkout the implementations for viz. Anthropic, Cohere, Google, and OpenAI for inspiration.

## Files to update
You will need to update a small number of files by adding entries providing information about your custom model.

To ensure various safety checks pass (e.g., the context is not overflowing), you'll need to modify 
`data/llm_model_details.yaml` by adding an entry of the form:

```yaml
your_custom_model:
  max_tokens: int
  prompt_cost: float
  completion_cost: float
  rpm: 0  # 'requests per minute' -> only for API based models
  tpm: 0  # 'tokens per minute' -> only for API based models
```

Additionally, you should add an entry to `data/api_settings/apis.yaml`:
```yaml
custom_non_api_model: None
custom_api_model:
  api_base: "https://custom_model_endpoint"
  headers:
    content-type: "application/json"
```

As well as add a .YAML file to the folder `data/model_settings`, e.g., `my_custom_model.yaml`, of the following format:
```yaml
model_provider: <your-model-provider-if-any>
model_name: <your-custom-model-name>
temperature: <float-temperature-default>

model:
  _target_: models.YourCustomModel  # note: this should be the name of the custom model subclass you wrote
  model_provider: ${..model_provider}
  model_name: ${..model_name}
  temperature: ${..temperature}
```

Lastly, the following two functions map models to the appropriate constructors:
1. `src/models/model_offer_extraction.py`, see `do_offer_extraction()`
2. `utils.py`, see `_update_model_constructor_hydra()`

For both functions, add your CustomModel to the if/else statements.

For convenience, you can also add a preset file under `data/model_settings` in the following .YAML format:
```yaml
model_provider: my_customs
model_name: <custom_model_name>
temperature: 0.2 

model:
  _target_: models.YourCustomModel
  model_provider: ${..model_provider}
  model_name: ${..model_name}
  temperature: ${..temperature}
```