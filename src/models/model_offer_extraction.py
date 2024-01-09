import json
import yaml
import sys
sys.path.append('src/')
from models.model_utils import ChatModel, BaseMessage, AIMessage, SystemMessage, HumanMessage
from models.openai_model import OpenAIModel
from models.anthropic_model import AnthropicModel
from models.cohere_model import CohereModel
from models.huggingface_model import HuggingFaceModel
from models.google_model import GoogleModel
from utils import extract_dictionary, get_api_key


def do_offer_extraction(data, issues, idx=0, is_note=False, model_provider='azure', model_name='gpt-3.5-turbo'):
    output = None
    if is_note:
        output = extract_dictionary(data)

    if (output is None) or (not is_note):
        fname = f'data/offer_extraction_prompts/{"note" if is_note else "msg"}_offer_extraction.yaml'
        prompts = yaml.safe_load(open(fname))['prompt']
        prompts = [BaseMessage(**pm) for pm in prompts]
        # format task
        formatted_issues = {i['name']: i['payoff_labels'][idx] for i in issues}
        task = f'''issues:
    {formatted_issues}
    {"note:" if is_note else "message:"}
    {data}
        '''
        prompts.append(HumanMessage(task))

        if model_provider in ['azure', 'openai']:
            model_f = OpenAIModel
        elif model_provider == 'anthropic':
            model_f = AnthropicModel
        elif model_provider == 'cohere':
            model_f = CohereModel
        elif model_provider == 'google':
            model_f = GoogleModel
        elif model_provider == 'llama':
            model_f = HuggingFaceModel
        else:
            raise NotImplementedError('feel free to extend to with custom models')

        model_ = model_f(model_name=model_name, model_key=get_api_key(provider=model_provider),
                         model_provider=model_provider)

        try:
            output = model_(prompts)
        except Exception as e:
            output = {}
            print(f'error: failed to extract offers from message - {e}')
        try:
            output = extract_dictionary(output)
        except json.decoder.JSONDecodeError as e:
            print(f"Error on conversion to dictionary - {e}")

        if output is None:
            output = {}
    return output


if __name__ == "__main__":
    issues_ = [{"name": "money", "payoff_labels": [[1, 2, 3, 4], [2, 3, 4, 5]]}]
    sentence_ = "Hello let's do a price of 3, and 50 employees"
    print(do_offer_extraction(sentence_, issues_, is_note=False, model_provider='openai'))
