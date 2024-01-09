"""
source documentation: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-chat
"""
from typing import List
from attr import define, field
import requests
import jwt
import json
import time
from retry import retry
import sys
sys.path.append("src/")
from .model_utils import ChatModel, BaseMessage, AIMessage, SystemMessage, HumanMessage
from utils import get_api_key


@define
class GoogleModel(ChatModel):
    model_provider: str = field(default='google')
    model_name: str = field(default='chat-bison')
    project_id: str = field(default="llm-eval-negotiating")
    api_endpoint: str = field(default="us-central1-aiplatform.googleapis.com")
    role_mapping = field(default={'role': 'author', 'content': 'content', 'assistant': 'bot', 'user': 'user'})
    
    def __attrs_post_init__(self):
        self._refresh_token()
        self.model_key = get_api_key(fname=self.model_key_path, provider=self.model_provider, key=self.model_key_name)

    @retry(Exception, tries=3, delay=2, backoff=2)
    def _generate(self, data) -> AIMessage:

        url, headers, package = self._preprocess(data)
        response = requests.post(url=url, headers=headers, json=package)
        ai_msg = self._postprocess(response)

        return ai_msg

    def _preprocess(self, data: List[BaseMessage]) -> (str, dict, dict):

        system_msg = ''
        # NOTE 1: moved away from isinstance(msg, SystemMessage) since relative imports might invalidate this evaluation
        # example: src.model_utils.SystemMessage will return False for msg instantiated using model_utils.SystemMessage
        # NOTE 2: Google API expects at least 1 message in the package body -> only parse system message to context if
        # more than 1 message in list
        if len(data) > 1 and data[0].role == SystemMessage("").role:
            system_msg = data[0].content
            data = data[1:]

        _messages = [m.prepare_for_generation(role_mapping=self.role_mapping) for m in data]

        url = f'https://{self.api_endpoint}/v1/projects/{self.project_id}/locations/us-central1/publishers/google/' \
              f'models/{self.model_name}:predict'
        headers = {
            "Authorization": f"Bearer {self.model_key}",
            "Content-type": "application/json"
        }
        package = {
            "instances": [
                {
                    "context": system_msg,
                    "examples": [],
                    "messages": _messages
                }
            ],
            "parameters": {
                "candidateCount": 1,
                "maxOutputTokens": 1024,
                "temperature": self.temperature,
                "topP": 0.8,
                "topK": 5
            }
        }

        return url, headers, package

    def _postprocess(self, response) -> AIMessage:
        body = response.json()
        content = body['predictions'][0]['candidates'][0]['content']
        self.prompt_tokens = body["metadata"]['tokenMetadata']['inputTokenCount']['totalTokens']
        self.completion_tokens = body["metadata"]['tokenMetadata']['outputTokenCount']['totalTokens']

        return AIMessage(content)

    @staticmethod
    def _refresh_token(json_filename: str = 'gcp_secrets.json', expires_in: int = 3600) -> str:
        # https://developers.google.com/identity/protocols/oauth2/service-account
        # https://www.jhanley.com/blog/
        # https://stackoverflow.com/a/53926983/3723434

        scopes = "https://www.googleapis.com/auth/cloud-platform"
        with open(json_filename, 'r') as f:
            cred = json.load(f)

        # Google Endpoint for creating OAuth 2.0 Access Tokens from Signed-JWT
        auth_url = "https://www.googleapis.com/oauth2/v4/token"
        issued = int(time.time())
        expires = issued + expires_in  # expires_in is in seconds

        # JWT Headers
        additional_headers = {
            'kid': cred['private_key'],
            "alg": "RS256",
            "typ": "JWT"  # Google uses SHA256withRSA
        }

        # JWT Payload
        payload = {
            "iss": cred['client_email'],  # Issuer claim
            "sub": cred['client_email'],  # Issuer claim
            "aud": auth_url,  # Audience claim
            "iat": issued,  # Issued At claim
            "exp": expires,  # Expire time
            "scope": scopes  # Permissions
        }

        # Encode the headers and payload and sign creating a Signed JWT (JWS)
        signed_jwt = jwt.encode(payload, cred['private_key'], algorithm="RS256", headers=additional_headers)
        auth_url = "https://www.googleapis.com/oauth2/v4/token"
        params = {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": signed_jwt
        }

        res = requests.post(url=auth_url, data=params)
        try:
            token = res.json()['access_token']
        except Exception as e:
            token = None
            print(f'error: unable to retrieve access token - {e}')
        print('token', token)
        with open('secrets.json', 'r') as f:
            data = json.load(f)

        # update the key
        data['google']['dlab_key'] = token

        # write the new version of the dictionary
        with open('secrets.json', 'w') as f:
            json.dump(data, f, indent=4)

        return token


if __name__ == "__main__":
    print('perform google api test...')
    messages = [SystemMessage("This is fun, right?"), HumanMessage("Test 1, 2, 3.")]
    model = GoogleModel()
    print(model(messages))
