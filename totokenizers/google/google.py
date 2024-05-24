from typing import Sequence, Optional, Mapping

import dotenv
import json
from ..schemas import (
    ChatMLMessage,
    FunctionCallChatMLMessage,
    FunctionChatMLMessage,
    Chat,
)
import vertexai
from vertexai.preview.generative_models import (
    GenerativeModel,
)
from google.api_core.exceptions import GoogleAPIError
from google.oauth2 import service_account
from google.auth.exceptions import GoogleAuthError
from ..errors import ModelNotSupported, CredentialsNotFound, InvalidCredentials
from binascii import Error as AsciiError
from pyasn1.error import PyAsn1Error
import os

dotenv.load_dotenv()


class GoogleTokenizer:
    """
    Tokenizer for Google's Gemini models (Vertex AI API).

    WARNING: you neeed to set up Google Cloud authentication before using this tokenizer.
    See the references below for more information.
    - https://googleapis.dev/python/google-api-core/latest/auth.html
    - https://cloud.google.com/docs/authentication/application-default-credentials

    Args:
        model_name (str): The model name of the tokenizer.

    Reference for token count via SDK and REST API:
    - https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/get-token-count

    Reference for messages requests:
    - https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-chat-prompts-gemini?hl=pt-br
    - https://cloud.google.com/vertex-ai/generative-ai/docs/chat/chat-prompts?hl=pt-br#gemini-1.0-pro
    """

    def __init__(self, model_name: str):
        # Initialize the Vertex AI API (gets Google credentialsl as well)

        if model_name in {
            "gemini-1.0-pro-001",
            "gemini-1.0-pro-002",
            "gemini-pro-vision-001",
        }:
            self.model = GenerativeModel(model_name)
            self.load_gemini()

        else:
            raise ModelNotSupported(model_name)

    def load_gemini(self):
        try:
            credetials_info = {
                "private_key": os.environ["private_key"].replace("\\n", "\n"),
                "client_email": os.environ["client_email"],
                "token_uri": os.environ["token_uri"],
            }
        except KeyError as e:
            raise CredentialsNotFound(credential=e)
        try:
            credentials = service_account.Credentials.from_service_account_info(
                credetials_info
            )
            vertexai.init(project=os.environ.get("project_id"), credentials=credentials)
            self.model.count_tokens("test")
        except GoogleAPIError as e:
            raise InvalidCredentials(error=e)
        except GoogleAuthError as e:
            raise InvalidCredentials(error=e)
        except AsciiError as e:
            raise InvalidCredentials(error=e)
        except PyAsn1Error as e:
            raise InvalidCredentials(error=e)

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("Method unavailable for Google's Gemini models.")

    def count_tokens(self, text: str) -> int:
        """
        Count the text tokens using the Vertex AI API.
        """
        response = self.model.count_tokens(text)
        return response.total_tokens

    def count_chatml_tokens(
        self, messages: Chat, functions: Optional[Sequence[Mapping]] = None
    ) -> int:
        """
        Handle the counting of tokens for Chat messages and functions.
        """
        tokens: int = 0
        if functions:
            tokens += self.count_functions_tokens(functions)
        messages_tokens: Sequence[int] = [
            (self.count_message_tokens(message)) for message in messages
        ]
        return tokens + sum(messages_tokens)

    def count_message_tokens(
        self, message: ChatMLMessage | FunctionCallChatMLMessage | FunctionChatMLMessage
    ) -> int:
        """
        Count the tokens for a single message.
        """
        if "function_call" in message:
            return (
                sum(
                    [
                        self.model.count_tokens(str(k)).total_tokens
                        + self.model.count_tokens(str(v)).total_tokens
                        for k, v in json.loads(
                            message["function_call"]["arguments"]
                        ).items()
                    ]
                )
                + self.model.count_tokens(message["function_call"]["name"]).total_tokens
            )
        elif message["role"] == "function":
            return (
                self.count_tokens(message["content"])
                + self.count_tokens(message["name"])
                + 1
            )
        else:
            return self.count_tokens(message["content"])

    def count_functions_tokens(self, functions: Sequence[Mapping]) -> int:
        """
        Count the tokens for a list of gemini functions
        """
        tokens = 0

        def _count_parameters(parameters: Mapping[str, str | Mapping[str, str]]) -> int:
            tokens = 0
            for param, param_value in list(parameters.items()):
                if param == "description":
                    if isinstance(param_value, str):
                        tokens += self.model.count_tokens(param_value).total_tokens
                if isinstance(param_value, dict):
                    tokens += _count_parameters(param_value)
                if param == "properties":
                    if isinstance(param_value, dict):
                        tokens += sum(
                            [
                                self.model.count_tokens(k).total_tokens
                                for k in list(param_value.keys())
                            ]
                        )
            return tokens

        for tool in functions:
            tokens += (
                self.model.count_tokens(tool["name"]).total_tokens
                + self.model.count_tokens(tool["description"]).total_tokens
                + sum(
                    [
                        self.model.count_tokens(param_value_i).total_tokens
                        for param_value_i in tool["parameters"]["required"]
                    ]
                )
                + _count_parameters(tool["parameters"])
            )
        return tokens
