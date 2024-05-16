from typing import Literal, Sequence

import vertexai
from vertexai.preview.generative_models import GenerativeModel

from ..schemas import ChatMLMessage, FunctionCallChatMLMessage, FunctionChatMLMessage


class GeminiTokenizer:
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

    def __init__(
        self,
        model_name: Literal[
            "gemini-pro",
            "gemini-pro-vision",
        ],
        project_id: str,
        location: str,
    ):
        # Initialize the Vertex AI API (gets Google credentialsl as well)
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("Method unavailable for Google's Gemini models.")

    def count_tokens(self, text: str) -> int:
        response = self.model.count_tokens(text)
        return response.total_tokens

    def count_chatml_tokens(self, messages: Sequence[ChatMLMessage]) -> int:

        return sum([self.count_message_tokens(message) for message in messages])

    def count_message_tokens(
        self, message: ChatMLMessage | FunctionCallChatMLMessage | FunctionChatMLMessage
    ) -> int:
        if isinstance(message, ChatMLMessage):
            return self.count_tokens(message)
        elif isinstance(message, FunctionCallChatMLMessage):
            return self.count_functions_tokens(message.functions)
        elif isinstance(message, FunctionChatMLMessage):
            return self.count_functions_tokens(message.functions)
        else:
            raise TypeError("Invalid message type.")

    def count_functions_tokens(self, functions: list[dict]) -> int: ...


"""
Without system
why is sky blue? 
Tokens: 5

****************

"""
