from typing import Literal, Sequence, Optional, Mapping

import vertexai
from vertexai.preview.generative_models import GenerativeModel
import json
from ..schemas import (
    ChatMLMessage,
    FunctionCallChatMLMessage,
    FunctionChatMLMessage,
    Chat,
)
from vertexai.preview.generative_models import (
    FunctionDeclaration,
    GenerativeModel,
    Part,
    Tool,
    Content,
)


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

    def __init__(
        self,
        model_name: Literal["gemini-pro", "gemini-pro-vision"],
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

    @classmethod
    def _set_tool_from_json(cls, tools: Sequence[dict]) -> None:
        """creates a tool object from each tool in the tools file. Necessary for the Vertex API."""
        return [
            Tool(function_declarations=[FunctionDeclaration(**tool) for tool in tools])
        ]

    @classmethod
    def _translate_chat_ml_message_to_gemini(cls, message: ChatMLMessage):
        return Content(
            **{
                "role": message["role"],
                "parts": [Part.from_text(message["content"])],
            }
        )

    @classmethod
    def _translate_function_call_chat_ml_message_to_gemini(
        cls, messages: FunctionCallChatMLMessage
    ):
        messages["function_call"]["args"] = json.loads(
            messages["function_call"]["arguments"]
        )
        messages["function_call"].pop("arguments")
        messages.pop("role")
        return Content(
            role="model",
            parts=[Part.from_dict(messages)],
        )

    @classmethod
    def _translate_function_chat_ml_message_to_gemini(
        cls, message: FunctionChatMLMessage
    ):
        return Content(
            parts=[
                Part.from_function_response(
                    name=message["name"], response={"content": message["content"]}
                )
            ],
        )

    def count_chatml_tokens(
        self, messages: Chat, functions: Optional[Sequence[Mapping]] = None
    ) -> int:

        return sum(
            [
                self.count_tokens(message["parts"][0]["text"])
                for message in messages
                if "parts"
            ]
        )

    def count_message_tokens(
        self, message: ChatMLMessage | FunctionCallChatMLMessage | FunctionChatMLMessage
    ) -> int:
        if "function_call" in message:
            return (
                sum(
                    [
                        self.model.count_tokens(k).total_tokens
                        + self.model.count_tokens(v).total_tokens
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

    def count_functions_tokens(self, functions: list[dict]) -> int:

        messagges = self._translate_chat_ml_message_to_gemini(functions["messages"])
        tools = functions["functions"]
        tokens = 0

        def _count_parameters(parameters: dict) -> int:
            tokens = 0
            for param, param_value in list(parameters.items()):
                if param == "description":
                    if isinstance(param_value, str):
                        tokens += self.model.count_tokens(param_value).total_tokens
                if isinstance(param_value, dict):
                    tokens += _count_parameters(param_value)
                if param == "properties":
                    tokens += sum(
                        [
                            self.model.count_tokens(k).total_tokens
                            for k in list(param_value.keys())
                        ]
                    )
            return tokens

        for tool in tools:
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
        tokens += self.count_chatml_tokens(messagges)
        return tokens
