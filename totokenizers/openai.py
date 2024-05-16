import logging
from typing import Literal, Mapping, Optional, Sequence

import tiktoken

from .errors import ModelNotFound, ModelNotSupported
from .jsonschema_formatter import FunctionJSONSchema
from .schemas import (
    Chat,
    ChatMLMessage,
    FunctionCallChatMLMessage,
    FunctionChatMLMessage,
)

logger = logging.getLogger("totokenizers")


class OpenAITokenizer:
    funcion_header = "\n".join(
        [
            "# Tools",
            "",
            "## functions",
            "",
            "namespace functions {",
            "",
            "} // namespace functions",
        ]
    )

    def __init__(
        self,
        model_name: str,
    ):
        self.model = model_name
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            raise ModelNotFound(model_name)
        self._init_model_params()

    def _init_model_params(self):
        """https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb"""
        if self.model in (
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-davinci-003",
            "gpt-3.5-turbo-instruct",
        ):
            self.count_chatml_tokens = NotImplementedError
            self.count_functions_tokens = NotImplementedError
            self.count_message_tokens = NotImplementedError
            return

        if self.model in {
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-3.5-turbo-1106",
            "gpt-4-1106-preview",
            "gpt-4-0125-preview",
            "gpt-4-turbo-preview",  # currently points to gpt-4-0125-preview
            "gpt-4-32k-0314",
            "gpt-4-32k-0613",
            "gpt-4-turbo-2024-04-09",
            "gpt-3.5-turbo-0125",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o-2024-05-13",
            "gpt-4o",
            "gpt-3.5-turbo",
        }:
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        elif self.model == "gpt-3.5-turbo-0301":
            self.tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            self.tokens_per_name = -1  # if there's a name, the role is omitted
        else:
            raise ModelNotSupported(self.model)

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def count_chatml_tokens(
        self, messages: Chat, functions: Optional[Sequence[Mapping]] = None
    ) -> int:
        num_tokens = sum(map(self.count_message_tokens, messages))
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        if functions:
            if messages[0]["role"] == "system":
                num_tokens -= (
                    1  # I believe a newline gets removed somewhere for somereason
                )
            else:
                num_tokens += self.tokens_per_message
            num_tokens += self.count_functions_tokens(functions)
        return num_tokens

    def count_message_tokens(
        self, message: ChatMLMessage | FunctionCallChatMLMessage | FunctionChatMLMessage
    ) -> int:
        """https://github.com/openai/openai-python/blob/main/chatml.md"""
        num_tokens = self.tokens_per_message
        if message["role"] == "function":
            num_tokens += (
                self.count_tokens(message["content"])
                + self.count_tokens(message["name"])
                + self.count_tokens(message["role"])
                - 1  # omission of a delimiter?
            )
        elif "function_call" in message:
            # https://github.com/forestwanglin/openai-java/blob/308a3423d34905bd28aca976fd0f2fa030f9a3a1/jtokkit/src/main/java/xyz/felh/openai/jtokkit/utils/TikTokenUtils.java#L202-L205
            num_tokens += (
                self.count_tokens(message["function_call"]["name"])
                + self.count_tokens(
                    message["function_call"]["arguments"]
                )  # TODO: what if there are no arguments?
                + self.count_tokens(message["role"])
                + 3  # I believe this is due to delimiter tokens being added
            )
        else:
            num_tokens += self.count_tokens(message["content"]) + self.count_tokens(
                message["role"]
            )
            if "name" in message:
                num_tokens += self.tokens_per_name + self.count_tokens(message["name"])
        return num_tokens

    def count_functions_tokens(self, functions: list[dict]) -> int:
        num_tokens = len(self.encode(self.funcion_header))
        num_tokens += len(self.encode(FunctionJSONSchema(functions).to_typescript()))
        return num_tokens
