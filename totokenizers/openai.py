import logging
from typing import Mapping, Optional, Sequence

import tiktoken

from .errors import ModelNotFound, ModelNotSupported
from .jsonschema_formatter import FunctionJSONSchema
from .schemas import (
    Chat,
    ChatImageContent,
    ChatMLMessage,
    ChatTextContent,
    FunctionCallChatMLMessage,
    FunctionChatMLMessage,
    Tool,
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
            if (
                model_name
                == "ft:gpt-4o-2024-08-06:osf-digital:revenue-cloud-4o:A5s5vXgB"
            ):
                self.encoder = tiktoken.encoding_for_model("gpt-4o")
            else:
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
            self.count_chatml_tokens = NotImplementedError  # type: ignore
            self.count_functions_tokens = NotImplementedError  # type: ignore
            self.count_message_tokens = NotImplementedError  # type: ignore
            return

        if self.model in {
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k-0613",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo",  # points to 0125
            "gpt-4-0125-preview",
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-4-1106-preview",
            "gpt-4-32k-0314",
            "gpt-4-32k-0613",
            "gpt-4-32k",  # points to 0613
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo-preview",  # points to gpt-4-0125-preview
            "gpt-4-turbo",  # points to 2024-04-09
            "gpt-4",  # points to 0613
            "gpt-4o-2024-05-13",
            "gpt-4o",  # points to 2024-05-13
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-mini",  # points to 2024-07-18
            "ft:gpt-4o-2024-08-06:osf-digital:revenue-cloud-4o:A5s5vXgB",
        }:
            self.tokens_per_message = 3
            self.tokens_per_name = 1
            self.tokens_per_image = 85
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
            num_tokens += self.count_content_tokens(content=message["content"])
            num_tokens += self.count_tokens(message["role"])
            if "name" in message:
                num_tokens += self.tokens_per_name + self.count_tokens(message["name"])
        return num_tokens

    def count_content_tokens(
        self, content: str | list[ChatTextContent | ChatImageContent]
    ) -> int:
        if isinstance(content, str):
            return self.count_tokens(content)

        num_tokens = 0
        for item in content:
            match item:
                case {"type": "text"}:
                    num_tokens += self.count_tokens(item["text"])
                case {"type": "image_url"}:
                    num_tokens += self.tokens_per_image
                case _:
                    raise TypeError(f"Unknown content type: {type(item)}")
        return num_tokens

    def count_functions_tokens(self, functions: list[dict]) -> int:
        num_tokens = len(self.encode(self.funcion_header))
        num_tokens += len(self.encode(FunctionJSONSchema(functions).to_typescript()))
        return num_tokens

    def count_tools_tokens(self, tools: Tool) -> int:
        model = self.model
        if "openai/" in model:
            model = model.split("/")[-1]

        """Calculate the total number of tokens for tools and messages."""
        func_init = 0
        prop_init = 0
        prop_key = 0
        enum_init = 0
        enum_item = 0
        func_end = 0

        if model in ["gpt-4o", "gpt-4o-mini"]:
            func_init = 7
            prop_init = 3
            prop_key = 3
            enum_init = -3
            enum_item = 3
            func_end = 12
        elif model in ["gpt-3.5-turbo", "gpt-4"]:
            func_init = 10
            prop_init = 3
            prop_key = 3
            enum_init = -3
            enum_item = 3
            func_end = 12
        else:
            logger.error(f"count_tools_tokens() is not implemented for model {model}.")
            raise NotImplementedError(
                f"count_tools_tokens() is not implemented for model {model}."
            )

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Model {model} not found. Using o200k_base encoding.")
            encoding = tiktoken.get_encoding("o200k_base")

        func_token_count = 0
        for tool in tools:
            if tool["content"] is not None:  # is ToolMLMessage         # type:ignore
                func_token_count += self.count_tokens(tool["content"])  # type:ignore
                if "name" in tool:
                    func_token_count += self.count_tokens(tool["name"])  # type:ignore
            else:  # is ToolCallMLMessage
                for call in tool["tool_calls"]:  # type:ignore
                    func_token_count += (
                        func_init  # Add tokens for start of each function
                    )
                    function = call["function"]
                    f_name = function["name"]
                    f_args = function["arguments"]
                    line = f"{f_name}:{f_args}"

                    func_token_count += len(
                        encoding.encode(line)
                    )  # Add tokens for function name and arguments

                    try:
                        for arg in function["arguments"].split(","):
                            key, prop = arg.split(":")
                            func_token_count += prop_key
                            func_token_count += len(encoding.encode(prop))
                            if "enum" in prop:
                                func_token_count += enum_init
                                func_token_count += enum_item
                        func_token_count += prop_init
                    except ValueError:
                        pass  # No arguments

        return func_token_count + func_end
