from typing import TypedDict, NotRequired, Literal, Sequence


class ChatMLMessage(TypedDict):
    content: str
    name: NotRequired[str]
    role: Literal["user", "assistant", "system"]


class FunctionCall(TypedDict):
    name: str
    arguments: str


class FunctionCallChatMLMessage(TypedDict):
    content: None
    function_call: FunctionCall
    role: Literal["assistant"]


class FunctionChatMLMessage(TypedDict):
    content: str
    name: str
    role: Literal["function"]


Chat = Sequence[ChatMLMessage | FunctionCallChatMLMessage | FunctionChatMLMessage]
