from typing import Literal, NotRequired, Sequence, TypedDict


class ChatTextContent(TypedDict):
    type: Literal["text"]
    text: str


class ImageURL(TypedDict):
    url: str
    detail: Literal["low", "high", "auto"]


class ChatImageContent(TypedDict):
    type: Literal["image_url"]
    image_url: ImageURL


class ChatMLMessage(TypedDict):
    content: str | list[ChatTextContent | ChatImageContent]
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
