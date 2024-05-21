import pytest
from totokenizers.google import GoogleTokenizer
from totokenizers.schemas import (
    ChatMLMessage,
    FunctionCall,
    FunctionCallChatMLMessage,
    FunctionChatMLMessage,
)


@pytest.fixture(scope="module")
def chatml_messages():
    return [
        ChatMLMessage(
            content="You are a bot.",
            name="system",
            role="system",
        ),
        ChatMLMessage(
            content="hello bot",
            name="user",
            role="user",
        ),
        ChatMLMessage(
            content="I am Skynet.",
            name="skynet",
            role="assistant",
        ),
    ]


def test_gpp4_o(chatml_messages):
    tokenizer = GoogleTokenizer(model_name="gemini-pro")

    count_tokens = tokenizer.count_chatml_tokens(chatml_messages)
    assert count_tokens == 34

    count_tokens = tokenizer.count_tokens("hello world")
    assert count_tokens == 2

    # TODO: count function call tokens
