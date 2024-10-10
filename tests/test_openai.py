import pytest
from melting_schemas.completion.buffered_ml_messages import ToolMLMessage

from totokenizers.openai import ChatMLMessage, OpenAITokenizer


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


@pytest.fixture(scope="module")
def tool_call_messages():
    return [
        ToolMLMessage(
            tool_id="abcdef_0",
            content="Rainy",
            name="weather",
            role="tool",
        ),
        ToolMLMessage(
            tool_id="abcdef_1",
            content="Sunny",
            name="weather",
            role="tool",
        ),
        ToolMLMessage(
            tool_id="abcdef_2",
            content="Cloudy",
            name="weather",
            role="tool",
        ),
    ]


def test_gpp4_o(chatml_messages, tool_call_messages):
    tokenizer = OpenAITokenizer(model_name="gpt-4o-2024-05-13")

    count_tokens = tokenizer.count_chatml_tokens(chatml_messages)
    assert count_tokens == 34

    count_tokens = tokenizer.count_tokens("hello world")
    assert count_tokens == 2

    count_tokens = tokenizer.count_chatml_tokens(tool_call_messages)
    assert count_tokens == 32
