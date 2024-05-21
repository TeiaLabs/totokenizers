import pytest

from totokenizers.google import GoogleTokenizer
from totokenizers.factories import Totokenizer
from totokenizers.schemas import (
    ChatMLMessage,
    FunctionCall,
    FunctionCallChatMLMessage,
    FunctionChatMLMessage,
)


@pytest.fixture(scope="module")
def model_name():
    return "gemini-1.0-pro-002"


@pytest.fixture(scope="module")
def model_tag(model_name: str):
    return f"google/{model_name}"


def test_count_tokens_gem(model_tag: str):
    tokenizer = Totokenizer.from_model(model_tag)
    message = "hello world"
    assert tokenizer.count_tokens(message) == 2


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
def function_call_chat_ml_message():
    return FunctionCallChatMLMessage(
        **{
            "function_call": FunctionCall(
                **{
                    "name": "get_current_weather",
                    "arguments": '{"location": "Boston, MA"}',
                }
            ),
            "role": "assistant",
        }
    )


@pytest.fixture(scope="module")
def function_chat_ml_message():
    return FunctionChatMLMessage(
        **{
            "content": '{"temperature": "22", "unit": "celsius", "description": "Sunny"}',
            "name": "get_current_weather",
            "role": "function",
        }
    )


@pytest.fixture(scope="module")
def tools():
    return [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ]


# Done
def test_gemini_chatml_messages(chatml_messages):
    tokenizer = GoogleTokenizer("gemini-pro")

    count_tokens = tokenizer.count_chatml_tokens(chatml_messages)
    assert count_tokens == 12

    count_tokens = tokenizer.count_tokens("hello world")
    assert count_tokens == 2


# Done
def test_gemini_function_call_chat_ml_message(function_call_chat_ml_message):
    tokenizer = GoogleTokenizer(model_name="gemini-pro")

    count_tokens = tokenizer.count_message_tokens(function_call_chat_ml_message)
    assert count_tokens == 9


# Done
def test_gemini_function_chat_ml_message(function_chat_ml_message):
    tokenizer = GoogleTokenizer(model_name="gemini-pro")

    count_tokens = tokenizer.count_message_tokens(function_chat_ml_message)
    assert count_tokens == 26


# Done
def test_gemini_tools(tools):
    tokenizer = GoogleTokenizer(model_name="gemini-pro")
    count_tokens = tokenizer.count_functions_tokens(tools)
    assert count_tokens == 28
